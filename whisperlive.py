import argparse
import os
import queue
import threading
import time
import traceback
import warnings
import wave
from contextlib import contextmanager
from datetime import datetime, timedelta

import numpy as np
import pyaudio
import torch
from dotenv import load_dotenv

# Suppress specific deprecation warnings from external libraries
warnings.filterwarnings("ignore", message=".*torchaudio._backend.list_audio_backends.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.custom_fwd.*")
warnings.filterwarnings(
    "ignore", message=".*huggingface_hub.*cache-system uses symlinks.*"
)
warnings.filterwarnings(
    "ignore", message=".*Module 'speechbrain.pretrained' was deprecated.*"
)

# Set environment variable to disable HuggingFace symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from pyannote.audio import Pipeline
from pyannote.core import Segment
from scipy.spatial.distance import cosine
from speechbrain.inference import SpeakerRecognition
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class WhisperLiveTranscription:
    def __init__(
        self,
        model_id="openai/whisper-large-v3-turbo",
        language="french",
        similarity_threshold=0.60,
        mode="subtitle",
    ):
        print("Launching WhisperLiveTranscription...")

        # Store configuration
        self.model_id = model_id
        self.language = language
        self.mode = mode
        self.max_merge_gap = 1.5  # Max silence in seconds between segments to merge

        # Load environment variables
        load_dotenv()
        hf_token = os.getenv("HUGGING_FACE_TOKEN")

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load Whisper model
        print("Loading Whisper model...")
        model = WhisperForConditionalGeneration.from_pretrained(model_id)
        if self.device.type == "cuda":
            model = model.half()
        self.model = model.to(self.device)
        self.processor = WhisperProcessor.from_pretrained(model_id)

        # Load diarization pipeline
        print("Loading diarization model...")
        self._load_diarization_pipeline(hf_token)

        # Load speaker embedding model with Windows compatibility
        print("Loading speaker embedding model...")
        self._load_speaker_embedding_model()

        # --- OPTIMIZATION: Compile models ---
        if self.device.type == "cuda":
            print("Compiling models for performance...")
            self.model = torch.compile(self.model, mode="max-autotune", fullgraph=True)
            self.diarization_pipeline.model = torch.compile(
                self.diarization_pipeline.model, mode="max-autotune", fullgraph=True
            )
            self.embedding_model.mods = torch.compile(
                self.embedding_model.mods, mode="max-autotune", fullgraph=True
            )
            print("Models compiled.")
        # ------------------------------------

        print("All models loaded successfully.")

        # Speaker registry for re-identification
        self.speaker_registry = {}
        self.next_speaker_id = 1
        self.similarity_threshold = similarity_threshold
        print(f"DEBUG: Configured similarity threshold: {self.similarity_threshold}")

        # Audio configuration
        self._setup_audio_config()

        # Initialize buffers and queues
        self._setup_buffers_and_queues()

        # Runtime state
        self.chunk_timestamps = {}
        self.chunk_counter = 0
        self.is_running = False
        self.start_time = None

        # Output file - Default name for live mode
        self.filename = (
            f"transcription_{self.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        self._init_output_file()

        # Pre-calculate forced_decoder_ids - Updated to use task and language parameters
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.language, task="transcribe"
        )

        # Overlap management
        self.last_transcription = ""
        self.last_timestamp = None

    def _load_diarization_pipeline(self, hf_token):
        """Load the diarization pipeline with proper error handling."""
        try:
            # Updated to use token parameter instead of use_auth_token (deprecated)
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token if hf_token else True,
            )
            if pipeline is None:
                raise RuntimeError("Pipeline.from_pretrained returned None.")

            if self.device.type == "cuda":
                pipeline.model = pipeline.model.half()
            self.diarization_pipeline = pipeline.to(self.device)
        except AttributeError as e:
            if "'NoneType' object has no attribute 'eval'" in str(e):
                raise RuntimeError(
                    "Failed to load diarization pipeline. Please check your "
                    "HUGGING_FACE_TOKEN and ensure you have accepted the user "
                    "agreement for pyannote models on Hugging Face Hub."
                ) from e
            else:
                raise e

    def _load_speaker_embedding_model(self):
        """Load speaker embedding model with Windows symlink workaround."""
        # Set environment variable to disable symlinks for SpeechBrain
        os.environ["SPEECHBRAIN_CACHE_DIR"] = os.path.join(
            os.getcwd(), "speechbrain_cache"
        )

        try:
            # Try loading without custom savedir first (uses default cache)
            embedding_model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb"
            )
            if self.device.type == "cuda":
                embedding_model = embedding_model.half()
            self.embedding_model = embedding_model.to(self.device)
        except OSError as e:
            if "privilège nécessaire" in str(e) or "WinError 1314" in str(e):
                # Windows symlink issue - use copy strategy
                print("Windows symlink issue detected. Using copy strategy...")
                import speechbrain as sb

                # Temporarily change the fetching strategy
                original_strategy = getattr(sb.utils.fetching, "LOCAL_STRATEGY", None)
                sb.utils.fetching.LOCAL_STRATEGY = sb.utils.fetching.CopyStrategy()

                try:
                    embedding_model = SpeakerRecognition.from_hparams(
                        source="speechbrain/spkrec-ecapa-voxceleb"
                    )
                    if self.device.type == "cuda":
                        embedding_model = embedding_model.half()
                    self.embedding_model = embedding_model.to(self.device)
                finally:
                    # Restore original strategy
                    if original_strategy:
                        sb.utils.fetching.LOCAL_STRATEGY = original_strategy
            else:
                raise e

    def _setup_audio_config(self):
        """Setup audio configuration constants."""
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024
        self.SILENCE_THRESHOLD = 0.01  # Increased threshold to reduce false positives
        self.SILENCE_DURATION = 3.0  # Increased duration for more stable chunks
        self.MAX_BUFFER_DURATION = 15.0  # Increased buffer duration
        self.OVERLAP_DURATION = 2.0
        self.SOFT_CUT_WINDOW_DURATION = 3.0  # Increased window

        # Pre-calculate constants
        self.required_silent_chunks = int(
            self.SILENCE_DURATION * self.RATE / self.CHUNK
        )
        self.max_buffer_samples = int(self.MAX_BUFFER_DURATION * self.RATE)
        self.max_soft_cut_buffer_samples = int(
            (self.MAX_BUFFER_DURATION + self.SOFT_CUT_WINDOW_DURATION) * self.RATE
        )
        self.overlap_samples = int(self.OVERLAP_DURATION * self.RATE)

    def _setup_buffers_and_queues(self):
        """Initialize buffers and queues."""
        self.audio_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=3)
        self.transcription_queue = queue.Queue()
        self._reset_buffers()

    def _reset_buffers(self):
        """Reset audio buffers to empty state."""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.full_audio_buffer_list = []
        self.silence_counter = 0
        self.overlap_buffer = np.array([], dtype=np.float32)

    def _init_output_file(self):
        """Initialize the output transcription file."""
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write(
                f"# Transcription started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

    def _get_speaker_id(self, embedding):
        """Get or assign speaker ID based on embedding similarity."""
        if not self.speaker_registry:
            # First speaker
            speaker_id = f"Speaker_{self.next_speaker_id}"
            self.speaker_registry[speaker_id] = embedding
            self.next_speaker_id += 1
            print(f"DEBUG: Created new speaker {speaker_id} (first speaker)")
            return speaker_id

        # Compare with existing speakers
        best_match = None
        best_similarity = 0

        for speaker_id, stored_embedding in self.speaker_registry.items():
            similarity = 1 - cosine(embedding.flatten(), stored_embedding.flatten())
            print(f"DEBUG: Comparing with {speaker_id}: similarity = {similarity:.4f}")
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = speaker_id

        print(
            f"DEBUG: Best match: {best_match} with similarity {best_similarity:.4f}, threshold: {self.similarity_threshold}"
        )

        if best_similarity > self.similarity_threshold:
            # Update embedding with running average
            self.speaker_registry[best_match] = (
                self.speaker_registry[best_match] * 0.7 + embedding * 0.3
            )
            print(f"DEBUG: Assigned to existing speaker {best_match}")
            return best_match
        else:
            # New speaker
            speaker_id = f"Speaker_{self.next_speaker_id}"
            self.speaker_registry[speaker_id] = embedding
            self.next_speaker_id += 1
            print(f"DEBUG: Created new speaker {speaker_id} (similarity too low)")
            return speaker_id

    def get_transcription(self):
        """Get the next transcription result from the queue."""
        try:
            return self.transcription_queue.get_nowait()
        except queue.Empty:
            return None

    @contextmanager
    def _audio_stream_context(self, callback=None):
        """Context manager for audio stream handling."""
        p = pyaudio.PyAudio()
        try:
            if callback:
                stream = p.open(
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.RATE,
                    input=True,
                    frames_per_buffer=self.CHUNK,
                    stream_callback=callback,
                    start=False,
                )
            else:
                stream = None
            yield p, stream
        finally:
            if "stream" in locals() and stream:
                stream.stop_stream()
                stream.close()
            p.terminate()

    def _transcribe_segment(self, audio_data):
        """Enhanced transcription with better preprocessing."""
        if len(audio_data) < self.RATE * 0.5:
            return ""

        # Audio preprocessing
        audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-8)

        # Simple noise reduction
        if len(audio_data) > 3:
            kernel = np.ones(3) / 3
            audio_data = np.convolve(audio_data, kernel, mode="same")

        processed_input = self.processor(
            audio_data, sampling_rate=self.RATE, return_tensors="pt"
        )
        features = processed_input.input_features.to(self.device)

        attention_mask = torch.ones(
            features.shape[:2], dtype=torch.long, device=self.device
        )

        # Updated generation parameters to avoid deprecated forced_decoder_ids
        predicted_ids = self.model.generate(
            features,
            attention_mask=attention_mask,
            language=self.language,
            task="transcribe",
            max_length=448,
            num_beams=2,
            temperature=0.0,
            do_sample=False,
            suppress_tokens=[],
        )

        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        return transcription

    def _add_audio_segment(self, chunk_id, audio_data, start_time, end_time):
        """Add audio segment to result queue."""
        self.chunk_timestamps[chunk_id] = {"start": start_time, "end": end_time}
        self.result_queue.put((chunk_id, audio_data.copy()))
        self.chunk_counter += 1
        return self.chunk_counter - 1

    def _calculate_video_timestamp(self, elapsed_seconds):
        """Calculate timestamp in HH:MM:SS format."""
        if elapsed_seconds < 0:
            elapsed_seconds = 0

        total_seconds = int(elapsed_seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _process_transcription(self, chunk_id, audio_data, is_final=False):
        """Process a chunk of audio: diarize, identify, and transcribe."""
        print(f"Processing segment {chunk_id}...")

        try:
            diarization_result = self.diarization_pipeline(
                {
                    "waveform": torch.from_numpy(audio_data).unsqueeze(0),
                    "sample_rate": self.RATE,
                }
            )

            # Count segments and unique speakers separately
            segments = list(diarization_result.itertracks(yield_label=True))
            unique_speakers = set(speaker_label for _, _, speaker_label in segments)

            print(
                f"Found {len(segments)} speech segments from {len(unique_speakers)} speakers in segment {chunk_id}"
            )

            # --- MODE SELECTION LOGIC ---
            if self.mode == "transcription":
                # --- TRANSCRIPTION MODE: Merge segments before transcribing ---
                merged_segments = self._merge_speaker_segments(
                    diarization_result, audio_data
                )
                for segment_info in merged_segments:
                    self._transcribe_and_display(segment_info)
            else:
                # --- SUBTITLE MODE: Transcribe each segment individually (original behavior) ---
                for turn, _, speaker_label in diarization_result.itertracks(
                    yield_label=True
                ):
                    speaker_audio_segment = audio_data[
                        int(turn.start * self.RATE) : int(turn.end * self.RATE)
                    ]
                    segment_info = {
                        "speaker_label": speaker_label,
                        "turn": turn,
                        "audio_segment": speaker_audio_segment,
                    }
                    self._transcribe_and_display(segment_info)

        except Exception:
            print(f"Error during transcription of segment {chunk_id}:")
            traceback.print_exc()
        finally:
            if chunk_id in self.chunk_timestamps:
                del self.chunk_timestamps[chunk_id]

    def _merge_speaker_segments(self, diarization_result, audio_data):
        """
        Two-step approach: First identify speakers for each segment, then merge consecutive segments.
        This is more robust than trying to merge and identify simultaneously.
        """
        merged_segments = []
        timeline = diarization_result.get_timeline().support()
        if not timeline:
            return merged_segments

        # Get the list of turns sorted by start time
        turns = list(diarization_result.itertracks(yield_label=True))
        if not turns:
            return merged_segments

        # --- STEP 1: IDENTIFY SPEAKERS FOR EACH SEGMENT ---
        identified_segments = []
        for turn, _, pyannote_speaker in turns:
            # Extract audio for this individual segment
            start_samples = int(turn.start * self.RATE)
            end_samples = int(turn.end * self.RATE)
            audio_segment = audio_data[start_samples:end_samples]

            # Skip segments that are too short for reliable speaker identification
            if len(audio_segment) < self.RATE * 0.5:  # Minimum 0.5 seconds
                print(
                    f"DEBUG: Skipping short segment {turn.start:.2f}s-{turn.end:.2f}s"
                )
                continue

            # Get speaker embedding and identify
            try:
                embedding = self.embedding_model.encode_batch(
                    torch.from_numpy(audio_segment).unsqueeze(0).to(self.device)
                )
                assigned_speaker = self._get_speaker_id(embedding.squeeze())

                identified_segments.append(
                    {
                        "turn": turn,
                        "pyannote_speaker": pyannote_speaker,
                        "assigned_speaker": assigned_speaker,
                        "audio_segment": audio_segment,
                    }
                )
                print(
                    f"DEBUG: Identified segment {turn.start:.2f}s-{turn.end:.2f}s as {assigned_speaker} (pyannote: {pyannote_speaker})"
                )

            except Exception as e:
                print(
                    f"DEBUG: Failed to identify speaker for segment {turn.start:.2f}s-{turn.end:.2f}s: {e}"
                )
                continue

        # --- STEP 2: MERGE CONSECUTIVE SEGMENTS FROM THE SAME IDENTIFIED SPEAKER ---
        if not identified_segments:
            return merged_segments

        # --- IMPROVED GROUPING LOGIC WITH SPEAKER INTERRUPTION DETECTION ---
        merged_groups = []
        if identified_segments:
            current_group = [identified_segments[0]]

            for i in range(1, len(identified_segments)):
                current_segment = identified_segments[i]
                last_segment_in_group = current_group[-1]

                same_speaker = (
                    current_segment["assigned_speaker"]
                    == last_segment_in_group["assigned_speaker"]
                )
                time_gap = (
                    current_segment["turn"].start - last_segment_in_group["turn"].end
                )

                # --- SIMPLIFIED LOGIC FOR TRANSCRIPTION MODE ---
                if self.mode == "transcription":
                    # In transcription mode, merge all consecutive segments from same speaker
                    # regardless of gap duration
                    should_merge = same_speaker
                    gap_reason = f"gap: {time_gap:.2f}s (ignored in transcription mode)"
                else:
                    # In subtitle mode, respect the time gap limit
                    should_merge = same_speaker and time_gap < self.max_merge_gap
                    gap_reason = f"gap: {time_gap:.2f}s (limit: {self.max_merge_gap}s)"

                print(
                    f"DEBUG: Comparing segment {current_segment['turn'].start:.2f}s ({current_segment['assigned_speaker']}) with last in group {last_segment_in_group['turn'].end:.2f}s ({last_segment_in_group['assigned_speaker']}) - {gap_reason}"
                )

                if should_merge:
                    # Merge: add to current group
                    current_group.append(current_segment)
                    print(
                        f"DEBUG: ✓ Merging segment {current_segment['turn'].start:.2f}s-{current_segment['turn'].end:.2f}s with group (gap: {time_gap:.2f}s)"
                    )
                else:
                    # Finalize current group and start new one
                    merged_groups.append(current_group)
                    reason = (
                        "different speaker"
                        if not same_speaker
                        else f"gap too large ({time_gap:.2f}s)"
                    )
                    print(
                        f"DEBUG: ✗ Finalizing group with {len(current_group)} segments for {current_group[0]['assigned_speaker']} ({reason})"
                    )
                    current_group = [current_segment]
                    print(
                        f"DEBUG: Starting new group for {current_segment['assigned_speaker']} at {current_segment['turn'].start:.2f}s"
                    )

            # Add the last group
            merged_groups.append(current_group)
            print(
                f"DEBUG: Final group with {len(current_group)} segments for {current_group[0]['assigned_speaker']}"
            )
        # ----------------------------------------------------------------

        # --- STEP 3: CREATE FINAL SEGMENTS WITH MERGED AUDIO ---
        for group_idx, group in enumerate(merged_groups):
            if not group:
                continue

            speaker_id = group[0]["assigned_speaker"]
            start_time = group[0]["turn"].start
            end_time = group[-1]["turn"].end

            # Concatenate audio from all segments in this group
            audio_chunks = [segment["audio_segment"] for segment in group]
            merged_audio = np.concatenate(audio_chunks)

            print(
                f"DEBUG: Group {group_idx + 1}: Created merged segment for {speaker_id}: {start_time:.2f}s-{end_time:.2f}s ({len(group)} segments, {len(merged_audio) / self.RATE:.2f}s audio)"
            )

            merged_segments.append(
                {
                    "speaker_label": speaker_id,  # Use the identified speaker, not pyannote label
                    "turn": Segment(start_time, end_time),
                    "audio_segment": merged_audio,
                }
            )

        print(
            f"DEBUG: Final result: {len(merged_groups)} groups -> {len(merged_segments)} merged segments"
        )
        return merged_segments

    def _transcribe_and_display(self, segment_info):
        """Helper function to handle transcription and display for a segment."""
        turn = segment_info["turn"]
        speaker_label = segment_info[
            "speaker_label"
        ]  # This is now the identified speaker ID
        speaker_audio_segment = segment_info["audio_segment"]

        # --- FILTER OUT SEGMENTS TOO SHORT FOR PROCESSING ---
        min_duration = 0.5  # Minimum 0.5 seconds
        min_samples = int(min_duration * self.RATE)

        if len(speaker_audio_segment) < min_samples:
            print(
                f"DEBUG: Skipping segment too short: {len(speaker_audio_segment)} samples ({len(speaker_audio_segment) / self.RATE:.2f}s)"
            )
            return
        # ---------------------------------------------------

        # --- SIMPLIFIED LOGGING (no embedding calculation needed here) ---
        print(f"DEBUG: Transcribing merged segment for {speaker_label}")
        print(
            f"DEBUG: Turn: {turn.start:.2f}s -> {turn.end:.2f}s (duration: {turn.end - turn.start:.2f}s)"
        )
        print(
            f"DEBUG: Audio segment length: {len(speaker_audio_segment)} samples ({len(speaker_audio_segment) / self.RATE:.2f}s)"
        )
        # ---------------------------------------------------------------

        # Transcribe (no need to identify speaker again, it's already done)
        transcription = self._transcribe_segment(speaker_audio_segment)
        if not transcription or transcription.isspace():
            print("DEBUG: Transcription was empty or whitespace only")
            return

        # Calculate timestamp
        chunk_start_time = self.chunk_timestamps.get(self.chunk_counter - 1, {}).get(
            "start", 0
        )
        elapsed_seconds = chunk_start_time + turn.start
        timestamp = self._calculate_video_timestamp(elapsed_seconds)

        # Format and save
        line = f"[{timestamp}][{speaker_label}] {transcription}\n"
        print(f"Live: {line.strip()}")
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(line)

    def transcribe_file(self, file_path):
        """
        Transcribe an entire audio file by chunking it and processing it
        like a live stream.
        """
        # Generate a descriptive filename for file transcription
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        model_name_safe = self.model_id.replace("/", "_")
        threshold_str = f"thresh{self.similarity_threshold:.2f}"
        self.filename = f"{base_name}_{self.mode}_{model_name_safe}_{threshold_str}.txt"
        self._init_output_file()

        print(f"Transcribing audio file: {file_path}")
        try:
            with wave.open(file_path, "rb") as wf:
                framerate = wf.getframerate()
                if framerate != self.RATE:
                    raise ValueError(
                        f"Unsupported sample rate: {framerate}. Please use {self.RATE} Hz."
                    )
                # ... (add other checks for channels, sampwidth if needed)
                audio_bytes = wf.readframes(wf.getnframes())

            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            print("Audio file loaded. Starting transcription process...")

            # 1. Start the transcription worker thread
            self.is_running = True
            self.transcribe_thread = threading.Thread(target=self._transcribe_audio)
            self.transcribe_thread.daemon = True
            self.transcribe_thread.start()

            # 2. Chunk the audio and add it to the processing queue
            chunk_duration_seconds = 15  # Process in 15-second chunks
            chunk_samples = int(chunk_duration_seconds * self.RATE)
            total_samples = len(audio_float32)

            for i in range(0, total_samples, chunk_samples):
                start_sample = i
                end_sample = i + chunk_samples
                chunk_data = audio_float32[start_sample:end_sample]

                if len(chunk_data) == 0:
                    continue

                start_time = start_sample / self.RATE
                end_time = end_sample / self.RATE

                # Use the existing method to add segments to the queue
                self._add_audio_segment(
                    self.chunk_counter, chunk_data, start_time, end_time
                )
                print(
                    f"Queued segment {self.chunk_counter - 1} ({start_time:.2f}s to {end_time:.2f}s)"
                )
                # Give the worker thread a moment to catch up
                time.sleep(0.1)

            # 3. Signal that we are done adding chunks
            self.is_running = False

            # 4. Wait for the worker thread to finish processing all chunks
            print("All segments queued. Waiting for final processing...")
            if hasattr(self, "transcribe_thread"):
                self.transcribe_thread.join()  # Remove timeout to wait as long as needed

            print(f"\nTranscription finished. Results saved to {self.filename}")

        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
        except Exception as e:
            print(f"An error occurred during file transcription: {e}")
            traceback.print_exc()

    def _transcribe_audio(self):
        """Audio transcription worker thread."""
        print("Audio transcription thread started")
        while self.is_running or not self.result_queue.empty():
            try:
                # Use a timeout to allow the loop to re-evaluate self.is_running
                chunk_id, audio_data = self.result_queue.get(timeout=0.5)
                self._process_transcription(
                    chunk_id, audio_data, is_final=not self.is_running
                )
            except queue.Empty:
                # This is expected when the queue is empty but self.is_running is still true
                continue
            except Exception:
                print("Transcription error:")
                traceback.print_exc()

    def _should_process_segment(self, audio_data):
        """Determine if audio segment should be processed."""
        if len(audio_data) == 0:
            return False

        energy = np.mean(audio_data**2)
        if energy < 1e-6:
            return False

        if len(audio_data) < self.RATE * 0.3:
            return False

        return True

    def _process_audio_stream(self, in_data, frame_count, time_info, status):
        """Audio stream callback with smart segmentation."""
        now = datetime.now()
        chunk = np.frombuffer(in_data, dtype=np.float32)
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
        self.full_audio_buffer_list.append(chunk)

        # Use RMS for better silence detection
        is_silent = np.sqrt(np.mean(chunk**2)) < self.SILENCE_THRESHOLD
        should_cut = False

        # Silence-based cutting with stricter requirements
        if is_silent:
            self.silence_counter += 1
            # Require sustained silence before cutting
            if self.silence_counter >= self.required_silent_chunks:
                should_cut = True
        else:
            self.silence_counter = 0

        # Buffer size-based cutting
        buffer_len = len(self.audio_buffer)
        if not should_cut and buffer_len > self.max_buffer_samples:
            if is_silent:
                print("Soft cut triggered")
                should_cut = True
            elif buffer_len >= self.max_soft_cut_buffer_samples:
                print("Hard cut triggered")
                should_cut = True

        # Only cut if we have sufficient audio content
        if should_cut and len(self.audio_buffer) > self.RATE * 2.0:  # Minimum 2 seconds
            segment_duration = len(self.audio_buffer) / self.RATE
            self._add_audio_segment(
                self.chunk_counter,
                self.audio_buffer,
                (now - timedelta(seconds=segment_duration)).timestamp(),
                now.timestamp(),
            )

            # Keep overlap for next segment
            self.overlap_buffer = self.audio_buffer[-self.overlap_samples :]
            self.audio_buffer = self.overlap_buffer.copy()
            self.silence_counter = 0

        return (in_data, pyaudio.paContinue)

    def start_recording(self):
        """Start audio recording and processing."""
        self.is_running = True
        self.start_time = datetime.now()
        self._reset_buffers()

        # Start transcription thread
        self.transcribe_thread = threading.Thread(target=self._transcribe_audio)
        self.transcribe_thread.daemon = True
        self.transcribe_thread.start()

        # Start audio stream
        self.audio_context = self._audio_stream_context(
            callback=self._process_audio_stream
        )
        self.pyaudio_instance, self.stream = self.audio_context.__enter__()

        print("Starting audio stream...")
        self.stream.start_stream()

    def stop_recording(self):
        """Stop recording and finalize transcriptions."""
        if not self.is_running:
            return

        print("\nStopping recording and finalizing transcriptions...")
        self.is_running = False

        # Stop audio stream
        if hasattr(self, "stream") and self.stream.is_active():
            self.stream.stop_stream()
        if hasattr(self, "audio_context"):
            self.audio_context.__exit__(None, None, None)

        # Process final buffer content
        if self.audio_buffer.size > self.overlap_samples:
            print("Processing final audio buffer...")
            now = datetime.now()
            segment_duration = len(self.audio_buffer) / self.RATE
            self._add_audio_segment(
                self.chunk_counter,
                self.audio_buffer,
                (now - timedelta(seconds=segment_duration)).timestamp(),
                now.timestamp(),
            )

        # Wait for the transcription thread to process all remaining items in the queue
        if hasattr(self, "transcribe_thread"):
            print("Waiting for transcription thread to finish...")
            self.transcribe_thread.join(timeout=30)  # Increased timeout for safety

        print(f"Transcription saved to {self.filename}")

    def _force_final_transcription(self):
        """Process remaining segments."""
        print("Processing remaining segments...")
        while not self.result_queue.empty():
            try:
                chunk_id, audio_data = self.result_queue.get_nowait()
                self._process_transcription(chunk_id, audio_data, is_final=True)
            except queue.Empty:
                break
            except Exception:
                print("Final transcription error:")
                traceback.print_exc()

    def save_audio(self):
        """Save recorded audio to WAV file."""
        if not self.full_audio_buffer_list:
            return

        self.full_audio_buffer = np.concatenate(self.full_audio_buffer_list)
        filename = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

        max_val = np.max(np.abs(self.full_audio_buffer))
        if max_val > 0:
            scaled = np.int16(self.full_audio_buffer / max_val * 32767.0)
        else:
            scaled = np.int16(self.full_audio_buffer)

        with wave.open(filename, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(self.RATE)
            wf.writeframes(scaled.tobytes())

        print(
            f"Audio saved ({len(self.full_audio_buffer) / self.RATE:.2f}s) to {filename}"
        )


if __name__ == "__main__":
    MODELS = [
        "openai/whisper-tiny",
        "openai/whisper-base",
        "openai/whisper-small",
        "openai/whisper-medium",
        "openai/whisper-large",
        "openai/whisper-large-v2",
        "openai/whisper-large-v3",
        "openai/whisper-large-v3-turbo",
    ]

    parser = argparse.ArgumentParser(
        description="Transcribe audio live from microphone or from a file."
    )
    parser.add_argument(
        "--language", type=str, default="french", help="Language for transcription."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=MODELS,
        default="openai/whisper-large-v3-turbo",
        help="Whisper model to use for transcription.",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a WAV audio file to transcribe. If not provided, runs in live mode.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Similarity threshold for speaker identification (0.0 to 1.0). Lower is less strict.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["subtitle", "transcription"],
        default="subtitle",
        help='Processing mode: "subtitle" for real-time display, "transcription" for merged, fluid text.',
    )

    args = parser.parse_args()

    try:
        transcriber = WhisperLiveTranscription(
            model_id=args.model,
            language=args.language,
            similarity_threshold=args.threshold,
            mode=args.mode,
        )

        if args.file:
            # File transcription mode
            transcriber.transcribe_file(args.file)
        else:
            # Live transcription mode
            print("Starting live transcription from microphone...")
            transcriber.start_recording()
            print("Recording in progress... Press Ctrl+C to stop")

            while True:
                result = transcriber.get_transcription()
                if result:
                    print(
                        f"[{result['timestamp']}][{result['speaker']}] {result['text']}"
                    )
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping process...")
        # Ensure transcriber object exists before trying to use it
        if "transcriber" in locals():
            # Signal the worker thread to stop its loop
            transcriber.is_running = False

            # Wait for the worker thread to finish its current task
            if (
                hasattr(transcriber, "transcribe_thread")
                and transcriber.transcribe_thread.is_alive()
            ):
                print("Waiting for transcription thread to finalize...")
                transcriber.transcribe_thread.join(timeout=10)

            # Specific cleanup for live mode (stream and audio saving)
            if not args.file:
                # The stream needs to be explicitly closed if it was running
                if hasattr(transcriber, "stream") and transcriber.stream.is_active():
                    transcriber.stream.stop_stream()
                if hasattr(transcriber, "audio_context"):
                    transcriber.audio_context.__exit__(None, None, None)
                transcriber.save_audio()

        print("Done.")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
