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
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.core import Segment
from scipy.spatial.distance import cosine
from speechbrain.inference import SpeakerRecognition

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


class WhisperLiveTranscription:
    def __init__(
        self,
        model_id="large-v3",
        language="fr",
        similarity_threshold=0.2,
        mode="transcription",
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
        print("Loading faster-whisper model...")
        compute_type = "float16" if self.device.type == "cuda" else "float32"
        self.model = WhisperModel(
            model_id, device=self.device.type, compute_type=compute_type
        )

        # Load diarization pipeline
        print("Loading diarization model...")
        self._load_diarization_pipeline(hf_token)

        # Load speaker embedding model with Windows compatibility
        print("Loading speaker embedding model...")
        self._load_speaker_embedding_model()

        # VAD model - lazy loading (will be loaded on first use)
        self.vad_model = None
        self.vad_utils = None

        print("All models loaded successfully.")

        # Speaker registry - only normalized embeddings
        self.speaker_embeddings_normalized = {}
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

        # State for transcription mode merging
        self.transcription_buffer = {"speaker": None, "timestamp": None, "text": ""}

    def _load_diarization_pipeline(self, hf_token):
        """Load the diarization pipeline with proper error handling."""
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
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
        os.environ["SPEECHBRAIN_CACHE_DIR"] = os.path.join(
            os.getcwd(), "speechbrain_cache"
        )

        try:
            embedding_model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb"
            )
            if self.device.type == "cuda":
                embedding_model = embedding_model.half()
            self.embedding_model = embedding_model.to(self.device)
        except OSError as e:
            if "privilège nécessaire" in str(e) or "WinError 1314" in str(e):
                print("Windows symlink issue detected. Using copy strategy...")
                import speechbrain as sb

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
        
        # MODE-SPECIFIC PARAMETERS
        if self.mode == "subtitle":
            # Subtitle mode: Send frequently to pyannote for immediate processing
            self.MAX_BUFFER_DURATION = 5.0  # Send every 5s for real-time response
        else:  # transcription mode
            # Transcription mode: Accumulate longer chunks for coherence
            self.MAX_BUFFER_DURATION = 75.0  # 1min15s - pyannote will cut at natural pauses

        # Pre-calculate constants
        self.max_buffer_samples = int(self.MAX_BUFFER_DURATION * self.RATE)

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

    def _init_output_file(self):
        """Initialize the output transcription file."""
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write(
                f"# Transcription started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

    def _ensure_vad_loaded(self):
        """Lazy loading of VAD model."""
        if self.vad_model is None:
            print("Loading VAD model...")
            self.vad_model, self.vad_utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", 
                model="silero_vad", 
                force_reload=False
            )

    def _get_speaker_id(self, embedding):
        """Get or assign speaker ID based on embedding similarity."""
        # Ensure we always work with NumPy arrays (not PyTorch tensors)
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
        
        # Force writable copy for NumPy 2.0 compatibility
        embedding = np.array(embedding, dtype=np.float32, copy=True)
        embedding_flat = embedding.flatten()
        
        # Normalize embedding
        norm = np.linalg.norm(embedding_flat)
        if norm == 0:
            print("DEBUG: Warning - zero norm embedding, skipping")
            return None
        embedding_norm = embedding_flat / norm
        
        if not self.speaker_embeddings_normalized:
            speaker_id = f"Speaker_{self.next_speaker_id}"
            self.speaker_embeddings_normalized[speaker_id] = embedding_norm
            self.next_speaker_id += 1
            print(f"DEBUG: Created new speaker {speaker_id} (first speaker)")
            return speaker_id

        best_match = None
        best_similarity = -1.0

        for speaker_id, stored_embedding_norm in self.speaker_embeddings_normalized.items():
            similarity = float(np.dot(embedding_norm, stored_embedding_norm))
            print(f"DEBUG: Comparing with {speaker_id}: similarity = {similarity:.4f}")
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = speaker_id

        print(
            f"DEBUG: Best match: {best_match} with similarity {best_similarity:.4f}, threshold: {self.similarity_threshold}"
        )

        if best_similarity > self.similarity_threshold:
            # Update the speaker's embedding with a weighted average
            # This helps create a more robust voiceprint over time
            existing_embedding = self.speaker_embeddings_normalized[best_match]
            
            # Weighted average: give more weight to the existing embedding
            # Adjust the weight (e.g., 0.7) as needed. A higher weight makes the
            # existing embedding more stable.
            weight = 0.7
            updated_embedding = (weight * existing_embedding) + ((1 - weight) * embedding_norm)
            
            # Re-normalize the updated embedding before storing
            updated_norm = np.linalg.norm(updated_embedding)
            if updated_norm > 0:
                self.speaker_embeddings_normalized[best_match] = updated_embedding / updated_norm
            
            print(f"DEBUG: Assigned to existing speaker {best_match} and updated their embedding.")
            return best_match
        else:
            speaker_id = f"Speaker_{self.next_speaker_id}"
            self.speaker_embeddings_normalized[speaker_id] = embedding_norm
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
        stream = None
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
            yield p, stream
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            p.terminate()

    def _transcribe_segment(self, audio_data):
        """Transcribe audio segment using faster-whisper."""
        segments, _ = self.model.transcribe(
            audio_data,
            language=self.language,
            task="transcribe",
            beam_size=5,
            temperature=0.0,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )
        return "".join(segment.text for segment in segments).strip()

    def _add_audio_segment(self, chunk_id, audio_data, start_time, end_time):
        """Add audio segment to result queue."""
        self.chunk_timestamps[chunk_id] = {"start": start_time, "end": end_time}
        self.result_queue.put((chunk_id, audio_data))
        self.chunk_counter += 1
        return self.chunk_counter - 1

    def _apply_vad(self, audio_data):
        """Apply VAD to audio data to extract speech segments."""
        self._ensure_vad_loaded()
        
        audio_tensor = torch.from_numpy(audio_data)

        (get_speech_timestamps, _, _, _, _) = self.vad_utils
        speech_timestamps = get_speech_timestamps(
            audio_tensor, self.vad_model, sampling_rate=self.RATE
        )

        if not speech_timestamps:
            return np.array([], dtype=np.float32)

        merged_audio = np.concatenate(
            [audio_data[ts["start"] : ts["end"]] for ts in speech_timestamps]
        )
        return merged_audio

    def _calculate_video_timestamp(self, elapsed_seconds):
        """Calculate timestamp in HH:MM:SS format."""
        if elapsed_seconds < 0:
            elapsed_seconds = 0

        total_seconds = int(elapsed_seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _write_to_file(self, line, force_flush=False):
        """Write directly to file without buffering."""
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(line)
            if force_flush:
                f.flush()

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

            segments_list = list(diarization_result.itertracks(yield_label=True))
            unique_speakers = set(speaker_label for _, _, speaker_label in segments_list)

            print(
                f"Found {len(segments_list)} speech segments from {len(unique_speakers)} speakers in segment {chunk_id}"
            )

            # --- MODE SELECTION LOGIC ---
            if self.mode == "transcription":
                # --- TRANSCRIPTION MODE: Merge segments before transcribing ---
                merged_segments = self._merge_speaker_segments(
                    segments_list, audio_data
                )
                for segment_info in merged_segments:
                    self._transcribe_and_display(segment_info)
            else:
                # --- SUBTITLE MODE: Transcribe each segment individually ---
                for turn, _, speaker_label in segments_list:
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

    def _merge_speaker_segments(self, segments_list, audio_data):
        """Two-step approach with batch embedding extraction for better performance."""
        merged_segments = []
        
        if not segments_list:
            return merged_segments

        valid_segments = []
        audio_segments_for_batch = []
        
        for turn, _, pyannote_speaker in segments_list:
            start_samples = int(turn.start * self.RATE)
            end_samples = int(turn.end * self.RATE)
            audio_segment = audio_data[start_samples:end_samples]

            if len(audio_segment) < self.RATE * 0.5:
                continue

            valid_segments.append({
                "turn": turn,
                "pyannote_speaker": pyannote_speaker,
                "audio_segment": audio_segment
            })
            audio_segments_for_batch.append(audio_segment)

        if not valid_segments:
            return merged_segments

        try:
            # Prepare batch processing
            max_len = max(len(seg) for seg in audio_segments_for_batch)
            padded_segments = []
            for seg in audio_segments_for_batch:
                if len(seg) < max_len:
                    padded = np.pad(seg, (0, max_len - len(seg)), mode='constant')
                else:
                    padded = seg
                padded_segments.append(padded)
            
            # Batch encode: PyTorch Tensor → PyTorch Tensor
            batch_tensor = torch.stack([
                torch.from_numpy(seg).float() for seg in padded_segments
            ]).to(self.device)
            
            batch_embeddings = self.embedding_model.encode_batch(batch_tensor)
            
            # Convert ALL embeddings to NumPy immediately
            batch_embeddings_np = batch_embeddings.cpu().numpy()
            
            identified_segments = []
            for idx, segment_info in enumerate(valid_segments):
                # Extract individual embedding as NumPy array
                embedding_np = batch_embeddings_np[idx]
                assigned_speaker = self._get_speaker_id(embedding_np)
                
                if assigned_speaker is None:
                    continue
                
                identified_segments.append({
                    "turn": segment_info["turn"],
                    "pyannote_speaker": segment_info["pyannote_speaker"],
                    "assigned_speaker": assigned_speaker,
                    "audio_segment": segment_info["audio_segment"]
                })
                print(
                    f"DEBUG: Identified segment {segment_info['turn'].start:.2f}s-{segment_info['turn'].end:.2f}s as {assigned_speaker}"
                )
                
        except Exception as e:
            print(f"DEBUG: Batch embedding failed: {e}")
            traceback.print_exc()
            identified_segments = []
            
            # Sequential fallback with proper NumPy conversion
            for segment_info in valid_segments:
                try:
                    audio_tensor = torch.from_numpy(segment_info["audio_segment"]).float().unsqueeze(0).to(self.device)
                    embedding_tensor = self.embedding_model.encode_batch(audio_tensor)
                    # Convert to NumPy immediately
                    embedding_np = embedding_tensor.squeeze().cpu().numpy()
                    assigned_speaker = self._get_speaker_id(embedding_np)
                    
                    if assigned_speaker is None:
                        continue
                    
                    identified_segments.append({
                        **segment_info,
                        "assigned_speaker": assigned_speaker
                    })
                except Exception as e2:
                    print(f"DEBUG: Failed to process segment at {segment_info['turn'].start:.2f}s: {e2}")
                    continue

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

                if self.mode == "transcription":
                    should_merge = same_speaker
                    gap_reason = f"gap: {time_gap:.2f}s (ignored in transcription mode)"
                else:
                    should_merge = same_speaker and time_gap < self.max_merge_gap
                    gap_reason = f"gap: {time_gap:.2f}s (limit: {self.max_merge_gap}s)"

                print(
                    f"DEBUG: Comparing segment {current_segment['turn'].start:.2f}s ({current_segment['assigned_speaker']}) with last in group {last_segment_in_group['turn'].end:.2f}s ({last_segment_in_group['assigned_speaker']}) - {gap_reason}"
                )

                if should_merge:
                    current_group.append(current_segment)
                    print(
                        f"DEBUG: ✓ Merging segment {current_segment['turn'].start:.2f}s-{current_segment['turn'].end:.2f}s with group (gap: {time_gap:.2f}s)"
                    )
                else:
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

            merged_groups.append(current_group)
            print(
                f"DEBUG: Final group with {len(current_group)} segments for {current_group[0]['assigned_speaker']}"
            )

        for group_idx, group in enumerate(merged_groups):
            if not group:
                continue

            speaker_id = group[0]["assigned_speaker"]
            start_time = group[0]["turn"].start
            end_time = group[-1]["turn"].end

            audio_chunks = [segment["audio_segment"] for segment in group]
            merged_audio = np.concatenate(audio_chunks)

            print(
                f"DEBUG: Group {group_idx + 1}: Created merged segment for {speaker_id}: {start_time:.2f}s-{end_time:.2f}s ({len(group)} segments, {len(merged_audio) / self.RATE:.2f}s audio)"
            )

            merged_segments.append(
                {
                    "speaker_label": speaker_id,
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
        speaker_label = segment_info["speaker_label"]
        speaker_audio_segment = segment_info["audio_segment"]

        speaker_audio_segment = self._apply_vad(speaker_audio_segment)

        min_duration = 0.5
        min_samples = int(min_duration * self.RATE)

        if len(speaker_audio_segment) < min_samples:
            print(
                f"DEBUG: Skipping segment too short: {len(speaker_audio_segment)} samples ({len(speaker_audio_segment) / self.RATE:.2f}s)"
            )
            return

        print(f"DEBUG: Transcribing merged segment for {speaker_label}")
        print(
            f"DEBUG: Turn: {turn.start:.2f}s -> {turn.end:.2f}s (duration: {turn.end - turn.start:.2f}s)"
        )
        print(
            f"DEBUG: Audio segment length: {len(speaker_audio_segment)} samples ({len(speaker_audio_segment) / self.RATE:.2f}s)"
        )

        transcription = self._transcribe_segment(speaker_audio_segment)
        if not transcription or transcription.isspace():
            print("DEBUG: Transcription was empty or whitespace only")
            return

        chunk_start_time = self.chunk_timestamps.get(self.chunk_counter - 1, {}).get(
            "start", 0
        )
        elapsed_seconds = chunk_start_time + turn.start
        timestamp = self._calculate_video_timestamp(elapsed_seconds)

        if self.mode == "transcription":
            if self.transcription_buffer["speaker"] == speaker_label:
                self.transcription_buffer["text"] += " " + transcription
            else:
                # Changement de locuteur → flush immédiat du buffer précédent
                self._flush_transcription_buffer()
                self.transcription_buffer["speaker"] = speaker_label
                self.transcription_buffer["timestamp"] = timestamp
                self.transcription_buffer["text"] = transcription
        else:
            line = f"[{timestamp}][{speaker_label}] {transcription}\n"
            print(f"Live: {line.strip()}")
            self._write_to_file(line, force_flush=True)

    def _flush_transcription_buffer(self):
        """Writes the content of the transcription buffer to the file."""
        if self.transcription_buffer["speaker"] and self.transcription_buffer["text"]:
            line = f"[{self.transcription_buffer['timestamp']}][{self.transcription_buffer['speaker']}] {self.transcription_buffer['text'].strip()}\n"
            print(f"Finalized: {line.strip()}")
            self._write_to_file(line, force_flush=True)
            self.transcription_buffer = {"speaker": None, "timestamp": None, "text": ""}

    def transcribe_file(self, file_path):
        """Transcribe an entire audio file by chunking it and processing it like a live stream."""
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
                audio_bytes = wf.readframes(wf.getnframes())

            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            print("Audio file loaded. Starting transcription process...")

            self.is_running = True
            self.transcribe_thread = threading.Thread(target=self._transcribe_audio)
            self.transcribe_thread.daemon = True
            self.transcribe_thread.start()

            # Use mode-specific buffer duration
            chunk_duration_seconds = self.MAX_BUFFER_DURATION
            chunk_samples = int(chunk_duration_seconds * self.RATE)
            total_samples = len(audio_float32)

            print(f"Using {chunk_duration_seconds}s chunks for {self.mode} mode")

            for i in range(0, total_samples, chunk_samples):
                # Check if interrupted by Ctrl+C
                if not self.is_running:
                    print("Interruption detected, stopping file queuing...")
                    break
                
                start_sample = i
                end_sample = i + chunk_samples
                chunk_data = audio_float32[start_sample:end_sample]

                if len(chunk_data) == 0:
                    continue

                start_time = start_sample / self.RATE
                end_time = end_sample / self.RATE

                self._add_audio_segment(
                    self.chunk_counter, chunk_data, start_time, end_time
                )
                print(
                    f"Queued segment {self.chunk_counter - 1} ({start_time:.2f}s to {end_time:.2f}s)"
                )
                # Give the worker thread a moment to catch up
                time.sleep(0.1)

            self.is_running = False

            print("All segments queued. Waiting for final processing...")
            if hasattr(self, "transcribe_thread"):
                self.transcribe_thread.join()

            if self.mode == "transcription":
                self._flush_transcription_buffer()

            self._write_to_file("", force_flush=True)

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
                chunk_id, audio_data = self.result_queue.get(timeout=0.5)
                self._process_transcription(
                    chunk_id, audio_data, is_final=not self.is_running
                )
            except queue.Empty:
                continue
            except Exception:
                print("Transcription error:")
                traceback.print_exc()

    def _process_audio_stream(self, in_data, frame_count, time_info, status):
        """Audio stream callback - accumulate and cut on buffer limit."""
        now = datetime.now()
        chunk = np.frombuffer(in_data, dtype=np.float32)
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
        self.full_audio_buffer_list.append(chunk)

        buffer_len = len(self.audio_buffer)
        
        # Cut when buffer reaches target duration (mode-specific)
        if buffer_len >= self.max_buffer_samples:
            buffer_duration = buffer_len / self.RATE
            timestamp_start = (now - timedelta(seconds=buffer_duration)).timestamp()
            
            mode_label = "subtitle (real-time)" if self.mode == "subtitle" else "transcription (coherence)"
            print(f"→ Buffer reached {buffer_duration:.1f}s ({mode_label}), sending to pyannote")
            
            self._add_audio_segment(
                self.chunk_counter,
                self.audio_buffer,
                timestamp_start,
                now.timestamp(),
            )

            # Reset buffer completely
            self.audio_buffer = np.array([], dtype=np.float32)

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
        if self.stream:
            self.stream.start_stream()

    def stop_recording(self):
        """Stop recording and finalize transcriptions."""
        if not self.is_running:
            return

        print("\nStopping recording and finalizing transcriptions...")
        self.is_running = False

        # Stop audio stream
        if hasattr(self, "stream") and self.stream and self.stream.is_active():
            self.stream.stop_stream()
        if hasattr(self, "audio_context"):
            self.audio_context.__exit__(None, None, None)

        # Process final buffer content
        if self.audio_buffer.size > 0:
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
            self.transcribe_thread.join(timeout=120)

        # Flush any remaining buffered transcription
        if self.mode == "transcription":
            self._flush_transcription_buffer()

        self._write_to_file("", force_flush=True)

        print(f"Transcription saved to {self.filename}")

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
        "tiny",
        "base",
        "small",
        "medium",
        "large",
        "large-v2",
        "large-v3",
        "large-v3-turbo",
    ]

    parser = argparse.ArgumentParser(
        description="Transcribe audio live from microphone or from a file."
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", "haw", "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jw", "ka", "kk", "km", "kn", "ko", "la", "lb", "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo", "zh", "yue"],
        default="fr",
        help="Language for transcription (use language code).",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=MODELS,
        default="large-v3",
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
        default=0.20,
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

    transcriber = None
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
        if transcriber:
            if args.file:
                # Mode fichier : arrêt simple
                transcriber.is_running = False
                
                if hasattr(transcriber, "transcribe_thread") and transcriber.transcribe_thread.is_alive():
                    print("Waiting for transcription thread to finalize...")
                    transcriber.transcribe_thread.join(timeout=30)
                
                # Vider le buffer de transcription si en mode "transcription"
                if transcriber.mode == "transcription":
                    transcriber._flush_transcription_buffer()
                
                # Forcer l'écriture de tout le buffer fichier
                transcriber._write_to_file("", force_flush=True)
                
            else:
                # Mode live : utiliser stop_recording() qui fait tout le nettoyage
                transcriber.stop_recording()

        print("Done.")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
