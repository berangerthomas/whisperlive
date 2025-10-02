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
os.environ["HF_HUB_LOCAL_DIR_IS_SYMLINK_SUPPORTED"] = "0"


class WhisperLiveTranscription:
    def __init__(
        self,
        model_id="large-v3",
        language="fr",
        similarity_threshold=0.7,
        mode="transcription",
        min_speakers=None,
        max_speakers=None,
        diarization_method="pyannote",
        enhancement_method="none",
        transcription_engine="auto",  # NEW: "auto", "faster-whisper", "transformers"
        auto_engine_threshold=15.0,   # NEW: threshold in seconds for auto mode
    ):
        print("Launching WhisperLiveTranscription...")

        # Store configuration
        self.model_id = model_id
        self.language = language
        self.mode = mode
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.diarization_method = diarization_method
        self.enhancement_method = enhancement_method
        self.transcription_engine = transcription_engine  # NEW
        self.auto_engine_threshold = auto_engine_threshold  # NEW
        self.max_merge_gap = 1.5  # Max silence in seconds between segments to merge

        # Load environment variables
        load_dotenv()
        hf_token = os.getenv("HUGGING_FACE_TOKEN")

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load Whisper model(s) based on engine choice
        self._load_transcription_models()  # MODIFIED

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

        # Output file - Will be set when recording or transcribing starts
        self.filename = None

        # State for transcription mode merging
        self.transcription_buffer = {"speaker": None, "timestamp": None, "text": ""}

        # Audio enhancement models - lazy loading
        self.nsnet2_session = None

    def _ensure_nsnet2_model_downloaded(self):
        """Downloads the NSNet2 ONNX model if it doesn't exist."""
        model_path = "nsnet2-20ms-baseline.onnx"
        if not os.path.exists(model_path):
            print("Downloading NSNet2 model...")
            try:
                import urllib.request
                url = "https://github.com/microsoft/DNS-Challenge/raw/master/NSNet2-baseline/nsnet2-20ms-baseline.onnx"
                urllib.request.urlretrieve(url, model_path)
                print("NSNet2 model downloaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to download NSNet2 model: {e}")
        return model_path

    def _apply_enhancement(self, audio_data):
        """Apply selected audio enhancement method."""
        if self.enhancement_method == "none":
            return audio_data

        if self.enhancement_method == "nsnet2":
            if self.nsnet2_session is None:
                print("Loading NSNet2 denoiser...")
                try:
                    import onnxruntime as ort
                    model_path = self._ensure_nsnet2_model_downloaded()
                    self.nsnet2_session = ort.InferenceSession(model_path)
                except ImportError:
                    warnings.warn("onnxruntime is not installed. Please run 'uv sync'. Skipping enhancement.")
                    return audio_data
                except Exception as e:
                    warnings.warn(f"Failed to load NSNet2 model: {e}. Skipping enhancement.")
                    return audio_data
            
            # Normalize audio to float32, as in the example
            audio = audio_data.astype(np.float32)
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            # NSNet2 processes in 20ms frames (320 samples at 16kHz) with 10ms overlap
            frame_size = 320
            hop_size = 160
            
            output_audio = []
            for i in range(0, len(audio) - frame_size, hop_size):
                frame = audio[i:i+frame_size]
                
                # Run inference
                enhanced_frame = self.nsnet2_session.run(
                    None, 
                    {"input": frame.reshape(1, -1)}
                )[0]
                
                # Append the first half (hop_size) of the enhanced frame to avoid overlap issues
                output_audio.append(enhanced_frame.flatten()[:hop_size])
            
            if not output_audio:
                return np.array([], dtype=np.float32)

            return np.concatenate(output_audio)

        elif self.enhancement_method == "demucs":
            # Demucs is heavy, so we import and load it only when needed.
            if self.mode != "transcription" and self.start_time is not None:
                 warnings.warn("Demucs is not recommended for live processing due to high latency. Using it anyway.")
            
            try:
                from demucs.pretrained import get_model
                from demucs.apply import apply_model
                import torchaudio
            except ImportError as e:
                print(f"!!! DEMUCS IMPORT FAILED: {e} !!!")
                warnings.warn("Demucs not installed. Please run 'uv sync'. Skipping enhancement.")
                return audio_data

            print("Loading Demucs model for audio separation...")
            demucs_model = get_model('htdemucs')
            demucs_model.to(self.device)
            demucs_model.eval()
            
            # Demucs expects 44.1kHz, we resample if necessary
            if self.RATE != 44100:
                audio_tensor = torch.from_numpy(audio_data).float()
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                
                resampler = torchaudio.transforms.Resample(
                    orig_freq=self.RATE,
                    new_freq=44100
                ).to(self.device)
                audio_tensor = resampler(audio_tensor.to(self.device))
            else:
                audio_tensor = torch.from_numpy(audio_data).float()
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
            
            # Demucs expects a batch [batch, channels, samples]
            if audio_tensor.shape[0] == 1:  # mono -> stereo
                audio_tensor = audio_tensor.repeat(2, 1)
            
            audio_tensor = audio_tensor.unsqueeze(0).to(self.device)
            
            # Apply the model
            print("Applying Demucs model...")
            with torch.no_grad():
                sources = apply_model(
                    demucs_model, 
                    audio_tensor,
                    split=True,
                    overlap=0.25
                )
            
            # Demucs returns [batch, stems, channels, samples]
            vocals = sources[0, 3]  # Take only vocals
            
            # Convert to mono by averaging the channels. This is robust.
            vocals_mono_tensor = vocals.mean(dim=0)
            
            # Resample back to the original sample rate
            if self.RATE != 44100:
                resampler_back = torchaudio.transforms.Resample(
                    orig_freq=44100,
                    new_freq=self.RATE
                ).to(self.device)
                vocals_mono_tensor = resampler_back(vocals_mono_tensor.unsqueeze(0)).squeeze()

            # Normalize the final mono vocals tensor to prevent low volume issues
            max_val = torch.max(torch.abs(vocals_mono_tensor))
            if max_val > 0:
                vocals_mono_tensor = vocals_mono_tensor / max_val

            # Ensure the final output is float32 for compatibility with Whisper
            vocals_mono = vocals_mono_tensor.cpu().numpy().astype(np.float32)
            
            print("Demucs processing complete.")
            return vocals_mono

        else:
            warnings.warn(f"Audio enhancement method '{self.enhancement_method}' is not yet implemented. Audio will not be processed.")
            return audio_data

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
        
        # VAD model requires specific chunk sizes
        # For 16kHz: 512 samples = 32ms
        # For 8kHz: 256 samples = 32ms
        self.CHUNK = 512  # Changed from 1024 to match VAD requirements
        
        # MODE-SPECIFIC PARAMETERS
        if self.mode == "subtitle":
            # Subtitle mode: Send frequently to pyannote for immediate processing
            self.MAX_BUFFER_DURATION = 5.0  # Send every 5s for real-time response
        else:  # transcription mode
            # Transcription mode: Accumulate longer chunks for coherence
            self.MAX_BUFFER_DURATION = 75.0  # 1min15s - pyannote will cut at natural pauses

        # Pre-calculate constants
        self.max_buffer_samples = int(self.MAX_BUFFER_DURATION * self.RATE)

        # VAD parameters for subtitle mode
        if self.mode == "subtitle":
            self.VAD_SPEECH_THRESHOLD = 0.5  # Probability threshold for speech
            self.VAD_SILENCE_DURATION_S = 0.5  # Silence duration to trigger segment cut
            self.VAD_MIN_SPEECH_DURATION_S = 0.2 # Minimum speech duration to process
            self.vad_silence_samples = int(self.VAD_SILENCE_DURATION_S * self.RATE)

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
        
        # VAD-specific state for subtitle mode
        if self.mode == "subtitle":
            self.vad_speech_buffer = np.array([], dtype=np.float32)
            self.vad_is_speaking = False
            self.vad_silence_counter = 0

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
            # Format to SPEAKER_00, SPEAKER_01, etc.
            speaker_id = f"SPEAKER_{self.next_speaker_id - 1:02d}"
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
            speaker_id = f"SPEAKER_{self.next_speaker_id - 1:02d}"
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
        """
        Transcribe audio segment with intelligent engine selection.
        
        Strategy:
        - Auto mode: Use transformers for short segments (< threshold), faster-whisper for long segments
        - Manual mode: Use the specified engine
        """
        audio_duration = len(audio_data) / self.RATE
        
        # Protection: Don't transcribe very short segments
        if audio_duration < 0.5:
            print(f"DEBUG: Segment too short ({audio_duration:.2f}s), skipping")
            return ""
        
        # Determine which engine to use
        if self.transcription_engine == "auto":
            # Intelligent selection based on duration and configured threshold
            use_transformers = audio_duration < self.auto_engine_threshold
            engine_name = "transformers" if use_transformers else "faster-whisper"
            print(f"DEBUG: Auto-selecting {engine_name} for {audio_duration:.2f}s segment (threshold: {self.auto_engine_threshold}s)")
        elif self.transcription_engine == "transformers":
            use_transformers = True
            engine_name = "transformers"
        else:  # faster-whisper
            use_transformers = False
            engine_name = "faster-whisper"
        
        # Transcribe with selected engine
        if use_transformers:
            return self._transcribe_with_transformers(audio_data, audio_duration)
        else:
            return self._transcribe_with_faster_whisper(audio_data, audio_duration)

    def _transcribe_with_faster_whisper(self, audio_data, audio_duration):
        """Transcribe with faster-whisper (optimized for speed, with anti-repetition)."""
        segments, info = self.faster_whisper_model.transcribe(
            audio_data,
            language=self.language,
            task="transcribe",
            beam_size=5,
            
            # Anti-repetition parameters
            repetition_penalty=1.2,
            no_repeat_ngram_size=5,
            
            # Temperature and sampling
            temperature=0.0,
            
            # Context management
            condition_on_previous_text=False,
            initial_prompt="Transcription précise et naturelle sans répétitions.",
            
            # Quality thresholds
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            
            # Token control
            prefix=None,
            suppress_blank=True,
            suppress_tokens=[-1],
            
            # Timestamp management
            without_timestamps=False,
            max_initial_timestamp=audio_duration,
            word_timestamps=False,
            
            prepend_punctuations="\"'¿([{-",
            append_punctuations="\"'.。,，!！?？:：)]}、",
        )
        
        result_text = "".join(segment.text for segment in segments).strip()
        
        # Post-processing: detect and warn about potential repetitions
        if result_text:
            words = result_text.split()
            if len(words) > 10:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.3:
                    print(f"WARNING: High repetition detected (unique ratio: {unique_ratio:.2f}) in faster-whisper output")
                    print(f"  → Consider lowering --auto-engine-threshold (current: {self.auto_engine_threshold}s)")
        
        return result_text

    def _transcribe_with_transformers(self, audio_data, audio_duration):
        """Transcribe with transformers (more stable for short segments)."""
        # Normalize audio
        audio_normalized = audio_data / (np.max(np.abs(audio_data)) + 1e-8)
        
        # Process input
        processed_input = self.transformers_processor(
            audio_normalized,
            sampling_rate=self.RATE,
            return_tensors="pt"
        )
        features = processed_input.input_features.to(self.device)
        
        # Create attention mask
        attention_mask = torch.ones(
            features.shape[:-1], dtype=torch.long, device=self.device
        )
        
        # Generate with anti-repetition
        predicted_ids = self.transformers_model.generate(
            features,
            attention_mask=attention_mask,
            language=self.language,
            task="transcribe",
            max_length=448,
            num_beams=5,
            repetition_penalty=1.2,
            no_repeat_ngram_size=5,
            # Remove conflicting parameters:
            # - do_sample and temperature cause a warning (removed)
            # - suppress_tokens conflicts with forced_decoder_ids (removed)
            # - forced_decoder_ids is deprecated (removed)
        )
        
        transcription = self.transformers_processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0].strip()
        
        return transcription
    
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
        """
        Process a chunk of audio: diarize, identify, and transcribe.
        This method acts as a router, choosing the diarization method based on mode.
        """
        try:
            # For live subtitle mode with cluster method, we bypass pyannote.
            is_live_mode = self.start_time is not None
            if (
                is_live_mode
                and self.mode == "subtitle"
                and self.diarization_method == "cluster"
            ):
                self._process_transcription_cluster(chunk_id, audio_data)
            else:
                # All other cases (file mode, or pyannote in live mode) use pyannote.
                self._process_transcription_pyannote(chunk_id, audio_data)

        except Exception:
            print(f"Error during transcription of segment {chunk_id}:")
            traceback.print_exc()
        finally:
            if chunk_id in self.chunk_timestamps:
                del self.chunk_timestamps[chunk_id]

    def _process_transcription_cluster(self, chunk_id, audio_data):
        """Processes a chunk using VAD segmentation and embedding comparison (live mode)."""
        print(f"DEBUG: Processing segment {chunk_id} with live 'cluster' method.")

        # The audio_data is already a speech segment from VAD in subtitle mode.
        # We just need to get its embedding and identify the speaker.
        try:
            audio_tensor = (
                torch.from_numpy(audio_data).float().unsqueeze(0).to(self.device)
            )
            embedding_tensor = self.embedding_model.encode_batch(audio_tensor)
            embedding_np = embedding_tensor.squeeze().cpu().numpy()
        except Exception as e:
            print(f"DEBUG: Could not get embedding for segment {chunk_id}: {e}")
            return

        assigned_speaker = self._get_speaker_id(embedding_np)
        if assigned_speaker is None:
            print(f"DEBUG: Could not assign speaker for segment {chunk_id}.")
            return

        print(f"DEBUG: Identified segment {chunk_id} as {assigned_speaker}")

        # --- Logic adapted from _transcribe_and_display ---

        # VAD has already been applied in the audio callback, so we skip it here.
        min_duration = 0.5
        min_samples = int(min_duration * self.RATE)

        if len(audio_data) < min_samples:
            print(
                f"DEBUG: Skipping segment too short: {len(audio_data)} samples ({len(audio_data) / self.RATE:.2f}s)"
            )
            return

        transcription = self._transcribe_segment(audio_data)
        if not transcription or transcription.isspace():
            print("DEBUG: Transcription was empty or whitespace only")
            return

        chunk_start_time = self.chunk_timestamps.get(chunk_id, {}).get("start", 0)
        # The 'turn' starts at 0 relative to the chunk
        elapsed_seconds = chunk_start_time
        timestamp = self._calculate_video_timestamp(elapsed_seconds)

        # This live method is only for subtitle mode, so we write directly.
        line = f"[{timestamp}][{assigned_speaker}] {transcription}\n"
        print(f"Live: {line.strip()}")
        self._write_to_file(line, force_flush=True)

    def _process_transcription_pyannote(self, chunk_id, audio_data):
        """Processes a chunk using pyannote for diarization."""
        print(f"Processing segment {chunk_id} with 'pyannote' method...")
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

        merged_segments = self._merge_speaker_segments(segments_list, audio_data)
        for segment_info in merged_segments:
            self._transcribe_and_display(segment_info)

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

                # In subtitle mode, we merge if the gap is small.
                # In transcription mode (file-based), this function is called on the whole file,
                # so we only care about merging consecutive segments from the same speaker, regardless of gap.
                should_merge = same_speaker and (
                    self.mode == "transcription" or time_gap < self.max_merge_gap
                )
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

        min_duration = 0.5
        min_samples = int(min_duration * self.RATE)

        if len(speaker_audio_segment) < min_samples:
            print(
                f"DEBUG: Skipping segment too short: {len(speaker_audio_segment)} samples ({len(speaker_audio_segment) / self.RATE:.2f}s) at {turn.start:.2f}s"
            )
            return

        print(f"DEBUG: Transcribing chunk for {speaker_label} [{turn.start:.2f}s - {turn.end:.2f}s]")
        print(f"DEBUG: Audio segment length: {len(speaker_audio_segment)} samples ({len(speaker_audio_segment) / self.RATE:.2f}s)")

        transcription = self._transcribe_segment(speaker_audio_segment)

        if not transcription or transcription.isspace():
            print(f"DEBUG: Transcription was empty or whitespace only for segment at {turn.start:.2f}s")
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

    def _create_optimal_transcription_chunks(self, segments_with_speakers, audio_data):
        """
        Create optimal chunks (1-2 minutes) from speaker segments for transcription.
        """
        TARGET_DURATION = 90.0
        MAX_DURATION = 120.0
        MIN_SILENCE_GAP = 0.5
        
        optimal_chunks = []
        current_chunk = {
            "speaker": None,
            "start_time": None,
            "segments": [],
            "duration": 0.0
        }
        
        print(f"\nCreating optimal transcription chunks from {len(segments_with_speakers)} segments...")
        
        # DEBUG: Log all input segments to detect gaps
        print("\nDEBUG: Input segments from pyannote:")
        for i, segment_info in enumerate(segments_with_speakers):
            turn = segment_info["turn"]
            speaker = segment_info["speaker_label"]
            duration = turn.end - turn.start
            print(f"  Segment {i}: {speaker} [{turn.start:.2f}s - {turn.end:.2f}s] duration={duration:.2f}s")
            
            # Check for gaps
            if i > 0:
                prev_turn = segments_with_speakers[i-1]["turn"]
                gap = turn.start - prev_turn.end
                if gap > 5.0:  # Gap > 5 seconds
                    print(f"    ⚠️  GAP DETECTED: {gap:.2f}s silence before this segment")
        
        for i, segment_info in enumerate(segments_with_speakers):
            speaker = segment_info["speaker_label"]
            turn = segment_info["turn"]
            audio_segment = segment_info["audio_segment"]
            segment_duration = turn.end - turn.start
            
            # Check if we need to start a new chunk
            should_cut = False
            cut_reason = ""
            
            if current_chunk["speaker"] is None:
                # First segment - start new chunk
                should_cut = False
            elif current_chunk["speaker"] != speaker:
                # Speaker change - always cut
                should_cut = True
                cut_reason = "speaker change"
            elif current_chunk["duration"] + segment_duration > MAX_DURATION:
                # Would exceed hard limit - must cut
                should_cut = True
                cut_reason = f"max duration ({MAX_DURATION}s)"
            elif current_chunk["duration"] > TARGET_DURATION:
                # Past target duration - look for silence
                if i < len(segments_with_speakers) - 1:
                    next_segment = segments_with_speakers[i + 1]
                    gap = next_segment["turn"].start - turn.end
                    if gap >= MIN_SILENCE_GAP:
                        should_cut = True
                        cut_reason = f"natural pause ({gap:.2f}s silence)"
        
            if should_cut and current_chunk["segments"]:
                # Finalize current chunk
                chunk_start = current_chunk["segments"][0]["turn"].start
                chunk_end = current_chunk["segments"][-1]["turn"].end
                chunk_audio = np.concatenate([s["audio_segment"] for s in current_chunk["segments"]])
                
                optimal_chunks.append({
                    "speaker_label": current_chunk["speaker"],
                    "turn": Segment(chunk_start, chunk_end),
                    "audio_segment": chunk_audio,
                    "num_segments": len(current_chunk["segments"])
                })
                
                print(f"  ✓ Chunk {len(optimal_chunks)}: {current_chunk['speaker']} "
                      f"[{chunk_start:.1f}s-{chunk_end:.1f}s] duration={current_chunk['duration']:.1f}s "
                      f"({len(current_chunk['segments'])} segments) - Cut reason: {cut_reason}")
                
                # Reset for new chunk
                current_chunk = {
                    "speaker": None,
                    "start_time": None,
                    "segments": [],
                    "duration": 0.0
                }
            
            # Add segment to current chunk
            if current_chunk["speaker"] is None:
                current_chunk["speaker"] = speaker
                current_chunk["start_time"] = turn.start
            
            current_chunk["segments"].append(segment_info)
            current_chunk["duration"] += segment_duration
        
        # Don't forget the last chunk
        if current_chunk["segments"]:
            chunk_start = current_chunk["segments"][0]["turn"].start
            chunk_end = current_chunk["segments"][-1]["turn"].end
            chunk_audio = np.concatenate([s["audio_segment"] for s in current_chunk["segments"]])
            
            optimal_chunks.append({
                "speaker_label": current_chunk["speaker"],
                "turn": Segment(chunk_start, chunk_end),
                "audio_segment": chunk_audio,
                "num_segments": len(current_chunk["segments"])
            })
            
            print(f"  ✓ Chunk {len(optimal_chunks)}: {current_chunk['speaker']} "
                  f"[{chunk_start:.1f}s-{chunk_end:.1f}s] duration={current_chunk['duration']:.1f}s "
                  f"({len(current_chunk['segments'])} segments) - Final chunk")
        
        print(f"\nCreated {len(optimal_chunks)} optimal chunks (avg duration: "
              f"{np.mean([c['turn'].end - c['turn'].start for c in optimal_chunks]):.1f}s)")
        
        return optimal_chunks

    def _generate_filename(self, base_name=None, found_speakers=None):
        """Generates a standardized filename for the transcription output."""
        model_name_safe = self.model_id.replace("/", "_")
        
        # Use input file's base name or 'live' for microphone input
        base = os.path.splitext(os.path.basename(base_name))[0] if base_name else "live"

        # Core components
        parts = [
            base,
            self.mode,
            model_name_safe,
            self.diarization_method
        ]

        # Add diarization-specific details
        if self.diarization_method == 'cluster':
            parts.append(f"thresh{self.similarity_threshold:.2f}")

        # Add speaker count if available (only in file mode)
        if found_speakers is not None:
            parts.append(f"{found_speakers}-speakers")
        
        # Add timestamp for live recordings to ensure uniqueness
        if not base_name:
            parts.append(datetime.now().strftime('%Y%m%d_%H%M%S'))

        return "_".join(parts) + ".txt"

    def transcribe_file(self, file_path):
        """
        Transcribe an entire audio file using the optimal offline approach.
        Uses pyannote for full-file diarization, then creates optimal chunks for Whisper.
        """
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        model_name_safe = self.model_id.replace("/", "_")

        self.filename = None

        print(f"Transcribing audio file (offline mode): {file_path}")
        print(f"Using diarization method: {self.diarization_method}")

        try:
            # Load audio file
            with wave.open(file_path, "rb") as wf:
                framerate = wf.getframerate()
                if framerate != self.RATE:
                    raise ValueError(
                        f"Unsupported sample rate: {framerate}. Please use {self.RATE} Hz."
                    )
                audio_bytes = wf.readframes(wf.getnframes())
            audio_float32 = (
                np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            )
            print("Audio file loaded.")

            # Apply audio enhancement if selected
            audio_float32 = self._apply_enhancement(audio_float32)

            # Save cleaned audio if an enhancement method was used
            if self.enhancement_method != "none":
                cleaned_filename = f"{base_name}_{self.enhancement_method}_cleaned.wav"
                print(f"\nSaving cleaned audio to {cleaned_filename}...")
                
                # Scale to int16
                max_val = np.max(np.abs(audio_float32))
                if max_val > 0:
                    scaled_audio = np.int16(audio_float32 / max_val * 32767.0)
                else:
                    scaled_audio = np.int16(audio_float32)

                # Write to WAV file
                with wave.open(cleaned_filename, "wb") as wf:
                    wf.setnchannels(self.CHANNELS)
                    wf.setsampwidth(2)  # 2 bytes for int16
                    wf.setframerate(self.RATE)
                    wf.writeframes(scaled_audio.tobytes())
                print("Cleaned audio saved.")

            # STEP 1: Diarization (pyannote processes the entire file)
            if self.diarization_method == "pyannote":
                print("\n" + "*"*10 + " Step 1: Diarizing with pyannote (full file)... " + "*"*10)
                diarization_params = {}
                if self.min_speakers is not None:
                    diarization_params["min_speakers"] = self.min_speakers
                if self.max_speakers is not None:
                    diarization_params["max_speakers"] = self.max_speakers
                
                diarization_result = self.diarization_pipeline(
                    {"waveform": torch.from_numpy(audio_float32).unsqueeze(0), "sample_rate": self.RATE},
                    **diarization_params
                )
                segments_list = list(diarization_result.itertracks(yield_label=True))
                print(f"Pyannote found {len(segments_list)} speech segments.")

                # Convert to our format with audio data
                segments_with_speakers = [
                    {
                        "speaker_label": speaker_label,
                        "turn": turn,
                        "audio_segment": audio_float32[int(turn.start * self.RATE):int(turn.end * self.RATE)],
                    }
                    for turn, _, speaker_label in segments_list
                ]
                found_speakers = len(set(seg["speaker_label"] for seg in segments_with_speakers))

            else:  # cluster method - same as before
                print("\n" + "*"*10 + " Step 1: Segmenting speech with Silero VAD... " + "*"*10)
                self._ensure_vad_loaded()
                (get_speech_timestamps, _, _, _, _) = self.vad_utils
                speech_timestamps = get_speech_timestamps(
                    torch.from_numpy(audio_float32), self.vad_model, sampling_rate=self.RATE
                )
                print(f"VAD found {len(speech_timestamps)} speech segments.")

                print("\n" + "*"*10 + " Step 2: Identifying speakers with clustering... " + "*"*10)
                # [Le code de clustering reste identique]
                try:
                    from sklearn.cluster import AgglomerativeClustering
                    from sklearn.preprocessing import normalize
                except ImportError:
                    raise ImportError("scikit-learn is required. Please run: pip install scikit-learn")

                valid_segments_info = []
                audio_segments_for_batch = []
                for ts in speech_timestamps:
                    start_samples, end_samples = ts["start"], ts["end"]
                    audio_segment = audio_float32[start_samples:end_samples]
                    if len(audio_segment) < self.RATE * 0.5: continue
                    
                    valid_segments_info.append({
                        "turn": Segment(start_samples / self.RATE, end_samples / self.RATE),
                        "audio_segment": audio_segment
                    })
                    audio_segments_for_batch.append(audio_segment)

                if not valid_segments_info:
                    segments_with_speakers, found_speakers = [], 0
                else:
                    try:
                        max_len = max(len(seg) for seg in audio_segments_for_batch)
                        padded_segments = [np.pad(seg, (0, max_len - len(seg)), mode="constant") if len(seg) < max_len else seg for seg in audio_segments_for_batch]
                        batch_tensor = torch.stack([torch.from_numpy(seg).float() for seg in padded_segments]).to(self.device)
                        all_embeddings = self.embedding_model.encode_batch(batch_tensor).cpu().numpy()
                    except Exception:
                        all_embeddings_list = []
                        for seg_audio in audio_segments_for_batch:
                            try:
                                embedding = self.embedding_model.encode_batch(torch.from_numpy(seg_audio).float().unsqueeze(0).to(self.device)).cpu().numpy()
                                all_embeddings_list.append(embedding)
                            except Exception:
                                pass
                        all_embeddings = np.concatenate(all_embeddings_list, axis=0) if all_embeddings_list else np.array([])

                    if all_embeddings.size == 0:
                        segments_with_speakers, found_speakers = [], 0
                    else:
                        # Ensure embeddings are 2D [num_segments, embedding_dim]
                        if all_embeddings.ndim == 3 and all_embeddings.shape[1] == 1:
                            all_embeddings = all_embeddings.squeeze(1)

                        normalized_embeddings = normalize(all_embeddings, norm="l2", axis=1)
                        clustering = AgglomerativeClustering(
                            n_clusters=self.max_speakers if self.max_speakers else None,
                            metric="cosine", linkage="average",
                            distance_threshold=1 - self.similarity_threshold if not self.max_speakers else None
                        ).fit(normalized_embeddings)
                        
                        cluster_labels = clustering.labels_
                        found_speakers = len(set(cluster_labels))
                        
                        segments_with_speakers = [{
                            "speaker_label": f"SPEAKER_{label:02d}",
                            "turn": info["turn"],
                            "audio_segment": info["audio_segment"]
                        } for info, label in zip(valid_segments_info, cluster_labels)]

            print(f"Diarization complete. Found {found_speakers} unique speakers.")

            # Build filename
            self.filename = self._generate_filename(
                base_name=file_path, found_speakers=found_speakers
            )
            self._init_output_file()

            # STEP 2: Create optimal chunks for transcription (NEW!)
            print("\n" + "*"*10 + " Step 2: Creating optimal transcription chunks... " + "*"*10)
            optimal_chunks = self._create_optimal_transcription_chunks(segments_with_speakers, audio_float32)

            # STEP 3: Transcribe optimal chunks
            print("\n" + "*"*10 + " Step 3: Transcribing chunks... " + "*"*10)
            self.chunk_timestamps[0] = {"start": 0, "end": len(audio_float32) / self.RATE}
            
            for chunk_info in optimal_chunks:
                self._transcribe_and_display(chunk_info)

            # Final flush
            self._flush_transcription_buffer()
            self._write_to_file("\n# Transcription complete.\n", force_flush=True)
            print(f"\nTranscription finished. Results saved to {self.filename}")

        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
        except Exception as e:
            print(f"An error occurred during file transcription: {e}")
            traceback.print_exc()
        finally:
            if 0 in self.chunk_timestamps:
                del self.chunk_timestamps[0]

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
        """Audio stream callback."""
        now = datetime.now()
        chunk = np.frombuffer(in_data, dtype=np.float32)
        
        # Apply audio enhancement if selected
        chunk = self._apply_enhancement(chunk)
        
        self.full_audio_buffer_list.append(chunk)  # Always save full audio

        if self.mode == "subtitle":
            self._process_subtitle_mode(chunk, now)
        else:  # transcription mode
            self._process_transcription_mode(chunk, now)

        return (in_data, pyaudio.paContinue)

    def _process_subtitle_mode(self, chunk, now):
        """Process audio chunk for subtitle mode using VAD."""
        # VAD expects a torch tensor - make a copy to ensure writability
        chunk_tensor = torch.from_numpy(np.copy(chunk))
        speech_prob = self.vad_model(chunk_tensor, self.RATE).item()

        if speech_prob > self.VAD_SPEECH_THRESHOLD:
            # Speech detected
            self.vad_silence_counter = 0
            if not self.vad_is_speaking:
                print("VAD: Speech started")
                self.vad_is_speaking = True
            self.vad_speech_buffer = np.concatenate([self.vad_speech_buffer, chunk])
        else:
            # Silence detected
            if self.vad_is_speaking:
                self.vad_silence_counter += len(chunk)
                if self.vad_silence_counter >= self.vad_silence_samples:
                    print(
                        f"VAD: Speech ended (silence duration: {self.vad_silence_counter / self.RATE:.2f}s)"
                    )

                    # Process the captured speech segment if it's long enough
                    if (
                        len(self.vad_speech_buffer) / self.RATE
                        > self.VAD_MIN_SPEECH_DURATION_S
                    ):
                        buffer_duration = len(self.vad_speech_buffer) / self.RATE
                        timestamp_start = (
                            now - timedelta(seconds=buffer_duration)
                        ).timestamp()

                        print(
                            f"→ VAD cut: sending {buffer_duration:.2f}s segment for processing"
                        )
                        self._add_audio_segment(
                            self.chunk_counter,
                            self.vad_speech_buffer,
                            timestamp_start,
                            now.timestamp(),
                        )
                    else:
                        print("VAD: Skipping segment, too short.")

                    # Reset state
                    self.vad_speech_buffer = np.array([], dtype=np.float32)
                    self.vad_is_speaking = False
                    self.vad_silence_counter = 0

    def _process_transcription_mode(self, chunk, now):
        """Process audio chunk for transcription mode (fixed buffer)."""
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
        if len(self.audio_buffer) >= self.max_buffer_samples:
            buffer_duration = len(self.audio_buffer) / self.RATE
            timestamp_start = (now - timedelta(seconds=buffer_duration)).timestamp()

            print(f"→ Buffer full: sending {buffer_duration:.1f}s segment for processing")
            self._add_audio_segment(
                self.chunk_counter,
                self.audio_buffer,
                timestamp_start,
                now.timestamp(),
            )
            self.audio_buffer = np.array([], dtype=np.float32)

    def start_recording(self):
        """Start audio recording and processing."""
        # Set filename for live mode and initialize it
        if self.filename is None:
            self.filename = self._generate_filename()
            self._init_output_file()

        if self.mode == "subtitle":
            self._ensure_vad_loaded()

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

    def _load_transcription_models(self):
        """Load transcription model(s) based on engine configuration."""
        if self.transcription_engine in ["faster-whisper", "auto"]:
            print("Loading faster-whisper model...")
            compute_type = "float16" if self.device.type == "cuda" else "float32"
            self.faster_whisper_model = WhisperModel(
                self.model_id, device=self.device.type, compute_type=compute_type
            )
        else:
            self.faster_whisper_model = None

        if self.transcription_engine in ["transformers", "auto"]:
            print("Loading transformers Whisper model...")
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            
            model_name = f"openai/whisper-{self.model_id}"
            self.transformers_processor = WhisperProcessor.from_pretrained(model_name)
            self.transformers_model = WhisperForConditionalGeneration.from_pretrained(
                model_name
            ).to(self.device)
            
            if self.device.type == "cuda":
                self.transformers_model = self.transformers_model.half()
        else:
            self.transformers_processor = None
            self.transformers_model = None

        # Log the configuration
        if self.transcription_engine == "auto":
            print(f"Auto-selection enabled: segments <{self.auto_engine_threshold}s will use transformers, ≥{self.auto_engine_threshold}s will use faster-whisper")
        elif self.transcription_engine == "faster-whisper":
            print("Using faster-whisper for all segments")
        else:
            print("Using transformers for all segments")

        # Set default model reference for compatibility
        if self.transcription_engine == "faster-whisper":
            self.model = self.faster_whisper_model
        elif self.transcription_engine == "transformers":
            self.model = self.transformers_model
        else:  # auto
            self.model = self.faster_whisper_model  # Default for non-transcription uses

    def _transcribe_segment(self, audio_data):
        """
        Transcribe audio segment with intelligent engine selection.
        
        Strategy:
        - Auto mode: Use transformers for short segments (< threshold), faster-whisper for long segments
        - Manual mode: Use the specified engine
        """
        audio_duration = len(audio_data) / self.RATE
        
        # Protection: Don't transcribe very short segments
        if audio_duration < 0.5:
            print(f"DEBUG: Segment too short ({audio_duration:.2f}s), skipping")
            return ""
        
        # IMPORTANT: Add silence padding to improve transcription quality
        # This prevents hallucinations at segment boundaries
        padding_duration = 1.5  # 1500ms of silence
        padding_samples = int(padding_duration * self.RATE)
        silence_padding = np.zeros(padding_samples, dtype=np.float32)
        
        # Pad audio with silence at start and end
        padded_audio = np.concatenate([silence_padding, audio_data, silence_padding])
        padded_duration = len(padded_audio) / self.RATE
        
        print(f"DEBUG: Original duration: {audio_duration:.2f}s, padded duration: {padded_duration:.2f}s")
        
        # Determine which engine to use
        if self.transcription_engine == "auto":
            # Use original duration (not padded) for threshold decision
            use_transformers = audio_duration < self.auto_engine_threshold
            engine_name = "transformers" if use_transformers else "faster-whisper"
            print(f"DEBUG: Auto-selecting {engine_name} for {audio_duration:.2f}s segment (threshold: {self.auto_engine_threshold}s)")
        elif self.transcription_engine == "transformers":
            use_transformers = True
            engine_name = "transformers"
        else:  # faster-whisper
            use_transformers = False
            engine_name = "faster-whisper"
        
        # Transcribe with selected engine using PADDED audio
        if use_transformers:
            return self._transcribe_with_transformers(padded_audio, padded_duration)
        else:
            return self._transcribe_with_faster_whisper(padded_audio, padded_duration)

if __name__ == "__main__":
    MODELS = [
        "tiny.en",
        "tiny",
        "base.en",
        "base",
        "small.en",
        "small",
        "medium.en",
        "medium",
        "large-v1",
        "large-v2",
        "large-v3",
        "large",
        "distil-large-v2",
        "distil-medium.en",
        "distil-small.en"
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
        default=0.7,
        help="Similarity threshold for speaker identification (0.0 to 1.0). Higher is stricter. Used in 'subtitle' mode or with '--diarization cluster'.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["subtitle", "transcription"],
        default="subtitle",
        help='Processing mode: "subtitle" for real-time display, "transcription" for merged, fluid text.',
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Minimum number of speakers to detect (optional). Used only in file mode.",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum number of speakers to detect (optional). Used only in file mode.",
    )
    parser.add_argument(
        "--diarization",
        type=str,
        choices=["pyannote", "cluster"],
        default="pyannote",
        help="Speaker diarization method. 'pyannote' is the default. 'cluster' uses VAD and speaker embeddings, and is available for file transcription or live subtitle mode.",
    )
    parser.add_argument(
        "--enhancement",
        type=str,
        choices=["none", "nsnet2", "demucs"],
        default="none",
        help="Audio enhancement method to apply before transcription.",
    )
    parser.add_argument(
        "--transcription-engine",
        type=str,
        choices=["auto", "faster-whisper", "transformers"],
        default="auto",
        help=(
            "Transcription engine to use. "
            "'auto' (default): intelligently selects based on segment duration and --auto-engine-threshold. "
            "'faster-whisper': always use faster-whisper (faster but may repeat on short segments). "
            "'transformers': always use transformers (slower but more stable)."
        ),
    )
    parser.add_argument(
        "--auto-engine-threshold",
        type=float,
        default=15.0,
        help=(
            "Duration threshold in seconds for auto engine selection (default: 15.0). "
            "Segments shorter than this will use transformers (more stable), "
            "longer segments will use faster-whisper (faster). "
            "Only used when --transcription-engine is 'auto'. "
            "Recommended range: 10-30 seconds."
        ),
    )

    args = parser.parse_args()

    # --- Argument Validation ---
    if (args.min_speakers is not None or args.max_speakers is not None) and not args.file:
        parser.error("--min-speakers and --max-speakers can only be used in file mode (--file).")

    if not args.file and args.mode == "transcription" and args.diarization == "cluster":
        parser.error(
            "In live mode (no --file), '--diarization cluster' is only compatible with '--mode subtitle'."
        )

    default_threshold = parser.get_default("threshold")
    if args.diarization == "pyannote" and args.threshold != default_threshold:
        warnings.warn(
            "Warning: --threshold is only used with '--diarization cluster'."
        )
    
    # Validation: warn if threshold is specified but engine is not auto
    if args.auto_engine_threshold != parser.get_default("auto_engine_threshold") and args.transcription_engine != "auto":
        warnings.warn(
            f"Warning: --auto-engine-threshold={args.auto_engine_threshold} will be ignored "
            f"because --transcription-engine is set to '{args.transcription_engine}' (not 'auto')."
        )
    # -------------------------

    transcriber = None
    try:
        transcriber = WhisperLiveTranscription(
            model_id=args.model,
            language=args.language,
            similarity_threshold=args.threshold,
            mode=args.mode,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            diarization_method=args.diarization,
            enhancement_method=args.enhancement,
            transcription_engine=args.transcription_engine,  # NEW
            auto_engine_threshold=args.auto_engine_threshold,  # NEW
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
                if transcriber.filename:
                    transcriber._write_to_file("", force_flush=True)
                
            else:
                # Mode live : utiliser stop_recording() qui fait tout le nettoyage
                transcriber.stop_recording()

        print("Done.")

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
