import argparse
import os
import queue
import string
import time
import traceback
import wave
from contextlib import contextmanager
from datetime import datetime, timedelta

import numpy as np
import pyaudio
import torch
from dotenv import load_dotenv

# --- AJOUTS POUR LA DIARISATION ET LES EMBEDDINGS ---
from pyannote.audio import Pipeline
from speechbrain.pretrained import SpeakerRecognition
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ----------------------------------------------------


# ffmpeg -i input_video.ext -vn -acodec pcm_s16le -ac 1 -ar 16000 output_audio.wav


class WhisperLiveTranscription:
    def __init__(self, model_id="openai/whisper-small", language="english"):
        print("Launching WhisperLiveTranscription...")

        # --- Chargement du token depuis le .env ---
        load_dotenv()
        hf_token = os.getenv("HUGGING_FACE_TOKEN")
        if hf_token is None:
            print("Warning: HUGGING_FACE_TOKEN not found. Diarization might fail.")
        # -------------------------------------------

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(
            self.device
        )
        self.processor = WhisperProcessor.from_pretrained(model_id)

        # --- Initialisation des pipelines de diarisation et d'embedding ---
        print("Loading diarization and embedding models...")
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
        ).to(self.device)

        # Utilisation de SpeechBrain pour la reconnaissance (local après 1er dl)
        self.embedding_model = SpeakerRecognition.from_huggingface(
            "speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
        ).to(self.device)
        print("Models loaded.")
        # ----------------------------------------------------------------

        # --- Registre des locuteurs pour la ré-identification ---
        self.speaker_registry = {}
        self.next_speaker_id = 1
        self.similarity_threshold = (
            0.65  # Seuil pour considérer que c'est le même locuteur
        )
        # -------------------------------------------------------

        # Audio configuration - Use constants for better readability
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024
        self.SILENCE_THRESHOLD = 0.005
        self.SILENCE_DURATION = 2.0
        self.MAX_BUFFER_DURATION = 10.0
        self.OVERLAP_DURATION = 2.0
        # --- NOUVEAU : Fenêtre de recherche de silence ---
        self.SOFT_CUT_WINDOW_DURATION = (
            2.0  # Attendre jusqu'à 2s de plus pour un silence
        )

        # Pre-calculate constants
        self.required_silent_chunks = int(
            self.SILENCE_DURATION * self.RATE / self.CHUNK
        )
        self.max_buffer_samples = int(self.MAX_BUFFER_DURATION * self.RATE)
        # --- NOUVEAU : Taille maximale incluant la fenêtre de recherche ---
        self.max_soft_cut_buffer_samples = int(
            (self.MAX_BUFFER_DURATION + self.SOFT_CUT_WINDOW_DURATION) * self.RATE
        )
        self.overlap_samples = int(self.OVERLAP_DURATION * self.RATE)

        # Data management - Smaller queue sizes for better memory usage
        self.audio_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=3)
        self.transcription_queue = queue.Queue()

        # Initialize buffers
        self._reset_buffers()

        self.chunk_timestamps = {}
        self.chunk_counter = 0
        self.language = language
        self.is_running = False

        # Output file
        self.filename = f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self._init_output_file()

        # Pre-calculate forced_decoder_ids (avoid repeated calls)
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.language, task="transcribe"
        )

        # Nouveau : pour gérer les doublons
        self.last_transcription = ""
        self.last_timestamp = None
        self.overlap_threshold = 0.9  # Plus strict pour éviter les fausses détections

        # Pour l'horodatage relatif à la vidéo
        self.start_time = None  # Temps de début de l'enregistrement

    def _reset_buffers(self):
        """Reset audio buffers to empty state."""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.full_audio_buffer_list = []
        self.silence_counter = 0
        self.overlap_buffer = np.array([], dtype=np.float32)  # Nouveau

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
        # Vérifier la longueur minimale
        if len(audio_data) < self.RATE * 0.5:  # Moins de 0.5 secondes
            return ""

        # Préprocessing audio amélioré
        # Normalisation
        audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-8)

        # Filtrage simple pour réduire le bruit
        if len(audio_data) > 3:
            # Moyenne mobile simple pour lisser
            kernel = np.ones(3) / 3
            audio_data = np.convolve(audio_data, kernel, mode="same")

        processed_input = self.processor(
            audio_data, sampling_rate=self.RATE, return_tensors="pt"
        )
        features = processed_input.input_features.to(self.device)

        # Create attention mask
        attention_mask = torch.ones(
            features.shape[:2], dtype=torch.long, device=self.device
        )

        # Paramètres de génération améliorés
        predicted_ids = self.model.generate(
            features,
            attention_mask=attention_mask,
            forced_decoder_ids=self.forced_decoder_ids,
            max_length=448,  # Plus long pour capturer plus de contexte
            num_beams=2,  # Augmenté pour une meilleure qualité
            temperature=0.0,  # Déterministe
            do_sample=False,
            suppress_tokens=[],  # Ne pas supprimer de tokens
        )

        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        return transcription

    def _add_audio_segment(self, chunk_id, audio_data, start_time, end_time):
        """Unified method to add audio segment to result queue."""
        self.chunk_timestamps[chunk_id] = {"start": start_time, "end": end_time}
        self.result_queue.put((chunk_id, audio_data.copy()))
        self.chunk_counter += 1
        return self.chunk_counter - 1

    def _find_overlap_boundary(self, prev_text, current_text):
        """
        Finds the boundary of the overlapping text between two segments.
        This version works on a word-by-word basis for more robustness.
        """
        if not prev_text or not current_text:
            return 0

        # Fonction pour normaliser le texte et le diviser en mots
        def get_normalized_words(text):
            # Met en minuscule, supprime la ponctuation
            normalized = text.lower().translate(
                str.maketrans("", "", string.punctuation)
            )
            # Remplace les espaces multiples par un seul et retourne une liste de mots
            return normalized.split()

        prev_words_normalized = get_normalized_words(prev_text)
        current_words_normalized = get_normalized_words(current_text)
        prev_words_original = prev_text.split()
        current_words_original = current_text.split()

        if not prev_words_normalized or not current_words_normalized:
            return 0

        # Chercher le plus long suffixe commun
        max_possible_overlap = min(
            len(prev_words_normalized), len(current_words_normalized)
        )
        best_overlap_word_count = 0

        for n in range(max_possible_overlap, 0, -1):
            # Suffixe de la liste de mots précédente
            prev_suffix = prev_words_normalized[-n:]
            # Préfixe de la liste de mots actuelle
            current_prefix = current_words_normalized[:n]

            if prev_suffix == current_prefix:
                best_overlap_word_count = n
                break

        if best_overlap_word_count > 0:
            # Calculer la longueur en caractères de l'overlap à partir du texte original
            # pour gérer correctement la ponctuation et les espaces.
            # On prend la longueur du préfixe du texte courant qui correspond.
            overlapping_words = current_words_original[:best_overlap_word_count]
            # On ajoute la longueur des espaces entre les mots
            overlap_char_len = len(" ".join(overlapping_words))

            # Petite sécurité pour ne pas couper plus que le texte lui-même
            overlap_char_len = min(overlap_char_len, len(current_text))

            print(
                f"Found overlap of {best_overlap_word_count} words: '{' '.join(overlapping_words)}'"
            )
            return overlap_char_len

        return 0

    def _merge_overlapping_transcriptions(
        self, current_transcription, current_timestamp
    ):
        """
        Merges transcriptions by removing the overlapping part from the new transcription.
        If no perfect overlap is found, it keeps the new transcription as is.
        """
        if not self.last_transcription:
            self.last_transcription = current_transcription
            self.last_timestamp = current_timestamp
            return current_transcription, current_timestamp

        # Find the length of the overlapping text
        overlap_len = self._find_overlap_boundary(
            self.last_transcription, current_transcription
        )

        merged_text = current_transcription
        if overlap_len > 0:
            # Remove the overlapping part from the beginning of the current transcription
            merged_text = current_transcription[overlap_len:]

        # Update the last transcription for the next iteration
        # We use the full current transcription to have the correct context for the next segment
        self.last_transcription = current_transcription
        self.last_timestamp = current_timestamp

        return merged_text, current_timestamp

    def _calculate_video_timestamp(self, elapsed_seconds):
        """Calcule le timestamp relatif en secondes écoulées."""
        if elapsed_seconds < 0:
            elapsed_seconds = 0

        total_seconds = int(elapsed_seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _process_transcription(self, chunk_id, audio_data, is_final=False):
        """
        Enhanced transcription processing with diarization and speaker re-identification.
        """
        if not self._should_process_segment(audio_data):
            if chunk_id in self.chunk_timestamps:
                del self.chunk_timestamps[chunk_id]
            return

        ts_data = self.chunk_timestamps.get(chunk_id)
        segment_start_time_seconds = ts_data["start"] if ts_data else 0

        print(f"Diarizing segment {chunk_id}...")
        audio_for_diarization = {
            "waveform": torch.from_numpy(audio_data).unsqueeze(0),
            "sample_rate": self.RATE,
        }

        try:
            diarization = self.diarization_pipeline(audio_for_diarization)
        except Exception as e:
            print(f"Diarization failed for chunk {chunk_id}: {e}")
            return

        print(
            f"Diarization complete for segment {chunk_id}. Found {len(diarization.labels())} local speakers."
        )

        for turn, _, local_speaker_label in diarization.itertracks(yield_label=True):
            start_sample = int(turn.start * self.RATE)
            end_sample = int(turn.end * self.RATE)
            speaker_audio_segment = audio_data[start_sample:end_sample]

            if (
                len(speaker_audio_segment) < self.RATE * 0.5
            ):  # Ignorer les segments trop courts
                continue

            # --- Logique de ré-identification ---
            with torch.no_grad():
                embedding = self.embedding_model.encode_batch(
                    torch.from_numpy(speaker_audio_segment).unsqueeze(0).to(self.device)
                )
                embedding = embedding.squeeze().cpu().numpy()

            global_speaker_id = self._get_speaker_id(embedding.reshape(1, -1))
            # ------------------------------------

            speaker_turn_start_time = segment_start_time_seconds + turn.start
            transcription = self._transcribe_segment(speaker_audio_segment)

            if transcription.strip():
                timestamp_str = self._calculate_video_timestamp(speaker_turn_start_time)
                formatted_text = (
                    f"[{timestamp_str}][{global_speaker_id}] {transcription}"
                )

                with open(self.filename, "a", encoding="utf-8") as f:
                    f.write(formatted_text + "\n")

                prefix = "Final" if is_final else "Live"
                print(f"{prefix}: {formatted_text}")

                if not is_final:
                    self.transcription_queue.put(
                        {
                            "text": transcription,
                            "timestamp": timestamp_str,
                            "speaker": global_speaker_id,
                        }
                    )

        if chunk_id in self.chunk_timestamps:
            del self.chunk_timestamps[chunk_id]

    def _transcribe_audio(self):
        """Optimized transcription processing."""
        print("Audio transcription thread started")
        while self.is_running or not self.result_queue.empty():
            try:
                chunk_id, audio_data = self.result_queue.get(timeout=0.5)
                self._process_transcription(chunk_id, audio_data)
            except queue.Empty:
                continue
            except Exception:
                print("Transcription error:")
                traceback.print_exc()

    def _should_process_segment(self, audio_data):
        """Determine if segment should be processed based on content quality."""
        if len(audio_data) == 0:
            return False

        # Vérifier l'énergie du signal
        energy = np.mean(audio_data**2)
        if energy < 1e-6:  # Signal trop faible
            return False

        # Vérifier la durée minimale
        if len(audio_data) < self.RATE * 0.3:  # Moins de 300ms
            return False

        return True

    def _calculate_similarity(self, text1, text2):
        """Calcule la similarité entre deux textes."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def stop_recording(self):
        """Improved stop method with better resource management."""
        print("\nFinalizing transcriptions...")
        if not self.is_running:
            return

        self.is_running = False  # Signal threads to stop

        # Process final buffer if it exists
        if self.audio_buffer.size > 0:
            now = datetime.now()
            self._add_audio_segment(
                self.chunk_counter,
                self.audio_buffer,
                now - timedelta(seconds=len(self.audio_buffer) / self.RATE),
                now,
            )
            print(f"Final processing of audio buffer (chunk {self.chunk_counter - 1})")

        # Wait for threads with timeout
        if hasattr(self, "process_thread"):
            self.process_thread.join(timeout=2)
        if hasattr(self, "transcribe_thread"):
            self.transcribe_thread.join(timeout=5)

        self._force_final_transcription()
        print(f"Complete transcription saved in {self.filename}")

    def _force_final_transcription(self):
        """Process remaining segments with unified transcription method."""
        print("Forcing processing of remaining segments...")
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
        """Save audio with improved normalization."""
        if not self.full_audio_buffer_list:
            return

        # Concatenate all chunks at once
        self.full_audio_buffer = np.concatenate(self.full_audio_buffer_list)
        filename = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

        # Normalize to int16 range
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
            f"Audio saved ({len(self.full_audio_buffer) / self.RATE:.2f}s) in {filename}"
        )

    def _process_audio_stream(self, in_data, frame_count, time_info, status):
        """
        Callback function to process incoming audio chunks with 'soft cutting'.
        This replaces the previous, simpler audio handling logic.
        """
        now = datetime.now()
        chunk = np.frombuffer(in_data, dtype=np.float32)
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
        self.full_audio_buffer_list.append(chunk)

        is_silent = np.mean(np.abs(chunk)) < self.SILENCE_THRESHOLD

        # --- Logique de segmentation améliorée ---
        should_cut = False

        # 1. Coupure par détection de silence
        if is_silent:
            self.silence_counter += 1
            if self.silence_counter >= self.required_silent_chunks:
                should_cut = True
        else:
            self.silence_counter = 0

        # 2. Logique de coupure "douce" ou "nette"
        buffer_len = len(self.audio_buffer)

        # Si on dépasse la durée max, on entre dans la fenêtre de recherche de silence
        if not should_cut and buffer_len > self.max_buffer_samples:
            # Si on trouve un silence dans la fenêtre, on coupe.
            if is_silent:
                print("Soft cut triggered within search window.")
                should_cut = True
            # Si on atteint la limite absolue (durée max + fenêtre), on coupe de force.
            elif buffer_len >= self.max_soft_cut_buffer_samples:
                print("Hard cut triggered: max buffer size reached.")
                should_cut = True

        if (
            should_cut and len(self.audio_buffer) > self.RATE * 0.5
        ):  # Ne pas traiter les segments trop courts
            segment_duration = len(self.audio_buffer) / self.RATE
            self._add_audio_segment(
                self.chunk_counter,
                self.audio_buffer,
                (now - timedelta(seconds=segment_duration)).timestamp(),
                now.timestamp(),
            )

            # Conserver l'overlap pour le prochain segment
            self.overlap_buffer = self.audio_buffer[-self.overlap_samples :]
            self.audio_buffer = self.overlap_buffer.copy()
            self.silence_counter = 0

        return (in_data, pyaudio.paContinue)

    def start_recording(self):
        """Start recording and processing audio."""
        self.is_running = True
        self.start_time = datetime.now()
        self._reset_buffers()

        # Démarrer le thread de transcription
        self.transcribe_thread = threading.Thread(target=self._transcribe_audio)
        self.transcribe_thread.daemon = True
        self.transcribe_thread.start()

        # Utiliser le context manager pour le flux audio
        self.audio_context = self._audio_stream_context(
            callback=self._process_audio_stream
        )
        self.pyaudio_instance, self.stream = self.audio_context.__enter__()

        print("Starting audio stream...")
        self.stream.start_stream()

    def stop_recording(self):
        """Improved stop method with better resource management."""
        print("\nFinalizing transcriptions...")
        if not self.is_running:
            return

        self.is_running = False  # Signal threads to stop

        # Arrêter le flux audio proprement
        if hasattr(self, "stream") and self.stream.is_active():
            self.stream.stop_stream()
        if hasattr(self, "audio_context"):
            self.audio_context.__exit__(None, None, None)

        # Process final buffer if it exists
        if (
            self.audio_buffer.size > self.overlap_samples
        ):  # S'il y a plus que juste l'overlap
            now = datetime.now()
            self._add_audio_segment(
                self.chunk_counter,
                self.audio_buffer,
                now - timedelta(seconds=len(self.audio_buffer) / self.RATE),
                now,
            )
            print(f"Final processing of audio buffer (chunk {self.chunk_counter - 1})")

        # Wait for threads with timeout
        if hasattr(self, "process_thread"):
            self.process_thread.join(timeout=2)
        if hasattr(self, "transcribe_thread"):
            self.transcribe_thread.join(timeout=5)

        self._force_final_transcription()
        print(f"Complete transcription saved in {self.filename}")

    def _force_final_transcription(self):
        """Process remaining segments with unified transcription method."""
        print("Forcing processing of remaining segments...")
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
        """Save audio with improved normalization."""
        if not self.full_audio_buffer_list:
            return

        # Concatenate all chunks at once
        self.full_audio_buffer = np.concatenate(self.full_audio_buffer_list)
        filename = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

        # Normalize to int16 range
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
            f"Audio saved ({len(self.full_audio_buffer) / self.RATE:.2f}s) in {filename}"
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
        "audio_file",
        nargs="?",
        default=None,
        help="Path to audio file to transcribe. If not provided, starts live transcription.",
    )
    parser.add_argument(
        "--language", type=str, default="french", help="Language for transcription."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=MODELS,
        default="openai/whisper-small",
        help="Whisper model to use for transcription.",
    )

    args = parser.parse_args()
    transcriber = WhisperLiveTranscription(model_id=args.model, language=args.language)

    if args.audio_file:
        transcriber.transcribe_from_file(args.audio_file)
    else:
        print("Starting live transcription from microphone...")
        try:
            transcriber.start_recording()
            print("Recording in progress... Press Ctrl+C to stop")
            while True:
                result = transcriber.get_transcription()
                if result:
                    print(f"[{result['timestamp']}] {result['text']}")
                time.sleep(0.1)
        except KeyboardInterrupt:
            transcriber.stop_recording()
            transcriber.save_audio()
            print("Stopping the script")
