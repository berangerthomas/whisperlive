import numpy as np
import pyaudio
import torch
import queue
import threading
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import wave
from datetime import datetime
import time


class WhisperLiveTranscription:
    def __init__(self, model_id="openai/whisper-large-v3", language="french"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Utilisation du device: {self.device}")

        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(
            self.device
        )

        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024
        self.RECORD_SECONDS = 3
        self.SILENCE_THRESHOLD = 0.01

        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_running = False
        self.language = language

        self.audio_buffer = []
        self.last_process_time = time.time()

    def start_recording(self):
        self.is_running = True
        self.p = pyaudio.PyAudio()

        def audio_callback(in_data, frame_count, time_info, status):
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_queue.put(audio_data)
            return (in_data, pyaudio.paContinue)

        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=audio_callback,
        )

        print("Enregistrement démarré")
        self.stream.start_stream()

        self.process_thread = threading.Thread(target=self._process_audio)
        self.transcribe_thread = threading.Thread(target=self._transcribe_audio)
        self.process_thread.start()
        self.transcribe_thread.start()

    def _process_audio(self):
        while self.is_running:
            try:
                raw_chunk = self.audio_queue.get(timeout=1)
                audio_chunk = raw_chunk  # Pas de traitement

                if np.max(np.abs(audio_chunk)) > self.SILENCE_THRESHOLD:
                    self.audio_buffer.extend(audio_chunk)

                current_time = time.time()
                if current_time - self.last_process_time >= self.RECORD_SECONDS:
                    if len(self.audio_buffer) > 0:
                        audio_data = np.array(self.audio_buffer)
                        self.result_queue.put(audio_data)
                        self.audio_buffer = []
                        self.last_process_time = current_time

            except queue.Empty:
                continue

    def _transcribe_audio(self):
        while self.is_running:
            try:
                audio_data = self.result_queue.get(timeout=1)
                input_features = self.processor(
                    audio_data, sampling_rate=self.RATE, return_tensors="pt"
                ).input_features.to(self.device)

                with torch.no_grad():
                    predicted_ids = self.model.generate(
                        input_features,
                        language=self.language,
                        task="transcribe",
                        max_length=448,
                        no_repeat_ngram_size=3,
                    )

                transcription = self.processor.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )[0]

                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] {transcription}")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Erreur lors de la transcription: {e}")
                continue

    def stop_recording(self):
        print("\nArrêt en cours... Traitement des derniers segments audio...")

        if hasattr(self, "stream"):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, "p"):
            self.p.terminate()

        if len(self.audio_buffer) > 0:
            final_audio = np.array(self.audio_buffer)
            self.result_queue.put(final_audio)

        self.is_running = False

        while not self.audio_queue.empty():
            chunk = self.audio_queue.get()
            self.audio_buffer.extend(chunk)

        if len(self.audio_buffer) > 0:
            final_audio = np.array(self.audio_buffer)
            self.result_queue.put(final_audio)

        if hasattr(self, "process_thread"):
            self.process_thread.join()
        if hasattr(self, "transcribe_thread"):
            timeout = 5
            self.transcribe_thread.join(timeout=timeout)
            if self.transcribe_thread.is_alive():
                print(
                    "Attention: La dernière transcription n'a pas pu être terminée dans le temps imparti"
                )

        print("Enregistrement et transcription terminés")

    def save_audio(self, filename):
        if len(self.audio_buffer) > 0:
            wf = wave.open(filename, "wb")
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b"".join(self.audio_buffer))
            wf.close()
            print(f"Audio sauvegardé dans {filename}")


if __name__ == "__main__":
    try:
        transcriber = WhisperLiveTranscription(
            model_id="openai/whisper-medium", language="french"
        )
        transcriber.start_recording()

        print("Appuyez sur Ctrl+C pour arrêter...")
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        try:
            transcriber.stop_recording()
        except Exception as e:
            print(f"Erreur lors de l'arrêt: {e}")
        finally:
            if hasattr(transcriber, "p"):
                try:
                    transcriber.p.terminate()
                except:
                    pass
