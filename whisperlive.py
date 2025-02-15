import queue
import threading
import time
import wave
from datetime import datetime, timedelta

import numpy as np
import pyaudio
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration


class WhisperLiveTranscription:
    def __init__(self, model_id="openai/whisper-small", language="english"):
        print("Launching WhisperLiveTranscription...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).to(
            self.device
        )
        self.processor = WhisperProcessor.from_pretrained(model_id)

        # Audio configuration
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 1024
        self.SILENCE_THRESHOLD = 0.001
        self.SILENCE_DURATION = 1.0
        self.MAX_BUFFER_DURATION = 5.0

        # Data management
        self.audio_queue = queue.Queue(maxsize=20)
        self.result_queue = queue.Queue(maxsize=5)
        self.transcription_queue = queue.Queue()
        self.audio_buffer = np.array([], dtype=np.float32)
        self.full_audio_buffer = np.array([], dtype=np.float32)
        self.silence_counter = 0
        self.chunk_timestamps = {}
        self.chunk_counter = 0

        # Derived calculations
        self.required_silent_chunks = int(
            self.SILENCE_DURATION * self.RATE / self.CHUNK
        )
        self.max_buffer_samples = int(self.MAX_BUFFER_DURATION * self.RATE)

        # Output file
        self.filename = f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self._init_output_file()

        # Execution control
        self.is_running = False
        self.language = language

    def _init_output_file(self):
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write(f"Transcription started on {datetime.now()}\n\n")
        print(f"Transcription file created: {self.filename}")

    def start_recording(self):
        self.is_running = True
        self.p = pyaudio.PyAudio()

        def audio_callback(in_data, frame_count, time_info, status):
            self.audio_queue.put(np.frombuffer(in_data, dtype=np.float32))
            return (in_data, pyaudio.paContinue)

        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
            stream_callback=audio_callback,
            start=False,
        )

        self.stream.start_stream()
        print("Recording started")

        # Démarrage des threads
        self.process_thread = threading.Thread(target=self._process_audio, daemon=True)
        self.transcribe_thread = threading.Thread(
            target=self._transcribe_audio, daemon=True
        )
        self.process_thread.start()
        self.transcribe_thread.start()
        print("Processing and transcription threads started")

    def _process_audio(self):
        """
        Function executed in a thread to process audio data from the microphone.

        It reads audio data from the `self.audio_queue` and accumulates it in
        `self.audio_buffer`. When it detects silence (i.e., when the audio level
        is below the `self.SILENCE_THRESHOLD` for a duration greater than or equal
        to `self.SILENCE_DURATION`), it sends the `self.audio_buffer` to the
        `self.result_queue` with a unique chunk identifier. If the `self.audio_buffer`
        becomes too large (i.e., greater than `self.max_buffer_samples`), it also
        sends it to the `self.result_queue` for safety reasons.

        The recorded chunks are stored in a dictionary `self.chunk_timestamps`
        which contains start and end timestamps for each chunk.

        The function stops when `self.is_running` is `False`.
        """
        print("Audio processing thread started")
        while self.is_running:
            try:
                chunk = self.audio_queue.get(timeout=0.5)
                start_time = datetime.now()

                # Accumulation des buffers
                self.audio_buffer = np.concatenate((self.audio_buffer, chunk))
                self.full_audio_buffer = np.concatenate((self.full_audio_buffer, chunk))

                # Détection de silence
                current_level = np.max(np.abs(chunk))
                if current_level < self.SILENCE_THRESHOLD:
                    self.silence_counter += 1
                    if self.silence_counter >= self.required_silent_chunks:
                        samples_to_keep = len(self.audio_buffer) - (
                            self.required_silent_chunks * self.CHUNK
                        )
                        if samples_to_keep > 0:
                            chunk_id = self.chunk_counter
                            end_time = start_time - timedelta(
                                seconds=(len(chunk) / self.RATE)
                            )
                            self.chunk_timestamps[chunk_id] = {
                                "start": end_time
                                - timedelta(seconds=(samples_to_keep / self.RATE)),
                                "end": end_time,
                            }
                            self.result_queue.put(
                                (chunk_id, self.audio_buffer[:samples_to_keep].copy())
                            )
                            print(
                                f"Audio segment detected (chunk {chunk_id}) – size: {samples_to_keep} samples"
                            )
                            self.chunk_counter += 1
                        self.audio_buffer = np.array([], dtype=np.float32)
                        self.silence_counter = 0
                else:
                    self.silence_counter = 0

                # Découpage de sécurité
                if len(self.audio_buffer) >= self.max_buffer_samples:
                    chunk_id = self.chunk_counter
                    self.chunk_timestamps[chunk_id] = {
                        "start": datetime.now()
                        - timedelta(seconds=(self.max_buffer_samples / self.RATE)),
                        "end": datetime.now(),
                    }
                    self.result_queue.put((chunk_id, self.audio_buffer.copy()))
                    print(f"Audio segment saved for safety (chunk {chunk_id})")
                    self.chunk_counter += 1
                    self.audio_buffer = np.array([], dtype=np.float32)

            except queue.Empty:
                continue

    def _transcribe_audio(self):
        """
        Continuously processes audio data from the result queue and transcribes it.

        This function runs in a separate thread, fetching audio segments from the
        `result_queue`, and performs the transcription using the model. The average
        timestamp for each audio segment is calculated and used to log the transcription
        result. The transcriptions are saved to a file, and also added to the
        `transcription_queue` for further processing or retrieval.

        The function handles queue timeout exceptions to ensure smooth operation
        and logs any errors encountered during transcription.

        The loop continues until the transcription process is no longer running
        and the `result_queue` is empty.
        """

        print("Audio transcription thread started")
        while self.is_running or not self.result_queue.empty():
            try:
                chunk_id, audio_data = self.result_queue.get(timeout=0.5)
                ts_data = self.chunk_timestamps.get(chunk_id)

                # Calcul du timestamp moyen
                mid_time = ts_data["start"] + (ts_data["end"] - ts_data["start"]) / 2
                timestamp_str = mid_time.strftime("%H:%M:%S")

                # Transcription
                features = self.processor(
                    audio_data, sampling_rate=self.RATE, return_tensors="pt"
                ).input_features.to(self.device)

                predicted_ids = self.model.generate(
                    features,
                    language=self.language,
                    task="transcribe",
                    max_length=200,
                    num_beams=1,
                    early_stopping=True,
                )

                transcription = self.processor.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )[0]

                # Enregistrement
                with open(self.filename, "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp_str}] {transcription}\n")
                print(
                    f"Transcribed segment (chunk {chunk_id}) at {timestamp_str}: {transcription}"
                )

                # Mise à jour des queues
                self.transcription_queue.put(
                    {"text": transcription, "timestamp": timestamp_str}
                )

                del self.chunk_timestamps[chunk_id]

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Transcription error: {str(e)}")

    def get_transcription(self, block=False, timeout=None):
        try:
            return self.transcription_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def stop_recording(self):
        """
        Stops the recording process, finalizes transcriptions, and cleans up resources.

        This method sets the `is_running` flag to False to halt the recording process.
        It closes the audio stream if it exists and terminates the audio processing.
        Any remaining audio data in the buffer is processed and added to the result
        queue with its timestamp information.

        It waits for the processing and transcription threads to finish using a timeout
        mechanism. Finally, it forces the transcription of any remaining audio segments
        and saves the complete transcription to a file.

        Note: Ensures all resources such as threads and streams are properly released
        and any pending audio data is transcribed.
        """

        print("\nFinalizing transcriptions...")
        self.is_running = False

        # Closing the audio stream
        if hasattr(self, "stream"):
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

        # Final buffer processing
        if self.audio_buffer.size > 0:
            chunk_id = self.chunk_counter
            self.chunk_timestamps[chunk_id] = {
                "start": datetime.now()
                - timedelta(seconds=(len(self.audio_buffer) / self.RATE)),
                "end": datetime.now(),
            }
            self.result_queue.put((chunk_id, self.audio_buffer))
            print(f"Final processing of audio buffer (chunk {chunk_id})")
            self.chunk_counter += 1

        # Waiting for threads
        self.process_thread.join(timeout=2)
        self.transcribe_thread.join(timeout=5)

        # Final transcriptions
        self._force_final_transcription()
        print(f"Complete transcription saved in {self.filename}")

    def _force_final_transcription(self):
        """
        Processes and transcribes any remaining audio segments in the result queue.

        This method is called to ensure that any remaining audio data in the
        `result_queue` is transcribed and saved. It processes each audio segment,
        calculates the approximate midpoint timestamp, performs transcription
        using the model, and appends the results to the output file.

        The transcription is similar to the `_transcribe_audio` method but is
        executed inline to finalize processing. If a segment's timestamp data
        is not available, a default timestamp of "00:00:00" is used.

        This method continues processing until the `result_queue` is empty.
        """

        print("Forcing processing of remaining segments...")
        while not self.result_queue.empty():
            try:
                chunk_id, audio_data = self.result_queue.get_nowait()
                # Similar to _transcribe_audio but inline to finish processing
                ts_data = self.chunk_timestamps.get(chunk_id)
                if ts_data:
                    mid_time = (
                        ts_data["start"] + (ts_data["end"] - ts_data["start"]) / 2
                    )
                    timestamp_str = mid_time.strftime("%H:%M:%S")
                else:
                    timestamp_str = "00:00:00"

                features = self.processor(
                    audio_data, sampling_rate=self.RATE, return_tensors="pt"
                ).input_features.to(self.device)
                predicted_ids = self.model.generate(
                    features,
                    language=self.language,
                    task="transcribe",
                    max_length=200,
                    num_beams=1,
                    early_stopping=True,
                )
                transcription = self.processor.batch_decode(
                    predicted_ids, skip_special_tokens=True
                )[0]
                with open(self.filename, "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp_str}] {transcription}\n")
                print(
                    f"Final transcribed segment (chunk {chunk_id}) at {timestamp_str}: {transcription}"
                )
            except queue.Empty:
                break

    def save_audio(self):
        """
        Saves the collected audio data to a WAV file.

        If the `full_audio_buffer` is not empty, the audio data is saved to a
        WAV file with a timestamped filename. The audio data is normalized to
        16-bit signed integers before saving.

        The saved audio file is printed to the console.

        """
        if len(self.full_audio_buffer) > 0:
            filename = f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

            # Conversion and normalization
            max_val = np.max(np.abs(self.full_audio_buffer))
            scaled = np.int16(self.full_audio_buffer * 32767 / max(max_val, 1e-5))

            with wave.open(filename, "wb") as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(self.RATE)
                wf.writeframes(scaled.tobytes())
            print(
                f"Audio saved ({len(self.full_audio_buffer)/self.RATE:.2f}s) in {filename}"
            )


if __name__ == "__main__":
    print("Starting live transcription script...")
    transcriber = WhisperLiveTranscription(language="french")
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
