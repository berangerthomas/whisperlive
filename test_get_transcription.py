from whisperlive import WhisperLiveTranscription
import time

modeles = [
    "openai/whisper-tiny",
    "openai/whisper-base",
    "openai/whisper-small",
    "openai/whisper-medium",
    "openai/whisper-large",
    "openai/whisper-large-v2",
    "openai/whisper-large-v3",
    "openai/whisper-large-v3-turbo",
]

resu = []
try:
    transcriber = WhisperLiveTranscription(model_id=modeles[2], language="french")
    transcriber.start_recording()
    transcription = transcriber.get_transcription(block=False)
    print("Press Ctrl+C to stop...")

    while True:
        transcription = transcriber.get_transcription(block=False)
        if transcription:
            resu.append(transcription["text"])
            # print(f"{transcription['timestamp']}: {transcription['text']}")
        time.sleep(0.1)

except KeyboardInterrupt:
    try:
        transcriber.stop_recording()
    except Exception as e:
        print(f"Error stopping: {e}")
    finally:
        if hasattr(transcriber, "p"):
            try:
                transcriber.p.terminate()
            except:
                pass


print(resu)

# resu = []
# # Blocking retrieval with timeout
# try:
#     transcription = transcriber.get_transcription(block=False, timeout=5)
#     resu.append(transcription)
# except KeyboardInterrupt:
#     print("No transcription received within 5 seconds")

# transcriber.stop_recording()

# print(resu)
