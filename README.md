# Whisper Live Diarization

This project provides a real-time audio transcription solution with speaker diarization, based on the Whisper, Pyannote.audio, and SpeechBrain models. The script can transcribe audio captured from a microphone or process an existing audio file, generating a timestamped transcription file that identifies the different speakers.

## Features

*   **Real-Time Transcription**: Transcribes speech live from the microphone.
*   **File Transcription**: Processes and transcribes pre-recorded audio files.
*   **Speaker Diarization**: Detects who is speaking and when (`pyannote/speaker-diarization-3.1`).
*   **Speaker Re-identification**: Assigns a unique and consistent ID to each detected speaker, even across audio segments (`speechbrain/spkrec-ecapa-voxceleb`).
*   **Intelligent Audio Segmentation**: Uses silence detection and a "soft cut" window to avoid splitting words, thereby improving transcription quality.
*   **Overlap-based Merging**: Manages unavoidable segment cuts by detecting and merging overlapping transcriptions for a smooth, repetition-free result.
*   **Timestamped Output**: Generates a clean transcription with timestamps and speaker IDs for each speech segment.
*   **Audio Export**: Saves the entire audio session to a `.wav` file at the end of the recording.

## Technical Architecture

The system is built around several key technologies:

*   **Transcription Model**: Uses OpenAI's **Whisper** model family via the Hugging Face `transformers` library for high-quality speech recognition.
*   **Diarization Pipeline**: Employs **`pyannote.audio`** to segment the audio and assign local speaker tags to each turn of speech.
*   **Speaker Recognition**: Uses **`speechbrain`** (the `spkrec-ecapa-voxceleb` model) to create embeddings (voiceprints) from speech segments. These embeddings are compared to re-identify speakers consistently throughout the session.
*   **Audio Capture**: Manages the real-time audio stream with **`pyaudio`**, chunking it into segments for processing.
*   **Parallel Processing**: Uses threads to separate audio capture from transcription, ensuring low latency and a responsive interface.
*   **Computation**: The script is optimized to use a GPU (CUDA) if available, which significantly speeds up model processing. It also runs on CPU.

## Installation

### Prerequisites

*   Python 3.8+
*   FFmpeg
*   Git

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/whisperlive.git
    cd whisperlive
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Python dependencies:**
    The project relies on several libraries. Install them via `pip`.
    ```bash
    pip install torch transformers "pyannote.audio>=3.1" speechbrain pyaudio numpy python-dotenv
    ```
    *Note:* Installing `PyAudio` may require system dependencies like `portaudio`. Please consult the `PyAudio` documentation for your operating system if you encounter issues. For `torch`, ensure you install the version compatible with your hardware (CPU or CUDA GPU).

4.  **Configure Hugging Face Access:**
    Diarization with `pyannote/speaker-diarization-3.1` requires authentication with Hugging Face.
    *   Create an account on [Hugging Face](https://huggingface.co/).
    *   Accept the terms of use for the `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` models on their respective pages.
    *   Generate an access token in your account settings.
    *   Create a `.env` file in the project root and add your token to it:
        ```
        # .env
        HUGGING_FACE_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        ```

## Usage

The script can be run in two modes: live transcription or file processing.

### 1. Live Transcription (from microphone)

To start real-time transcription, simply run the script without a file argument. You can specify the Whisper model and language.

```bash
python whisperlive.py --model openai/whisper-small --language english
```

*   The program will start listening and transcribing.
*   Press `Ctrl+C` to stop the recording.
*   Upon stopping, a final transcription (`transcription_YYYYMMDD_HHMMSS.txt`) and an audio file (`audio_YYYYMMDD_HHMMSS.wav`) will be saved.

### 2. Transcription from a File

To transcribe an existing audio file, provide the file path as the first argument. The file must be in a format that FFmpeg can read (e.g., `.wav`, `.mp3`, `.m4a`).

```bash
python whisperlive.py path/to/your/file.wav --model openai/whisper-medium --language english
```

### Command-Line Arguments

*   `audio_file` (optional): Path to the audio file to transcribe. If omitted, the script enters live mode.
*   `--language` (optional): Language of the audio (e.g., `french`, `english`). Default: `french`.
*   `--model` (optional): Whisper model to use. Default: `openai/whisper-small`. Larger models are more accurate but slower.

## Contribution

Contributions to improve this project are welcome. Here are some guidelines:

1.  **Fork the project.**
2.  **Create a feature branch** (`git checkout -b feature/NewFeature`).
3.  **Commit your changes** (`git commit -m 'Add some NewFeature'`).
4.  **Push to the branch** (`git push origin feature/NewFeature`).
5.  **Open a Pull Request.**

Please ensure your code is documented and follows the existing project style.
