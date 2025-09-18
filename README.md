# Whisper Live Diarization

This project provides a real-time audio transcription solution with speaker diarization, based on the Whisper, Pyannote.audio, and SpeechBrain models. The script can transcribe audio captured from a microphone or process an existing audio file, generating a timestamped transcription file that identifies the different speakers.

## Features

*   **Dual Processing Modes**:
    *   **`subtitle` mode**: Generates short, time-accurate segments ideal for live subtitling.
    *   **`transcription` mode**: Intelligently merges consecutive speech segments from the same speaker to produce a fluid, readable transcript, perfect for reports and documentation.
*   **Real-Time Transcription**: Transcribes speech live from the microphone.
*   **File Transcription**: Processes and transcribes long pre-recorded audio files efficiently.
*   **Speaker Diarization**: Detects who is speaking and when (`pyannote/speaker-diarization-3.1`).
*   **Speaker Re-identification**: Assigns a unique and consistent ID to each detected speaker, even across audio segments (`speechbrain/spkrec-ecapa-voxceleb`).
*   **Timestamped Output**: Generates a clean transcription with timestamps and speaker IDs for each speech segment.
*   **Audio Export**: Saves the entire audio session to a `.wav` file at the end of the recording (live mode only).
*   **Performance Optimized**: Uses `torch.compile` and `float16` precision on compatible GPUs for maximum performance.

## Technical Architecture

The system is built around several key technologies:

*   **Transcription Model**: Uses OpenAI's **Whisper** model family via the Hugging Face `transformers` library for high-quality speech recognition.
*   **Diarization Pipeline**: Employs **`pyannote.audio`** to segment the audio and assign local speaker tags to each turn of speech.
*   **Speaker Recognition**: Uses **`speechbrain`** (the `spkrec-ecapa-voxceleb` model) to create embeddings (voiceprints) from speech segments. These embeddings are compared to re-identify speakers consistently throughout the session.
*   **Segment Merging Logic**: A crucial step where speech segments are processed based on the selected mode. In `transcription` mode, consecutive segments from the same identified speaker are merged to form coherent paragraphs, ignoring silence gaps. In `subtitle` mode, segments remain short to ensure precise timing.
*   **Audio Capture**: Manages the real-time audio stream with **`pyaudio`**, chunking it into segments for processing.
*   **Parallel Processing**: Uses threads to separate audio capture from transcription, ensuring low latency and a responsive interface.
*   **Computation**: The script is optimized to use a GPU (CUDA) if available, which significantly speeds up model processing. It also runs on CPU.

## Installation

### Prerequisites

*   Python 3.9+
*   **For Linux users**: `portaudio` development files might be needed (`sudo apt-get install portaudio19-dev`).

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

3.  **Install PyTorch:**
    Install PyTorch first, following the official instructions for your specific CUDA version to ensure GPU acceleration.
    ```bash
    # Example for CUDA 12.1
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

4.  **Install Python dependencies:**
    Install the remaining packages from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure Hugging Face Access:**
    Diarization with `pyannote/speaker-diarization-3.1` requires authentication with Hugging Face.
    *   Create an account on [Hugging Face](https://huggingface.co/).
    *   Accept the terms of use for the [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) models on their respective pages.
    *   Generate an access token in your account settings (with at least `read` permissions).
    *   Create a `.env` file in the project root and add your token to it:
        ```
        # .env
        HUGGING_FACE_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        ```

## Usage

The script can be run in two modes: live transcription or file processing. The `--mode` argument allows you to choose the output format.

### 1. Live Transcription (from microphone)

To start real-time transcription, run the script without the `--file` argument.

```bash
# For a fluid, paragraph-style transcript
python whisperlive.py --model openai/whisper-small --language english --mode transcription

# For live, time-accurate subtitles
python whisperlive.py --model openai/whisper-small --language english --mode subtitle
```

*   The program will start listening and transcribing.
*   Press `Ctrl+C` to stop the recording.
*   Upon stopping, a final transcription (`transcription_YYYYMMDD_HHMMSS.txt`) and an audio file (`audio_YYYYMMDD_HHMMSS.wav`) will be saved.

### 2. Transcription from a File

To transcribe an existing audio file, use the `--file` argument.

```bash
# Generate a clean report from a meeting recording
python whisperlive.py --model openai/whisper-medium --language english --file "path/to/your/audio.wav" --mode transcription
```
The output will be saved to a descriptive file, such as `your_audio_transcription_openai_whisper-medium_thresh0.60.txt`, which includes the original filename, the model used, and the similarity threshold.

### Command-Line Arguments

*   `--model` (optional): Whisper model to use. Default: `openai/whisper-small`. Larger models are more accurate but slower.
*   `--language` (optional): Language of the audio (e.g., `french`, `english`). Default: `french`.
*   `--mode` (optional): Output mode. Can be `subtitle` (default) or `transcription`.
*   `--file` (optional): Path to the audio file to transcribe. If omitted, the script enters live mode.
*   `--threshold` (optional): Similarity threshold for speaker identification (0.0 to 1.0). Lower is less strict. Default: `0.60`.

## Configuration and Tuning

### Choosing the Right Mode

*   Use **`--mode subtitle`** when you need precise timing for each short phrase, such as for creating video subtitles or monitoring a live conversation turn-by-turn.
*   Use **`--mode transcription`** when your goal is a final, readable document, like a meeting summary or an article. This mode prioritizes text flow over exact timing of short pauses.

### Adjusting Speaker Identification

If you find the system is creating too many unique speakers (e.g., `Speaker_3`, `Speaker_4` for the same person), the similarity threshold is likely too strict.

*   **What it is**: The `similarity_threshold` is a value between 0 and 1 that determines how similar a new voice segment must be to an existing speaker's voiceprint to be matched with them.
*   **The Problem**: A high threshold is very strict. Minor variations in a person's voice can cause the similarity score to drop below the threshold, creating a new speaker ID unnecessarily.
*   **The Solution**: To make the system more tolerant and group speakers more effectively, **lower the threshold** using the `--threshold` command-line argument.

```bash
python whisperlive.py --file "path/to/your/audio.wav" --threshold 0.55
```
Good values to test are between `0.50` and `0.60`. Experiment to find the best balance for your audio.

## Contribution

Contributions to improve this project are welcome. Please follow these steps:

1.  **Fork the project.**
2.  **Create a feature branch** (`git checkout -b feature/NewFeature`).
3.  **Commit your changes** (`git commit -m 'Add some NewFeature'`).
4.  **Push to the branch** (`git push origin feature/NewFeature`).
5.  **Open a Pull Request.**

Please ensure your code is documented and follows the existing project style.
