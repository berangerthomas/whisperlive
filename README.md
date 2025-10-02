# Whisper Live Diarization

This project provides a real-time audio transcription solution with speaker diarization, based on the Faster-Whisper, Pyannote.audio, and SpeechBrain models. The script can transcribe audio captured from a microphone or process an existing audio file, generating a timestamped transcription file that identifies the different speakers.

## Features

*   **Dual Processing Modes**:
    *   **`subtitle` mode**: Generates short, time-accurate segments ideal for live subtitling or creating precisely timed subtitle files (e.g., `.srt`).
    *   **`transcription` mode**: Intelligently merges consecutive speech segments from the same speaker to produce a fluid, readable transcript, perfect for reports and documentation.
*   **Flexible Diarization Methods**:
    *   **`pyannote` (default)**: Uses the powerful `pyannote/speaker-diarization-3.1` pipeline for robust speaker identification.
    *   **`cluster`**: Uses a VAD (Voice Activity Detection) model for segmentation followed by speaker embedding clustering. This method can be faster and is a great alternative.
*   **Real-Time & File-Based**: Works seamlessly for live transcription from a microphone and for processing pre-recorded audio files.
*   **Speaker Re-identification**: Assigns a unique and consistent ID to each detected speaker throughout the session (`speechbrain/spkrec-ecapa-voxceleb`).
*   **Timestamped Output**: Generates a clean transcription with timestamps and speaker IDs.
*   **Audio Export**: Saves the entire audio session to a `.wav` file at the end of a live recording.
*   **Performance Optimized**: Uses `faster-whisper` with `float16` precision on compatible GPUs for maximum performance.

## Technical Architecture

The system is built around several key technologies:

*   **Transcription Model**: Uses the **Whisper** model family via the high-performance `faster-whisper` library.
*   **Diarization Pipeline**:
    *   **`pyannote.audio`**: The default method, which segments audio and assigns speaker tags.
    *   **VAD + Clustering**: An alternative method using Silero VAD for speech detection and `speechbrain` for creating and clustering voiceprints (embeddings).
*   **Speaker Recognition**: Uses **`speechbrain`** (`spkrec-ecapa-voxceleb`) to create robust voiceprints for speaker identification.
*   **Segment Merging Logic**: In `transcription` mode, consecutive segments from the same speaker are merged to form coherent paragraphs. In `subtitle` mode, segments remain short for precise timing.
*   **Audio Capture**: Manages the real-time audio stream with **`pyaudio`**.
*   **Parallel Processing**: Uses threads to separate audio capture from transcription, ensuring low latency.
*   **Computation**: Optimized for GPU (CUDA) usage but runs efficiently on CPU as well.

## Installation

### Prerequisites

*   Python 3.11+
*   [uv](https://github.com/astral-sh/uv), an extremely fast Python package installer.
*   **For Linux users**: `portaudio` development files might be needed (`sudo apt-get install portaudio19-dev`).

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/berangerthomas/whisperlive.git
    cd whisperlive
    ```

2.  **Create a virtual environment and install dependencies with `uv`:**
    ```bash
    uv venv
    uv sync
    ```
    This command installs all necessary dependencies for the project.

    Then, activate the environment:

    *   **On Windows:**
        ```bash
        .venv\Scripts\activate
        ```
    *   **On Linux/macOS:**
        ```bash
        source .venv/bin/activate
        ```

3.  **Install PyTorch (for GPU acceleration):**
    To benefit from GPU acceleration, install the PyTorch version matching your CUDA version.
    ```bash
    # Example for CUDA 12.1
    uv pip install torch --index-url https://download.pytorch.org/whl/cu121
    ```
    If you don't have a GPU, the PyTorch version installed by `uv sync` is sufficient.

4.  **Configure Hugging Face Access:**
    Diarization with `pyannote/speaker-diarization-3.1` requires authentication with Hugging Face.
    *   Create an account on [Hugging Face](https://huggingface.co/).
    *   Accept the terms of use for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).
    *   Generate an access token in your account settings (with `read` permissions).
    *   Create a `.env` file in the project root and add your token:
        ```
        # .env
        HUGGING_FACE_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        ```

## Usage

The script can be run for live transcription or file processing, with different modes and diarization methods.

### 1. Live Transcription (from microphone)

To start real-time transcription, run the script without the `--file` argument.

```bash
# Live subtitles with the fast 'cluster' diarization method
python whisperlive.py --model small --language fr --mode subtitle --diarization cluster

# Live transcription into paragraphs with the default 'pyannote' method
python whisperlive.py --model small --language fr --mode transcription
```

*   The program will start listening and transcribing.
*   Press `Ctrl+C` to stop.
*   A transcription file (`transcription_*.txt`) and an audio recording (`audio_*.wav`) will be saved.

### 2. Transcription from a File

To transcribe an existing audio file, use the `--file` argument.

```bash
# Generate a clean report from a meeting recording using VAD + clustering
python whisperlive.py --model large-v3 --language fr --file "path/to/audio.wav" --mode transcription --diarization cluster --threshold 0.7

# Generate a subtitle file using pyannote
python whisperlive.py --model large-v3 --language fr --file "path/to/audio.wav" --mode subtitle --diarization pyannote
```
The output will be saved to a descriptive filename, including the original name, mode, model, and diarization settings.

### Command-Line Arguments

*   `--model`: Whisper model to use (e.g., `tiny`, `base`, `small`, `medium`, `large-v3`). Default: `large-v3`.
*   `--language`: Language of the audio (e.g., `fr`, `en`, `es`, etc.). Default: `fr`.
*   `--mode`: Output mode. `subtitle` (default) for time-accurate segments, or `transcription` for fluid paragraphs.
*   `--file`: Path to the audio file to transcribe. If omitted, the script enters live mode.
*   `--diarization`: Speaker diarization method. `pyannote` (default) or `cluster`. The `cluster` method is not available for live transcription mode (`--mode transcription` without `--file`).
*   `--threshold`: Similarity threshold (0.0 to 1.0) for the `cluster` diarization method. Higher is stricter. Default: `0.7`.
*   `--min-speakers`, `--max-speakers`: (File mode only) Hint for the number of speakers.
*   `--enhancement`: Audio enhancement method (`none`, `nsnet2`, `demucs`). Default: `none`.

### Audio Enhancement

You can improve transcription accuracy in noisy environments by applying audio enhancement before processing. This is especially useful for cleaning up background noise or separating voice from music.

Use the `--enhancement` argument with one of the following methods:

-   `none`: (Default) No enhancement is applied.
-   `nsnet2`: Recommended for **live transcription**. Uses the lightweight and fast NSNet2 model for real-time noise suppression.
-   `demucs`: Recommended for **file transcription**. Uses the powerful Demucs model to separate vocals from other sounds. It provides very high quality but has significant processing overhead, making it unsuitable for live mode.

**Example (Live Mode with NSNet2):**

```bash
python whisperlive.py --model large-v3 --enhancement nsnet2
```

**Example (File Mode with Demucs):**

```bash
python whisperlive.py --file my_audio.wav --enhancement demucs
```

## Configuration and Tuning

### Choosing the Right Diarization Method

*   **`--diarization pyannote`**: The most robust and accurate method, especially for complex conversations. It is the recommended default.
*   **`--diarization cluster`**: A faster alternative that uses VAD for speech detection and embedding clustering. It performs very well and can be more efficient, especially in live subtitle mode. It is not available for live transcription mode.

### Adjusting Speaker Identification (`cluster` mode)

If the `cluster` method creates too many unique speakers (e.g., `SPEAKER_03`, `SPEAKER_04` for the same person), the similarity threshold may be too high.

*   **What it is**: The `--threshold` determines how similar a new voice segment must be to an existing speaker's voiceprint to be matched.
*   **The Solution**: To make the system more tolerant, **lower the threshold**. Good values to test are between `0.6` and `0.8`.

```bash
# Use a more tolerant threshold for a file with voice variations
python whisperlive.py --file "path/to/audio.wav" --diarization cluster --threshold 0.65
```

## Contribution

Contributions are welcome. Please fork the project, create a feature branch, and open a Pull Request.
