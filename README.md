# WhisperLive: A Python Tool for Transcription and Speaker Diarization

WhisperLive is a Python application for generating speaker-aware transcriptions from live or pre-recorded audio. It integrates several machine learning models for its core functions: speech-to-text via OpenAI's Whisper model (using `faster-whisper` or Hugging Face `transformers` implementations), speaker diarization using `pyannote.audio`, and speaker embedding generation with `SpeechBrain`.

## Core Concepts and Methodology

This tool employs several techniques to ensure accurate and efficient transcription and diarization.

### 1. Hybrid Transcription Strategy

The choice of transcription engine is critical for balancing speed and accuracy.

-   **`faster-whisper`**: A reimplementation of Whisper using CTranslate2, offering significant performance gains (3-5x faster on GPU) through quantization and optimized computation. It excels on long, clear audio segments.
-   **`transformers`**: The official Hugging Face implementation. While slower, it exhibits greater stability on short, ambiguous, or noisy audio segments, reducing the risk of repetitive hallucinations.

The default **`auto`** engine leverages the strengths of both. It processes audio segments shorter than a configurable duration (`--auto-engine-threshold`) with `transformers` and longer segments with `faster-whisper`. This hybrid approach provides a robust balance suitable for most use cases.

### 2. Hallucination Mitigation via Silence Padding

Segment-based transcription can cause the model to generate repetitive or nonsensical text (hallucinations), especially at the start or end of a chunk. This script implements a specific technique to mitigate this:

-   **Technique**: 1.5 seconds of digital silence is prepended and appended to each audio chunk before it is passed to the Whisper model.
-   **Mechanism**: This padding provides a clean, unambiguous acoustic context for the model's encoder. It effectively resets the model's attention state, preventing it from carrying over erroneous patterns from a previous chunk's boundary.
-   **Timestamp Integrity**: The timestamps for the final transcription are calculated based on the original, un-padded audio segment's position in the stream, ensuring that this internal processing step does not affect the final temporal accuracy.

While not described in a formal publication, this technique is a practical method for improving the robustness of segment-based transcription.

### 3. Speaker Diarization Methodologies

The tool offers two distinct methods for speaker identification:

-   **`pyannote` (Default)**: This method uses the pre-trained, end-to-end `pyannote/speaker-diarization-3.1` pipeline [[2]](#2). It is a comprehensive model that performs speech segmentation, embedding extraction, and clustering in a single step, offering high accuracy, especially in conversations with overlapping speech.

-   **`cluster`**: This is a multi-step, manual pipeline:
    1.  **Voice Activity Detection (VAD)**: The audio is first segmented into speech and non-speech regions using the `Silero-VAD` model [[4]](#4).
    2.  **Speaker Embedding**: For each speech segment, a fixed-dimensional vector representation (an embedding, or "voiceprint") is extracted using the `speechbrain/spkrec-ecapa-voxceleb` model, which is based on the ECAPA-TDNN architecture [[3]](#3).
    3.  **Clustering**: All embeddings are grouped using agglomerative clustering based on cosine similarity to identify the unique speakers.

### 4. Intelligent Chunking for File Processing

When transcribing a file, simply splitting the audio into fixed-size chunks can cut sentences in half and destroy conversational context. This script employs a more intelligent approach:

1.  The entire file is first diarized to identify all speaker turns.
2.  These turns are then grouped into optimal chunks of 60-120 seconds.
3.  Crucially, the cuts between chunks are made at natural pauses (silences) between speaker segments.

This method ensures that each chunk sent for transcription contains coherent conversational context, improving the accuracy and readability of the output.

## Installation

### Prerequisites

-   Python 3.11+
-   [uv](https://github.com/astral-sh/uv): A Python package installer.
-   A Hugging Face account and token.
-   **(Linux)**: `portaudio` development files may be required (`sudo apt-get install portaudio19-dev`).

### Setup

1.  **Clone Repository**: `git clone https://github.com/berangerthomas/whisperlive.git && cd whisperlive`
2.  **Install Dependencies**:
    ```bash
    uv venv
    uv sync
    ```
3.  **Activate Environment**:
    -   Windows: `.venv\Scripts\activate`
    -   Linux/macOS: `source .venv/bin/activate`
4.  **Install PyTorch for GPU (Optional)**:
    ```bash
    # Example for CUDA 12.1
    uv pip install torch --index-url https://download.pytorch.org/whl/cu121
    ```
5.  **Configure Hugging Face**: Create a `.env` file with your token:
    ```
    HUGGING_FACE_TOKEN="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    ```
    You must also accept the user agreements for `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` on the Hugging Face Hub.

## Command-Line Reference

| Argument | Default | Description |
|---|---|---|
| `--language <lang>` | `fr` | Language code (e.g., `en`, `es`, `de`). |
| `--model <id>` | `large-v3` | Whisper model [[1]](#1). Choices: `tiny`, `base`, `small`, `medium`, `large-v1/v2/v3`, and distilled variants. |
| `--file <path>` | `None` | Path to a `.wav` file. If omitted, runs in live mode. |
| `--mode <mode>` | `subtitle` | Output mode. Choices: `subtitle`, `transcription`. |
| `--diarization <method>` | `pyannote` | Diarization method. Choices: `pyannote`, `cluster`. |
| `--threshold <float>` | `0.7` | **(Cluster only)** Similarity threshold for speaker clustering (0.0-1.0). |
| `--min-speakers <int>` | `None` | **(File mode only)** Hint for the minimum number of speakers. |
| `--max-speakers <int>` | `None` | **(File mode only)** Hint for the maximum number of speakers. |
| `--transcription-engine <engine>` | `auto` | Engine. Choices: `auto`, `faster-whisper`, `transformers`. |
| `--auto-engine-threshold <float>` | `15.0` | **(Auto mode only)** Time in seconds to switch from `transformers` to `faster-whisper`. |
| `--enhancement <method>` | `none` | Audio enhancement. Choices: `none`, `nsnet2` [[6]](#6), `demucs` [[5]](#5). |

## Usage Scenarios

### Scenario 1: Live Subtitling of a Presentation

**Goal**: Low-latency, real-time captions for a single speaker.
**Rationale**: `cluster` is faster than `pyannote`. A lower threshold (`0.6`) prevents voice modulation from creating a new speaker identity. `nsnet2` provides lightweight noise reduction.
```bash
python whisperlive.py \
  --mode subtitle \
  --language en \
  --diarization cluster \
  --threshold 0.6 \
  --enhancement nsnet2
```

### Scenario 2: Generating a Transcript of a Recorded Interview

**Goal**: High-accuracy transcript of a 2-person conversation.
**Rationale**: `pyannote` is highly accurate for diarization. Specifying `--min-speakers 2 --max-speakers 2` constrains the model for optimal results. `transcription` mode creates a clean, readable document.
```bash
python whisperlive.py \
  --file "interview.wav" \
  --mode transcription \
  --min-speakers 2 \
  --max-speakers 2
```

### Scenario 3: Transcribing a Noisy Field Recording

**Goal**: Extract intelligible speech from a noisy environment.
**Rationale**: `demucs` is a powerful source separation model that can isolate vocals. The resulting audio may be fragmented, so the `transformers` engine is used for its stability on short segments.
```bash
python whisperlive.py \
  --file "field_recording.wav" \
  --mode transcription \
  --enhancement demucs \
  --transcription-engine transformers
```

### Scenario 4: Processing a Multi-Speaker Focus Group Recording

**Goal**: Transcribe a complex conversation with an unknown number of speakers.
**Rationale**: `pyannote` excels in complex, multi-speaker scenarios. The default `auto` engine will balance speed and accuracy across varying segment lengths.
```bash
python whisperlive.py \
  --file "focus_group.wav" \
  --mode transcription \
  --diarization pyannote
```

## Output File Naming

All transcriptions are saved to a `.txt` file with a name generated from the configuration settings.

-   **Format**: `{base_name}_{mode}_{model}_{diarization}_{details}_{timestamp}.txt`
-   **Components**:
    -   `base_name`: The original filename, or `live` for microphone recordings.
    -   `mode`: `subtitle` or `transcription`.
    -   `model`: The Whisper model used (e.g., `large-v3`).
    -   `diarization`: `pyannote` or `cluster`.
    -   `details`: `threshX.XX` for `cluster` mode, or `N-speakers` for file-based `pyannote` mode.
    -   `timestamp`: A `YYYYMMDD_HHMMSS` string added to live recordings.
-   **Examples**:
    -   `interview_transcription_large-v3_pyannote_2-speakers.txt`
    -   `live_subtitle_small_cluster_thresh0.60_20251002_213000.txt`

## Troubleshooting

### Repetitive or Inaccurate Text

-   **Symptom**: The transcription repeats a phrase or contains nonsensical text.
-   **Cause**: This can occur when short or silent segments are passed to the transcription model.
-   **Solution**:
    1.  Lower `--auto-engine-threshold` (e.g., to `10.0`) to use the `transformers` engine on more segments.
    2.  Force the `transformers` engine for the entire run with `--transcription-engine transformers`.

### Incorrect Speaker Labels

-   **Symptom**: One person is labeled as multiple speakers, or multiple people are merged.
-   **Solution**:
    1.  If using `--diarization cluster`, adjust the `--threshold`. A lower value merges more easily; a higher value separates more easily.
    2.  If using `--diarization pyannote` on a file, provide speaker hints with `--min-speakers` and `--max-speakers`.

### Slow Performance

-   **Symptom**: Transcription speed is significantly slower than real-time.
-   **Solution**:
    1.  Ensure the GPU-accelerated version of PyTorch is installed and being utilized.
    2.  Use a smaller model (e.g., `--model medium` or `--model small`).
    3.  For clean audio, force the faster engine with `--transcription-engine faster-whisper`.

## Bibliography

1.  <a name="1"></a>Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision*. [arXiv:2212.04356](https://arxiv.org/abs/2212.04356).
2.  <a name="2"></a>Bredin, H., Yin, R., Coria, J. M., Gelly, G., Korshunov, P., Lavechin, M., Fustes, D., Titeux, H., Bouaziz, W., & Gill, M. (2020). *pyannote.audio: neural building blocks for speaker diarization*. [arXiv:1911.01255](https://arxiv.org/abs/1911.01255).
3.  <a name="3"></a>Desplanques, B., Thienpondt, J., & Demuynck, K. (2020). *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification*. [arXiv:2005.07143](https://arxiv.org/abs/2005.07143).
4.  <a name="4"></a>Silero Team. (2024). *Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier*. [GitHub repository](https://github.com/snakers4/silero-vad).
5.  <a name="5"></a>Défossez, A., Usunier, N., Bottou, L., & Bach, F. (2019). *Music Source Separation in the Waveform Domain*. [arXiv:1911.13254](https://arxiv.org/abs/1911.13254).
6.  <a name="6"></a>Braun, S., Gamper, H., Reddy, C. K. A., & Cutler, R. (2021). *Towards efficient models for real-time deep noise suppression*. In *ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*.

## License

This project is licensed under the MIT License.
