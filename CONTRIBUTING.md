# Contributing to SenseVoice

Thank you for your interest in contributing to SenseVoice! This guide will help you get started.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- (Optional) CUDA-compatible GPU for faster inference

### Setting Up the Development Environment

1. **Fork and clone the repository:**

```bash
git clone https://github.com/<your-username>/SenseVoice.git
cd SenseVoice
```

2. **Create a virtual environment:**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Verify the installation:**

```bash
python -c "from funasr import AutoModel; print('Installation successful')"
```

### Running on CPU

If you don't have a GPU, you can run SenseVoice on CPU by setting the device to `"cpu"`:

```python
model = AutoModel(
    model="iic/SenseVoiceSmall",
    trust_remote_code=True,
    device="cpu",
)
```

For the FastAPI server:

```bash
export SENSEVOICE_DEVICE=cpu
fastapi run --port 50000
```

## How to Contribute

### Reporting Bugs

1. Search [existing issues](https://github.com/QwenAudio/SenseVoice/issues) to avoid duplicates.
2. Use the [Bug Report template](https://github.com/QwenAudio/SenseVoice/issues/new?template=bug_report.md).
3. Include your environment details (OS, Python version, PyTorch version, GPU, CUDA version).
4. Provide a minimal code sample to reproduce the issue.

### Resolving Issues

A merged pull request, a green `main` branch, or an available release does not by itself prove that a reported problem is resolved. Keep user-reported issues open until either:

- the reporter confirms the fix on the affected workflow; or
- a maintainer reproduces the original failure, verifies the fix in a publicly available version, records the evidence in the issue, and allows a reasonable feedback window.

When a fix needs a release, link the released version and ask the reporter to retest before closing. Use a waiting-for-feedback label when available instead of treating silence as confirmation.

### Submitting Pull Requests

1. **Fork the repository** and create a new branch from `main`:

```bash
git checkout -b your-branch-name
```

2. **Make your changes.** Keep commits focused and atomic.

3. **Test your changes** to make sure nothing is broken.

4. **Push to your fork** and open a Pull Request against `QwenAudio/SenseVoice:main`.

5. **Describe your changes** clearly in the PR description. Explain *what* changed and *why*.

### Types of Contributions We Welcome

- **Bug fixes** — Check [open issues labeled `bug`](https://github.com/QwenAudio/SenseVoice/issues?q=is%3Aissue+is%3Aopen+label%3Abug) for known problems.
- **Documentation improvements** — Typo fixes, clarifications, additional examples, translations.
- **New examples** — Demo scripts showing different use cases (emotion detection, event detection, multilingual transcription).
- **Performance improvements** — Optimizations for inference speed or memory usage.
- **Test coverage** — Unit tests and integration tests are very welcome.

### Code Style

- Follow existing code patterns and conventions in the repository.
- Use type hints where applicable.
- Add docstrings to new functions and classes.
- Keep lines under 120 characters.

## Project Structure

```
SenseVoice/
├── model.py          # Core SenseVoiceSmall model (encoder, CTC decoder, emotion/event embeddings)
├── api.py            # FastAPI server for inference
├── webui.py          # Gradio web interface
├── demo1.py          # Inference example using FunASR AutoModel
├── demo2.py          # Direct model inference with timestamp support
├── export.py         # ONNX model export
├── export_meta.py    # Export utilities for model rebuilding
├── finetune.sh       # Fine-tuning script with DeepSpeed support
├── requirements.txt  # Python dependencies
├── Dockerfile        # Docker build configuration
├── data/             # Example training and validation data
├── utils/            # Utilities (frontend, ONNX inference, CTC alignment, export)
└── image/            # Documentation images
```

## Understanding the Model

SenseVoice is a non-autoregressive encoder-only model that outputs:

- **Speech transcription** (ASR) across 50+ languages
- **Emotion labels**: `HAPPY`, `SAD`, `ANGRY`, `NEUTRAL`, `FEARFUL`, `DISGUSTED`, `SURPRISED`
- **Audio event labels**: `BGM`, `Speech`, `Applause`, `Laughter`, `Cry`, `Sneeze`, `Breath`, `Cough`
- **Language identification** for Mandarin, English, Cantonese, Japanese, and Korean

The model uses a SANM (Self-Attention with Normalized Memory) encoder architecture with CTC decoding. Emotion and event labels are predicted from the first 4 encoder output tokens, while the remaining tokens produce the transcription.

## Docker

You can also run SenseVoice using Docker:

```bash
# Build
docker build -t sensevoice .

# Run with GPU
docker run --gpus all -p 50000:50000 sensevoice

# Run on CPU
docker run -e SENSEVOICE_DEVICE=cpu -p 50000:50000 sensevoice
```

## Questions?

- Open an issue using the [Questions template](https://github.com/QwenAudio/SenseVoice/issues/new?template=ask_questions.md).
- Join the community via the DingTalk group (see [README](./README.md#community)).

## License

By contributing to SenseVoice, you agree that your contributions will be licensed under the same license as the project. See [FunASR License](https://github.com/modelscope/FunASR?tab=readme-ov-file#license) for details.
