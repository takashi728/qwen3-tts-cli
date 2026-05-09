# qwen3-tts-cli

Command-line interface for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — voice clone, voice design, and custom voice generation.

## Setup

### 🤖 Automatic (recommended)

Let an agent set it up for you. Point your agent at this repo and say:

> *"Clone qwen3-tts-cli, read SKILL.md, then install it and generate a test sentence."*

Agents that handle this well:

| Agent | Why |
|---|---|
| [**Pi**](https://github.com/earendil-works/pi) | Open-source. Reads SKILL.md, manages venvs, editable installs, full CLI awareness. |
| [**OpenCode**](https://github.com/opencode-ai/opencode) | Open-source. Native tool-calling, zero-config Python/venv workflows. |

Both are fully open-source — no paywalls, no API keys needed for local use.

### 🔧 Manual

```bash
git clone https://github.com/takashi728/qwen3-tts-cli.git
cd qwen3-tts-cli

# With uv (fast)
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .

# Or with pip
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Requirements: **Python ≥ 3.10**, **NVIDIA GPU** with ≥ 8GB VRAM, **CUDA** toolkit.

FlashAttention 2 reduces VRAM usage ~20% (optional):

```bash
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
```

If flash-attn fails to build (e.g. missing CUDA_HOME), the CLI falls back to PyTorch SDPA — works fine on 32GB cards.

## Quickstart

```bash
# List available speakers
qwen3-tts speakers

# List supported languages
qwen3-tts languages

# Speak with a preset voice (small 0.6B model - fastest)
qwen3-tts speak -T "Hello, this is a test." -s Ryan -l English

# Speak with instruction control (1.7B model)
qwen3-tts speak -m custom-1.7b -T "今天天气真好" -s Vivian -i "用开心的语气说"

# Voice Design: describe a voice
qwen3-tts design -T "Welcome to the show!" -l English -i "Deep, warm, authoritative male voice, slow pace"

# Voice Clone: clone from a reference audio
qwen3-tts clone -T "This is my cloned voice." -l English -r reference.wav -rt "transcript of reference"

# Voice Clone without transcript (x-vector only)
qwen3-tts clone -T "Cloned speech." -r reference.wav --x-vector-only

# Design + Clone pipeline: create a voice, then reuse it
qwen3-tts design-clone \
  -i "Shy teenage male, tenor, nervous, stammering slightly" \
  --ref-text "H-hey! You dropped your... uh... notebook?" \
  --synth-text "No problem! I actually finished those already." \
  --synth-text "Want to compare answers or something?" \
  -o shy_boy.wav

# Encode/decode audio with the tokenizer
qwen3-tts encode -i audio.wav -o codes.pt
qwen3-tts decode -i codes.pt -o recovered.wav

# Use a local model directory
qwen3-tts speak -m ./Qwen3-TTS-12Hz-0.6B-CustomVoice -T "Hello" -s Ryan

# Use the large 1.7B models
qwen3-tts clone -m base-1.7b -T "..." -r ref.wav -rt "..."
```

## Model Shortcuts

| Shortcut | HF Model ID |
|---|---|
| `custom-0.6b` | Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice |
| `custom-1.7b` | Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice |
| `design` | Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign |
| `base-0.6b` | Qwen/Qwen3-TTS-12Hz-0.6B-Base |
| `base-1.7b` | Qwen/Qwen3-TTS-12Hz-1.7B-Base |
| `tokenizer` | Qwen/Qwen3-TTS-Tokenizer-12Hz |

You can also pass a full HuggingFace model ID or a local directory path to `--model`.

## Speakers

| Speaker | Description | Native Language |
|---|---|---|
| Vivian | Bright, slightly edgy young female voice | Chinese |
| Serena | Warm, gentle young female voice | Chinese |
| Uncle_Fu | Seasoned male voice, low mellow timbre | Chinese |
| Dylan | Youthful Beijing male voice | Chinese (Beijing) |
| Eric | Lively Chengdu male voice | Chinese (Sichuan) |
| Ryan | Dynamic male voice, strong rhythmic drive | English |
| Aiden | Sunny American male voice, clear midrange | English |
| Ono_Anna | Playful Japanese female voice | Japanese |
| Sohee | Warm Korean female voice, rich emotion | Korean |

## Supported Languages

Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian (plus `Auto` for auto-detection).
