---
name: qwen3-tts
description: |
  Qwen3-TTS CLI — generate speech from text using Qwen3-TTS models on local GPU.
  Supports voice design (describe any voice in natural language), voice clone
  (clone from reference audio), and custom voice (9 preset speakers) across
  10 languages (Chinese, English, Japanese, Korean, German, French, Russian,
  Portuguese, Spanish, Italian). Trigger with "generate speech", "text to speech",
  "tts", "voice design", "voice clone", "qwen3-tts", "speak this", "read aloud",
  "clone this voice", "design a voice", or any request to turn text into spoken audio.
license: Apache-2.0
metadata:
  tools: [qwen3-tts, bash, python3]
  hardware: [nvidia-gpu, 8GB+ VRAM]
---

# Qwen3-TTS — Local GPU Speech Generation

Use `qwen3-tts` to generate high-quality speech from text on a local NVIDIA GPU. All models run locally — no API keys needed.

## Prerequisites

The CLI must be installed in a venv at `~/Projects/qwen3-tts-cli/.venv`. Activate with:

```bash
source ~/Projects/qwen3-tts-cli/.venv/bin/activate
```

Hardware: Single RTX 5090 (32GB VRAM). All models fit comfortably.

## Quickstart — Available Commands

| Command | Use when | Model |
|---|---|---|
| `qwen3-tts speak` | User says what to say + speaker name | CustomVoice (0.6B/1.7B) |
| `qwen3-tts design` | User describes a *type* of voice | VoiceDesign (1.7B) |
| `qwen3-tts clone` | User provides reference audio to copy | Base (0.6B/1.7B) |
| `qwen3-tts design-clone` | Design once → reuse for multiple lines | VoiceDesign + Base |
| `qwen3-tts speakers` | User asks "what voices are available" | — |
| `qwen3-tts languages` | User asks "what languages supported" | — |
| `qwen3-tts encode` / `decode` | Audio ↔ codec tokens | Tokenizer (12Hz) |

## Command Reference

### speak — Preset Speaker

```bash
qwen3-tts speak -T "text" -s SPEAKER -l LANGUAGE [-i "emotion/tone instruction"] [-o out.wav]
```

Speakers: `Vivian`, `Serena`, `Uncle_Fu`, `Dylan`, `Eric` (Chinese), `Ryan`, `Aiden` (English), `Ono_Anna` (Japanese), `Sohee` (Korean).

For the 1.7B model (supports `--instruct` for tone/emotion control):

```bash
qwen3-tts speak -m custom-1.7b -T "..." -s Vivian -l Chinese -i "用愤怒的语气"
```

### design — Describe Any Voice

```bash
qwen3-tts design -T "text" -l LANGUAGE -i "voice description" [-o out.wav]
```

The `-i` (instruct) is a natural language voice description. Examples:
- `"Deep, warm, authoritative male voice, slow pace, dramatic pauses"`
- `"Cheerful young female voice, bright and energetic, fast pace"`
- `"Elderly man, raspy but kind, speaks slowly with wisdom"`
- `"30代後半の大人の日本人女性、落ち着いたアルト、気品のある音色"`

### clone — Copy a Voice from Audio

```bash
qwen3-tts clone -T "text" -r ref_audio.wav -rt "transcript" [-o out.wav]
```

`ref_audio` can be a local file path or URL. `ref_text` is the transcript of that audio.

Without transcript (x-vector only, lower quality):

```bash
qwen3-tts clone -T "text" -r ref.wav --x-vector-only
```

Cache prompt for reuse (avoids re-extracting):

```bash
qwen3-tts clone -T "line 1" -r ref.wav -rt "..." --prompt-cache voice.pt --reuse-prompt
qwen3-tts clone -T "line 2" --prompt-cache voice.pt
```

### design-clone — Design + Batch Synthesis

```bash
qwen3-tts design-clone \
  -i "Shy teenage male, tenor, nervous" \
  --ref-text "H-hey! You dropped your notebook?" \
  --synth-text "No problem! I finished those." \
  --synth-text "Want to compare answers?" \
  -o character_lines.wav
```

### Batch Mode

Pass multiple `-T` texts or use `-F file.txt` (one utterance per line):

```bash
qwen3-tts speak -T "Hello." "How are you?" "Goodbye." -s Ryan -l English -o batch.wav
# → batch_0.wav, batch_1.wav, batch_2.wav
```

## Model Shortcuts

| Shortcut | HF Model | Size |
|---|---|---|
| `custom-0.6b` | Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice | ~1.2GB |
| `custom-1.7b` | Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice | ~3.4GB |
| `design` | Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign | ~3.4GB |
| `base-0.6b` | Qwen/Qwen3-TTS-12Hz-0.6B-Base | ~1.2GB |
| `base-1.7b` | Qwen/Qwen3-TTS-12Hz-1.7B-Base | ~3.4GB |

Full HF model IDs and local paths also accepted with `-m`.

## Supported Languages

`Auto`, `Chinese`, `English`, `Japanese`, `Korean`, `German`, `French`, `Russian`, `Portuguese`, `Spanish`, `Italian`.

## Agent Decision Guide

| User says | Use command | Notes |
|---|---|---|
| "Say X in Ryan's voice" | `speak` | Pick closest speaker |
| "Make this sound like [description]" | `design` | Voice from scratch |
| "Clone this voice from [file]" | `clone` | Needs reference audio |
| "Create character voice, make them say many lines" | `design-clone` | Design once, reuse |
| "What voices do you have?" | `speakers` | List 9 preset voices |
| "Read this text file aloud" | `speak -F file.txt` | Batch from file |

## Output

Always writes to a `.wav` file (24kHz, mono, 16-bit). Default paths: `output_speak.wav`, `output_design.wav`, `output_clone.wav`, etc. Always use `-o` to specify a meaningful path after activating the venv.

## Troubleshooting

- **flash-attn warning**: Safe to ignore. Falls back to SDPA. Optional `MAX_JOBS=4 pip install flash-attn --no-build-isolation` for ~20% less VRAM.
- **SoX not found**: `brew install sox`
- **Out of VRAM**: Use 0.6B models (`custom-0.6b`, `base-0.6b`) or reduce `--max-tokens`.
