#!/usr/bin/env python3
"""
qwen3-tts — CLI for Qwen3-TTS: voice clone, voice design, custom voice generation.

Usage:
  qwen3-tts speak    -T "text" [-s speaker] [-l lang] [-i instruct] [-o out.wav]
  qwen3-tts design   -T "text" [-l lang] -i instruct [-o out.wav]
  qwen3-tts clone    -T "text" -r ref.wav [-rt "ref text"] [-o out.wav]
  qwen3-tts design-clone  -i instruct --ref-text "..." --synth-text "..." [--synth-text "..." ...]
  qwen3-tts speakers
  qwen3-tts languages
  qwen3-tts encode   -i audio.wav [-o codes.pt]
  qwen3-tts decode   -i codes.pt [-o out.wav]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Model name shortcuts
# ---------------------------------------------------------------------------

MODEL_MAP = {
    "custom-0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "custom-1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "design":     "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "base-0.6b":  "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "base-1.7b":  "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "tokenizer":  "Qwen/Qwen3-TTS-Tokenizer-12Hz",
}

# Default model per command
DEFAULT_SPEAK_MODEL = "custom-0.6b"
DEFAULT_DESIGN_MODEL = "design"
DEFAULT_CLONE_MODEL = "base-0.6b"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_model(name: str) -> str:
    """Resolve a shortcut name or return the full HF path / local dir as-is."""
    return MODEL_MAP.get(name, name)


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m"


def _load_model(model_name: str, device: str, dtype_str: str, attn: str):
    """Load a Qwen3TTSModel (lazy import so --help is fast)."""
    import torch
    from qwen_tts import Qwen3TTSModel  # type: ignore[import-untyped]

    resolved = _resolve_model(model_name)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_str]

    print(f"Loading model: {_bold(resolved)}", file=sys.stderr)
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        resolved,
        device_map=device,
        dtype=dtype,
        attn_implementation=attn,
    )
    elapsed = time.time() - t0
    print(f"Model loaded in {elapsed:.1f}s on {device}", file=sys.stderr)
    return model


def _load_tokenizer(device: str):
    """Load Qwen3TTSTokenizer."""
    from qwen_tts import Qwen3TTSTokenizer  # type: ignore[import-untyped]

    resolved = _resolve_model("tokenizer")
    print(f"Loading tokenizer: {_bold(resolved)}", file=sys.stderr)
    return Qwen3TTSTokenizer.from_pretrained(resolved, device_map=device)


def _write_wav(wav, sr: int, path: str):
    """Write a single waveform to a WAV file."""
    import soundfile as sf  # type: ignore[import-untyped]
    sf.write(path, wav, sr)
    print(f"Saved  → {_green(path)}", file=sys.stderr)


def _write_wavs(wavs, sr: int, path: str):
    """Write one or multiple waveforms. If multiple, inserts a number before extension."""
    import soundfile as sf  # type: ignore[import-untyped]
    if len(wavs) == 1:
        sf.write(path, wavs[0], sr)
        print(f"Saved  → {_green(path)}", file=sys.stderr)
    else:
        stem = Path(path).stem
        suffix = Path(path).suffix
        for i, w in enumerate(wavs):
            out = f"{stem}_{i}{suffix}"
            sf.write(out, w, sr)
            print(f"Saved  → {_green(out)}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Subcommand: speak (CustomVoice)
# ---------------------------------------------------------------------------

def cmd_speak(args: argparse.Namespace) -> None:
    """Generate speech using a preset speaker (CustomVoice model)."""
    model = _load_model(args.model, args.device, args.dtype, args.attn)

    language = args.language or "Auto"
    instruct = args.instruct or ""

    # Read text from file or argument
    text = _read_text(args.text, args.file)
    lang = _broadcast_language(language, text)
    instructs = instruct if len(text) <= 1 else [instruct] * len(text)

    t0 = time.time()
    if torch_available():
        import torch
        if model.device.type == "cuda":
            torch.cuda.synchronize()

    wavs, sr = model.generate_custom_voice(
        text=text,
        language=lang,
        speaker=args.speaker,
        instruct=instructs,
        max_new_tokens=args.max_tokens,
    )

    if torch_available() and model.device.type == "cuda":
        import torch
        torch.cuda.synchronize()

    elapsed = time.time() - t0
    print(f"Generated {len(text)} line(s) in {elapsed:.1f}s ({len(wavs[0])/sr:.1f}s audio)", file=sys.stderr)

    _write_wavs(wavs, sr, args.output)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Subcommand: design (VoiceDesign)
# ---------------------------------------------------------------------------

def cmd_design(args: argparse.Namespace) -> None:
    """Generate speech from a text description of the voice (VoiceDesign model)."""
    if not args.instruct:
        print(f"{_yellow('Error:')} --instruct is required for voice design. Describe the voice you want.", file=sys.stderr)
        sys.exit(1)

    model = _load_model(args.model, args.device, args.dtype, args.attn)

    language = args.language or "Auto"
    text = _read_text(args.text, args.file)
    lang = _broadcast_language(language, text)
    instructs = args.instruct if len(text) <= 1 else [args.instruct] * len(text)

    t0 = time.time()
    if torch_available():
        import torch
        if model.device.type == "cuda":
            torch.cuda.synchronize()

    wavs, sr = model.generate_voice_design(
        text=text,
        language=lang,
        instruct=instructs,
        max_new_tokens=args.max_tokens,
    )

    if torch_available() and model.device.type == "cuda":
        import torch
        torch.cuda.synchronize()

    elapsed = time.time() - t0
    print(f"Generated {len(text)} line(s) in {elapsed:.1f}s ({len(wavs[0])/sr:.1f}s audio)", file=sys.stderr)

    _write_wavs(wavs, sr, args.output)


# ---------------------------------------------------------------------------
# Subcommand: clone (Voice Clone)
# ---------------------------------------------------------------------------

def cmd_clone(args: argparse.Namespace) -> None:
    """Clone a voice from reference audio and synthesize new speech."""
    model = _load_model(args.model, args.device, args.dtype, args.attn)

    language = args.language or "Auto"
    text = _read_text(args.text, args.file)
    lang = _broadcast_language(language, text)
    ref_audio = args.ref_audio

    # Allow x_vector_only mode (no ref_text needed)
    x_vector_only = args.x_vector_only
    ref_text = "" if x_vector_only else (args.ref_text or "")

    if not x_vector_only and not args.ref_text:
        print(f"{_yellow('Warning:')} No --ref-text provided. Use --x-vector-only to clone without transcript, or provide --ref-text.", file=sys.stderr)
        # Continue anyway — the underlying API may handle it

    t0 = time.time()
    if torch_available():
        import torch
        if model.device.type == "cuda":
            torch.cuda.synchronize()

    if args.reuse_prompt and args.prompt_cache:
        # Build prompt once, save for reuse
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only,
        )
        _save_prompt(prompt_items, args.prompt_cache)
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=lang,
            voice_clone_prompt=prompt_items,
            max_new_tokens=args.max_tokens,
        )
    elif args.prompt_cache:
        # Load cached prompt
        prompt_items = _load_prompt(args.prompt_cache)
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=lang,
            voice_clone_prompt=prompt_items,
            max_new_tokens=args.max_tokens,
        )
    else:
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=lang,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only,
            max_new_tokens=args.max_tokens,
        )

    if torch_available() and model.device.type == "cuda":
        import torch
        torch.cuda.synchronize()

    elapsed = time.time() - t0
    print(f"Generated {len(text)} line(s) in {elapsed:.1f}s ({len(wavs[0])/sr:.1f}s audio)", file=sys.stderr)

    _write_wavs(wavs, sr, args.output)


def _save_prompt(prompt_items, path: str):
    import torch
    torch.save(prompt_items, path)
    print(f"Cached voice prompt → {_green(path)}", file=sys.stderr)


def _load_prompt(path: str):
    import torch
    print(f"Loading cached voice prompt from {path}", file=sys.stderr)
    return torch.load(path, weights_only=False)


# ---------------------------------------------------------------------------
# Subcommand: design-clone (Voice Design → Clone pipeline)
# ---------------------------------------------------------------------------

def cmd_design_clone(args: argparse.Namespace) -> None:
    """Design a voice, then clone it for batch synthesis (2-step pipeline)."""
    if not args.instruct:
        print(f"{_yellow('Error:')} --instruct is required for voice design.", file=sys.stderr)
        sys.exit(1)
    if not args.ref_text:
        print(f"{_yellow('Error:')} --ref-text is required (the text to speak in the designed voice as reference).", file=sys.stderr)
        sys.exit(1)
    if not args.synth_text:
        text = _read_text(args.text, args.file)
        if not text:
            print(f"{_yellow('Error:')} Provide --synth-text or --text/--file for synthesis.", file=sys.stderr)
            sys.exit(1)
        synth_texts = text
    else:
        synth_texts = args.synth_text

    # Step 1: Voice Design
    design_model = _load_model(args.design_model, args.device, args.dtype, args.attn)
    ref_lang = args.ref_language or "Auto"

    print(f"Step 1/2: Designing reference voice...", file=sys.stderr)
    t0 = time.time()
    ref_wavs, sr = design_model.generate_voice_design(
        text=args.ref_text,
        language=ref_lang,
        instruct=args.instruct,
        max_new_tokens=args.max_tokens,
    )
    print(f"  Reference audio generated in {time.time()-t0:.1f}s", file=sys.stderr)

    if args.save_ref:
        _write_wav(ref_wavs[0], sr, args.save_ref)

    # Step 2: Voice Clone
    clone_model = _load_model(args.clone_model, args.device, args.dtype, args.attn)

    print(f"Step 2/2: Building clone prompt & synthesizing...", file=sys.stderr)
    t0 = time.time()
    voice_clone_prompt = clone_model.create_voice_clone_prompt(
        ref_audio=(ref_wavs[0], sr),
        ref_text=args.ref_text,
    )

    synth_lang = args.language or "Auto"
    synth_lang = _broadcast_language(synth_lang, synth_texts)
    wavs, sr_out = clone_model.generate_voice_clone(
        text=synth_texts,
        language=synth_lang,
        voice_clone_prompt=voice_clone_prompt,
        max_new_tokens=args.max_tokens,
    )
    print(f"  Synthesis complete in {time.time()-t0:.1f}s", file=sys.stderr)

    _write_wavs(wavs, sr_out, args.output)


# ---------------------------------------------------------------------------
# Subcommand: speakers
# ---------------------------------------------------------------------------

SPEAKER_INFO = [
    ("Vivian",    "Bright, slightly edgy young female voice",              "Chinese"),
    ("Serena",    "Warm, gentle young female voice",                       "Chinese"),
    ("Uncle_Fu",  "Seasoned male voice with a low, mellow timbre",         "Chinese"),
    ("Dylan",     "Youthful Beijing male voice, clear natural timbre",     "Chinese (Beijing)"),
    ("Eric",      "Lively Chengdu male voice, slightly husky brightness",  "Chinese (Sichuan)"),
    ("Ryan",      "Dynamic male voice with strong rhythmic drive",         "English"),
    ("Aiden",     "Sunny American male voice with a clear midrange",       "English"),
    ("Ono_Anna",  "Playful Japanese female voice, light nimble timbre",    "Japanese"),
    ("Sohee",     "Warm Korean female voice with rich emotion",            "Korean"),
]

LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]


def cmd_speakers(args: argparse.Namespace) -> None:
    """List available speakers for CustomVoice models."""
    print(f"\n{_bold('Available Speakers')} (for CustomVoice models)\n")
    print(f"  {'Speaker':<12} {'Native':<22} Description")
    print(f"  {'─'*12} {'─'*22} {'─'*40}")
    for name, desc, native in SPEAKER_INFO:
        print(f"  {_green(name):<12} {native:<22} {desc}")


def cmd_languages(args: argparse.Namespace) -> None:
    """List supported languages."""
    print(f"\n{_bold('Supported Languages')}\n")
    for lang in LANGUAGES:
        tag = " (auto-detect)" if lang == "Auto" else ""
        print(f"  • {_green(lang)}{tag}")


# ---------------------------------------------------------------------------
# Subcommand: encode (Tokenizer)
# ---------------------------------------------------------------------------

def cmd_encode(args: argparse.Namespace) -> None:
    """Encode audio to discrete codes using the 12Hz tokenizer."""
    import torch
    tokenizer = _load_tokenizer(args.device)

    audio = args.input
    enc = tokenizer.encode(audio)

    out_path = args.output
    torch.save({"audio_codes": enc.audio_codes, "sr": enc.sr if hasattr(enc, 'sr') else None}, out_path)
    print(f"Encoded → {_green(out_path)}", file=sys.stderr)

    # Print info
    if isinstance(enc.audio_codes, list):
        for i, codes in enumerate(enc.audio_codes):
            print(f"  sample {i}: codes shape {list(codes.shape)}", file=sys.stderr)
    else:
        print(f"  codes shape: {list(enc.audio_codes.shape)}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Subcommand: decode (Tokenizer)
# ---------------------------------------------------------------------------

def cmd_decode(args: argparse.Namespace) -> None:
    """Decode discrete codes back to audio using the 12Hz tokenizer."""
    import torch
    tokenizer = _load_tokenizer(args.device)

    data = torch.load(args.input, weights_only=False)
    if isinstance(data, dict) and "audio_codes" in data:
        wavs, sr = tokenizer.decode(data)
    else:
        # Assume the saved object is the encode result directly
        wavs, sr = tokenizer.decode(data)

    _write_wavs(wavs, sr, args.output)


# ---------------------------------------------------------------------------
# Shared argument helpers
# ---------------------------------------------------------------------------

def _read_text(text_arg, file_arg) -> list[str]:
    """Return a list of text lines from --text (multiple allowed) or --file."""
    if file_arg:
        text = Path(file_arg).read_text().strip()
        return [ln.strip() for ln in text.splitlines() if ln.strip()]
    if text_arg:
        return text_arg  # already a list from nargs='*' or single string
    return []


def _broadcast_language(language: str, texts: list[str]) -> str | list[str]:
    """If there are multiple texts, broadcast language to a list of same length."""
    if len(texts) <= 1:
        return language
    return [language] * len(texts)


def _add_model_args(parser: argparse.ArgumentParser, default_model: str):
    parser.add_argument(
        "--model", "-m", default=default_model,
        help=f"Model shortcut or full HF ID. Shortcuts: {', '.join(MODEL_MAP)} (default: {default_model})"
    )
    parser.add_argument(
        "--device", "-d", default="cuda:0",
        help="Device to run on (default: cuda:0)"
    )
    parser.add_argument(
        "--dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
        help="Model precision (default: bf16)"
    )
    parser.add_argument(
        "--attn", default="sdpa",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention implementation (default: sdpa). Use flash_attention_2 if installed for lower VRAM usage."
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048,
        help="Max new tokens for generation (default: 2048)"
    )


def _add_text_args(parser: argparse.ArgumentParser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", "-T", nargs="*", default=[], help="Text to synthesize (can pass multiple)")
    group.add_argument("--file", "-F", help="File containing text (one utterance per line)")


def _add_output_arg(parser: argparse.ArgumentParser, default: str):
    parser.add_argument("--output", "-o", default=default, help=f"Output WAV path (default: {default})")


def _add_language_arg(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--language", "-l", default=None,
        help="Language code (e.g. Chinese, English, Japanese, Auto). Default: Auto"
    )


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwen3-tts",
        description="CLI for Qwen3-TTS: voice clone, voice design, and custom voice generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", title="commands")

    # ---- speak ----
    p_speak = sub.add_parser("speak", help="Generate speech with a preset speaker (CustomVoice)")
    _add_model_args(p_speak, DEFAULT_SPEAK_MODEL)
    _add_text_args(p_speak)
    _add_language_arg(p_speak)
    _add_output_arg(p_speak, "output_speak.wav")
    p_speak.add_argument("--speaker", "-s", default="Vivian", help="Speaker name (default: Vivian)")
    p_speak.add_argument("--instruct", "-i", default=None, help="Optional instruction (e.g. 'angry tone')")

    # ---- design ----
    p_design = sub.add_parser("design", help="Generate speech from a voice description (VoiceDesign)")
    _add_model_args(p_design, DEFAULT_DESIGN_MODEL)
    _add_text_args(p_design)
    _add_language_arg(p_design)
    _add_output_arg(p_design, "output_design.wav")
    p_design.add_argument("--instruct", "-i", required=True, help="Voice description (e.g. 'deep, slow, authoritative male voice')")

    # ---- clone ----
    p_clone = sub.add_parser("clone", help="Clone a voice from reference audio (Base model)")
    _add_model_args(p_clone, DEFAULT_CLONE_MODEL)
    _add_text_args(p_clone)
    _add_language_arg(p_clone)
    _add_output_arg(p_clone, "output_clone.wav")
    p_clone.add_argument("--ref-audio", "-r", required=True, help="Reference audio file path or URL")
    p_clone.add_argument("--ref-text", "-rt", default=None, help="Transcript of the reference audio")
    p_clone.add_argument("--x-vector-only", action="store_true", help="Use x-vector only mode (no transcript needed)")
    p_clone.add_argument("--prompt-cache", "-pc", default=None, help="Path to save/load cached voice clone prompt (.pt)")
    p_clone.add_argument("--reuse-prompt", action="store_true", help="Save the voice clone prompt to --prompt-cache for reuse")

    # ---- design-clone ----
    p_dc = sub.add_parser("design-clone", help="Design a voice then reuse it for synthesis (2-step pipeline)")
    _add_model_args(p_dc, DEFAULT_DESIGN_MODEL)
    p_dc.add_argument("--design-model", default=DEFAULT_DESIGN_MODEL, help="Model for voice design step")
    p_dc.add_argument("--clone-model", default=DEFAULT_CLONE_MODEL, help="Model for voice clone step")
    _add_language_arg(p_dc)
    _add_output_arg(p_dc, "output_design_clone.wav")
    p_dc.add_argument("--instruct", "-i", required=True, help="Voice description for design step")
    p_dc.add_argument("--ref-text", required=True, help="Text for the reference audio (spoken in the designed voice)")
    p_dc.add_argument("--ref-language", default=None, help="Language for reference text (default: Auto)")
    p_dc.add_argument("--synth-text", action="append", default=[], help="Text(s) to synthesize with the cloned voice (repeatable)")
    p_dc.add_argument("--save-ref", default=None, help="Save the designed reference audio (.wav)")
    # Allow --text/--file as alternative to --synth-text
    group = p_dc.add_mutually_exclusive_group(required=False)
    group.add_argument("--text", "-T", nargs="*", default=[], help="Text to synthesize (alternative to --synth-text)")
    group.add_argument("--file", "-F", help="File containing text (one utterance per line)")

    # ---- speakers ----
    sub.add_parser("speakers", help="List available speakers for CustomVoice models")

    # ---- languages ----
    sub.add_parser("languages", help="List supported languages")

    # ---- encode ----
    p_enc = sub.add_parser("encode", help="Encode audio to discrete codes (tokenizer)")
    p_enc.add_argument("--input", "-i", required=True, help="Audio file path or URL")
    p_enc.add_argument("--output", "-o", default="encoded_codes.pt", help="Output path for encoded codes")
    p_enc.add_argument("--device", "-d", default="cuda:0", help="Device (default: cuda:0)")

    # ---- decode ----
    p_dec = sub.add_parser("decode", help="Decode discrete codes back to audio (tokenizer)")
    p_dec.add_argument("--input", "-i", required=True, help="Encoded codes file (.pt)")
    p_dec.add_argument("--output", "-o", default="decoded_audio.wav", help="Output WAV path")
    p_dec.add_argument("--device", "-d", default="cuda:0", help="Device (default: cuda:0)")

    return parser


# Map command names to handler functions
COMMAND_HANDLERS = {
    "speak":        cmd_speak,
    "design":       cmd_design,
    "clone":        cmd_clone,
    "design-clone": cmd_design_clone,
    "speakers":     cmd_speakers,
    "languages":    cmd_languages,
    "encode":       cmd_encode,
    "decode":       cmd_decode,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handler = COMMAND_HANDLERS.get(args.command)
    if handler:
        handler(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
