"""Local Cantonese (Yue Chinese) text-to-speech using k2-fsa/OmniVoice.

OmniVoice is a multilingual zero-shot TTS model (600+ languages) that
supports Cantonese natively. The pretrained weights (~1 GB) are downloaded
once from Hugging Face and cached locally; subsequent runs are fully offline.

Three generation modes are supported:
  * auto     – the model picks a voice automatically
  * design   – describe the voice with attributes (gender, accent, dialect…)
  * clone    – clone a voice from a short reference audio clip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from omnivoice import OmniVoice

DEFAULT_MODEL_ID = "k2-fsa/OmniVoice"
DEFAULT_OUTPUT = "output.wav"
DEFAULT_SAMPLE_RATE = 24_000  # OmniVoice always outputs at 24 kHz.


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    # fp16 on accelerated devices; fp32 on CPU for kernel compatibility.
    if device.startswith("cuda") or device == "mps":
        return torch.float16
    return torch.float32


def load_model(model_id: str, device: str, dtype: torch.dtype) -> OmniVoice:
    print(f"Loading model '{model_id}' on {device} ({dtype})...", file=sys.stderr)
    return OmniVoice.from_pretrained(model_id, device_map=device, dtype=dtype)


def synthesize(
    model: OmniVoice,
    text: str,
    *,
    instruct: str | None = None,
    ref_audio: str | None = None,
    ref_text: str | None = None,
    speed: float = 1.0,
    num_step: int = 32,
) -> np.ndarray:
    """Run TTS inference and return a 1-D waveform at 24 kHz."""
    kwargs: dict = {"text": text, "speed": speed, "num_step": num_step}
    if ref_audio:
        kwargs["ref_audio"] = ref_audio
        if ref_text:
            kwargs["ref_text"] = ref_text
    elif instruct:
        kwargs["instruct"] = instruct

    audio = model.generate(**kwargs)
    # `generate` returns a list of np.ndarray; one entry per input text.
    waveform = audio[0] if isinstance(audio, list) else audio
    return np.asarray(waveform, dtype=np.float32)


def save_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform, sample_rate)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local Cantonese TTS using k2-fsa/OmniVoice.",
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Cantonese text to synthesize. If omitted, reads from stdin.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output WAV file path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model id (default: {DEFAULT_MODEL_ID}).",
    )

    mode = parser.add_argument_group("voice mode (pick at most one)")
    mode.add_argument(
        "--instruct",
        help='Voice-design prompt, e.g. "female, low pitch, 粵語". '
        "Leave empty for auto-voice mode.",
    )
    mode.add_argument(
        "--ref-audio",
        help="Path to a 3-10s reference audio clip for voice cloning.",
    )
    mode.add_argument(
        "--ref-text",
        help="Optional transcription of the reference audio "
        "(auto-transcribed via Whisper if omitted).",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speaking-rate multiplier; >1 faster, <1 slower (default: 1.0).",
    )
    parser.add_argument(
        "--num-step",
        type=int,
        default=32,
        help="Diffusion steps (16 is faster, 32 is higher quality). Default: 32.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "cuda:0", "mps"],
        default="auto",
        help="Compute device (default: auto).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    text = args.text if args.text is not None else sys.stdin.read()
    text = text.strip()
    if not text:
        print("Error: no input text provided.", file=sys.stderr)
        return 1

    if args.ref_audio and args.instruct:
        print(
            "Error: --instruct and --ref-audio are mutually exclusive.",
            file=sys.stderr,
        )
        return 2

    device = pick_device() if args.device == "auto" else args.device
    if device == "cuda":
        device = "cuda:0"
    print(f"Using device: {device}")
    dtype = pick_dtype(device)

    model = load_model(args.model, device, dtype)

    mode = (
        "clone"
        if args.ref_audio
        else "design"
        if args.instruct
        else "auto"
    )
    print(
        f"Synthesizing {len(text)} characters (mode={mode}, "
        f"speed={args.speed}, steps={args.num_step})...",
        file=sys.stderr,
    )

    waveform = synthesize(
        model,
        text,
        instruct=args.instruct,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        speed=args.speed,
        num_step=args.num_step,
    )

    out_path = Path(args.output)
    save_wav(out_path, waveform, DEFAULT_SAMPLE_RATE)
    duration = len(waveform) / DEFAULT_SAMPLE_RATE
    print(
        f"Wrote {out_path} ({duration:.2f}s @ {DEFAULT_SAMPLE_RATE} Hz)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
