# Cantonese Text-to-Speech (Local)

A small Python tool that synthesizes Cantonese (Yue Chinese, 粵語) speech locally
using [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) — a
multilingual zero-shot TTS model that supports `yue` natively. After the first
run the model weights (~1 GB) are cached on disk and inference runs fully
offline.

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first invocation downloads the model from Hugging Face into
`~/.cache/huggingface/`.

## Usage

```bash
python tts.py "你好，歡迎使用粵語語音合成。" -o hello.wav

echo "今日天氣好好，我哋一齊去飲茶啦。" | python tts.py -o today.wav

python tts.py "唔該晒。" --speed 0.9 --num-step 16 -o thanks.wav
```

### Voice modes

OmniVoice exposes three generation modes; pick one or none:

| Mode | Flag(s) | Description |
| --- | --- | --- |
| Auto | (none) | Model picks a voice automatically. **Recommended for Cantonese.** |
| Voice design | `--instruct` | Describe the voice with attributes (e.g. `"female, low pitch"`). Trained mainly on Mandarin/English; Cantonese-specific dialect tags are not supported, so the auto mode usually sounds more natural. |
| Voice cloning | `--ref-audio` (and optional `--ref-text`) | Clone a voice from a 3-10 s reference audio clip. |

Examples:

```bash
python tts.py "Hello, this is a test." --instruct "female, british accent" -o en.wav

python tts.py "我哋一齊去飲茶啦。" \
    --ref-audio sample.wav \
    --ref-text "你好，歡迎使用粵語語音合成。" \
    -o clone.wav
```

If you omit `--ref-text`, OmniVoice auto-transcribes the reference audio with
Whisper.

### All flags

| Flag | Default | Description |
| --- | --- | --- |
| `-o`, `--output` | `output.wav` | Output WAV path |
| `--model` | `k2-fsa/OmniVoice` | Hugging Face model id |
| `--instruct` | – | Voice-design prompt (mutually exclusive with `--ref-audio`) |
| `--ref-audio` | – | Reference clip for voice cloning |
| `--ref-text` | – | Optional transcription of `--ref-audio` |
| `--speed` | `1.0` | Speaking-rate multiplier |
| `--num-step` | `32` | Diffusion steps (16 = faster, 32 = higher quality) |
| `--device` | `auto` | `auto` / `cpu` / `cuda` / `mps` |

On Apple Silicon the script automatically uses the MPS backend; on machines
with NVIDIA GPUs it uses CUDA; otherwise it falls back to CPU.

## Playing the result

```bash
afplay output.wav        # macOS
aplay output.wav         # Linux
```

## Notes

- Output is always 24 kHz mono float WAV.
- Cantonese is one of 600+ languages OmniVoice supports. The model identifies
  the language from the script (use Traditional Chinese with Cantonese-specific
  characters such as 嘅 / 啲 / 唔 / 哋 / 喺 for the most natural results).
- Mixed English/numbers in the input are read out, but for best clarity convert
  digits to characters first (e.g. `123` → `一百二十三`).
- For long passages, split the text into sentences before calling `tts.py` —
  diffusion TTS works best on short chunks (≲ 30 s of speech each).
- Model license: Apache-2.0. See the upstream
  [model card](https://huggingface.co/k2-fsa/OmniVoice) for the disclaimer
  about voice cloning ethics.
