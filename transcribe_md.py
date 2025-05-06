"""
CLI that converts an audio file to a Markdown transcript.
  • default  : fine-tuned Whisper-ATC + entities
  • --compare: additionally include baseline Whisper-tiny transcript
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
from typing import List, Tuple

import torch, torchaudio
from transformers import pipeline

from nlp import atc_ner

# ───────────────────────── config ──────────────────────────
FINETUNED_DIR = Path("model/whisper-atc-6")
BASE_MODEL_ID = "openai/whisper-tiny"
SR = 16_000
CHUNK_S = 30
STRIDE_S = (4, 2)
OUTPUT_DIR = Path("output")  # ← new directory
OUTPUT_DIR.mkdir(exist_ok=True)  # ensure it exists


# ─────────────────────── helper fn ─────────────────────────
def _fix_whisper(p):
    """Remove forced decoder ids so `generate()` stops complaining."""
    p.model.config.forced_decoder_ids = None
    p.model.generation_config.forced_decoder_ids = None
    return p


def generate_markdown(audio_path: Path, compare: bool = False) -> str:
    """
    Return the Markdown transcript for one audio file.
    Keeps the original *compare* logic but does **not** touch the disk.
    """
    pipe_ft, pipe_b = _load_pipelines()
    wav = _load_wave(audio_path)

    txt_ft = _transcribe(pipe_ft, wav)
    ents   = _entities(txt_ft)

    txt_b = _transcribe(pipe_b, wav) if compare else None

    md  = [f"# Transcript for `{audio_path.name}`\n"]
    md += ["## Fine-Tuned Whisper-ATC\n", txt_ft, ""]
    md += ["### Named Entities\n"]
    if ents:
        md += ["| text | label |", "|------|-------|"]
        md += [f"| {t} | {l} |" for t, l in ents]
    else:
        md += ["_None detected_"]
    if txt_b is not None:
        md += ["", "## Baseline Whisper-tiny", "", txt_b]
    return "\n".join(md)


# ─────────────────────── pipelines ─────────────────────────
def _load_pipelines():
    if not FINETUNED_DIR.exists():
        sys.exit(f"✗ fine-tuned model not found at {FINETUNED_DIR}")

    common = dict(
        task="automatic-speech-recognition",
        chunk_length_s=CHUNK_S,
        stride_length_s=STRIDE_S,
        return_timestamps="none",
        generate_kwargs={"return_timestamps": "none"},
    )

    print("🔗 loading fine-tuned pipeline …")
    pipe_ft = _fix_whisper(
        pipeline(
            **common,
            model=str(FINETUNED_DIR),
            tokenizer=str(FINETUNED_DIR),
            feature_extractor=str(FINETUNED_DIR),
            device=0 if torch.cuda.is_available() else -1,
        )
    )

    print("🔗 loading baseline Whisper-tiny …")
    pipe_b = _fix_whisper(
        pipeline(
            **common,
            model=BASE_MODEL_ID,
            tokenizer=BASE_MODEL_ID,
            feature_extractor=BASE_MODEL_ID,
            device=-1,
        )
    )
    return pipe_ft, pipe_b


# ───────────────────────── audio I/O ───────────────────────
def _ffmpeg_decode(path: str) -> torch.Tensor:
    cmd = ["ffmpeg", "-nostdin", "-i", path, "-f", "s16le",
           "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(SR), "-"]
    data = subprocess.run(cmd, capture_output=True, check=True).stdout
    return torch.frombuffer(data, dtype=torch.int16).float() / 32768.0


def _load_wave(path: Path) -> torch.Tensor:
    try:
        wav, sr = torchaudio.load(path)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        return wav.mean(0) if wav.ndim == 2 else wav
    except Exception:
        return _ffmpeg_decode(str(path))


# ───────────────────────── helpers ─────────────────────────
def _transcribe(pipe, wav: torch.Tensor) -> str:
    out = pipe({"array": wav.numpy(), "sampling_rate": SR})
    return out["text"].upper()


def _entities(txt: str) -> List[Tuple[str, str]]:
    return atc_ner.annotate(txt)


# ────────────────────────── main ───────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path, help="input wav / any audio file")
    ap.add_argument("--compare", action="store_true",
                    help="include baseline Whisper-tiny transcript")
    args = ap.parse_args()

    if not args.audio.exists():
        sys.exit(f"✗ file not found: {args.audio}")

    pipe_ft, pipe_b = _load_pipelines()
    wav = _load_wave(args.audio)

    print("📝 transcribing (fine-tuned)…")
    txt_ft = _transcribe(pipe_ft, wav)
    ents = _entities(txt_ft)

    txt_b = None
    if args.compare:
        print("📝 transcribing (baseline)…")
        txt_b = _transcribe(pipe_b, wav)

    md_path = OUTPUT_DIR / f"{args.audio.stem}.md"  # ← save to ./output/
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Transcript for `{args.audio.name}`\n\n")
        f.write("## Fine-Tuned Whisper-ATC\n\n")
        f.write(txt_ft + "\n\n")

        f.write("### Named Entities\n\n")
        if ents:
            f.write("| text | label |\n|------|-------|\n")
            for text, label in ents:
                f.write(f"| {text} | {label} |\n")
        else:
            f.write("_None detected_\n")

        if txt_b is not None:
            f.write("\n\n## Baseline Whisper-tiny\n\n")
            f.write(txt_b + "\n")

    print("✅ saved", md_path)


if __name__ == "__main__":
    main()
