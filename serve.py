"""
FastAPI server providing:

  • POST /transcribe   – fine-tuned model
  • POST /compare      – fine-tuned vs baseline model
  • GET  /ui/*         – static web UI
"""
from __future__ import annotations
import difflib
import tempfile
from pathlib import Path

import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import whisper

from nlp import atc_ner

# ---------------------------------------------------------------------- #
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FINE_TUNED_DIR = Path("model/whisper-atc")
BASELINE_MODEL = "openai/whisper-tiny"

fine_tuned = whisper.load_model(FINE_TUNED_DIR, device=DEVICE)
baseline = whisper.load_model(BASELINE_MODEL, device=DEVICE)

SAMPLE_RATE = 16_000
app = FastAPI(title="ATC-Whisper")


def _decode(wave: torch.Tensor, model) -> str:
    wave = whisper.pad_or_trim(wave.squeeze(0))
    mel = whisper.log_mel_spectrogram(wave).to(model.device)
    opts = whisper.DecodingOptions(language="en", fp16=model.device.type == "cuda")
    return whisper.decode(model, mel, opts).text


def _load_audio(file: UploadFile) -> torch.Tensor:
    with tempfile.NamedTemporaryFile(suffix=file.filename) as tmp:
        tmp.write(file.file.read());
        tmp.flush()
        wave, sr = torchaudio.load(tmp.name)
    if sr != SAMPLE_RATE:
        wave = torchaudio.functional.resample(wave, sr, SAMPLE_RATE)
    if wave.shape[0] > 1:
        wave = wave.mean(0, keepdim=True)
    return wave


# ---------------------------------------------------------------------- #
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    wave = _load_audio(file)
    txt = _decode(wave, fine_tuned).upper()
    ents = atc_ner.annotate(txt)
    return JSONResponse({"text": txt, "entities": ents})


@app.post("/compare")
async def compare(file: UploadFile = File(...)):
    wave = _load_audio(file)
    tuned_txt = _decode(wave, fine_tuned).upper()
    base_txt = _decode(wave, baseline).upper()
    diff = "\n".join(
        difflib.unified_diff(
            base_txt.split(), tuned_txt.split(),
            fromfile="baseline", tofile="fine-tuned", lineterm=""
        )
    )
    ents = atc_ner.annotate(tuned_txt)
    return JSONResponse(
        {
            "custom_transcript": tuned_txt,
            "baseline_transcript": base_txt,
            "diff": diff,
            "entities": ents,
        }
    )


# ---------------------------------------------------------------------- #
app.mount(
    "/ui",
    StaticFiles(directory="ui/frontend", html=True),
    name="frontend",
)
