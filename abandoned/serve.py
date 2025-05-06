"""
serve.py ─ FastAPI back-end for ATC-Whisper

Endpoints
    POST /transcribe   → fine-tuned transcript + entities
    POST /compare      → same audio, fine-tuned vs. baseline + entities
    GET  /ui/*         → static front-end
    GET  /health       → {"status":"ok"}
"""

from __future__ import annotations
import os, subprocess, tempfile
from pathlib import Path
from typing import List, Tuple

import torch, torchaudio
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from nlp import atc_ner

# ─────────────────── configuration ────────────────────
FINETUNED_DIR  = Path("model/whisper-atc")
BASE_MODEL_ID  = "openai/whisper-tiny"          # HF hub id
UI_DIR         = Path("ui/frontend")
SR             = 16_000
DEVICE_FINE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_BASE    = torch.device("cpu")            # keep baseline on CPU to save vRAM

# ─────────────────── model load  ──────────────────────
if not FINETUNED_DIR.exists():
    raise RuntimeError(f"Fine-tuned model not found at {FINETUNED_DIR}")

print("🔗  loading fine-tuned checkpoint …")
proc_ft  = WhisperProcessor.from_pretrained(FINETUNED_DIR)
model_ft = WhisperForConditionalGeneration.from_pretrained(FINETUNED_DIR).to(DEVICE_FINE)
model_ft.generation_config.forced_decoder_ids = None
model_ft.eval()

print("🔗  loading baseline Whisper-tiny …")
proc_base  = WhisperProcessor.from_pretrained(BASE_MODEL_ID)
model_base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID).to(DEVICE_BASE)
model_base.generation_config.forced_decoder_ids = None
model_base.eval()

PROMPT_IDS_FT   = proc_ft.get_decoder_prompt_ids(language="en", task="transcribe")
PROMPT_IDS_BASE = proc_base.get_decoder_prompt_ids(language="en", task="transcribe")

# ─────────────────── helpers ──────────────────────────
def _save_upload(upload: UploadFile) -> str:
    fd, path = tempfile.mkstemp(suffix=Path(upload.filename).suffix or ".dat")
    with os.fdopen(fd, "wb") as f:
        f.write(upload.file.read())
    return path


def _ffmpeg_decode(path: str) -> torch.Tensor:
    cmd = [
        "ffmpeg", "-nostdin", "-i", path, "-f", "s16le",
        "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(SR), "-"
    ]
    data = subprocess.run(cmd, capture_output=True, check=True).stdout
    return torch.frombuffer(data, dtype=torch.int16).float() / 32768.0


def _load_audio(upload: UploadFile) -> torch.Tensor:
    tmp = _save_upload(upload)
    try:
        try:
            wav, sr = torchaudio.load(tmp)
            if sr != SR:
                wav = torchaudio.functional.resample(wav, sr, SR)
            wav = wav.mean(0) if wav.ndim == 2 else wav
            return wav
        except Exception:
            return _ffmpeg_decode(tmp)
    finally:
        os.remove(tmp)


@torch.inference_mode()
def _transcribe(
        wav: torch.Tensor,
        proc: WhisperProcessor,
        model: WhisperForConditionalGeneration,
        prompt_ids: List[int],
        device: torch.device,
) -> str:
    inputs = proc(wav.numpy(), sampling_rate=SR, return_tensors="pt").to(device)
    ids = model.generate(
        **inputs,
        decoder_input_ids=torch.tensor([prompt_ids], device=device),
        max_length=225,
    )
    return proc.batch_decode(ids, skip_special_tokens=True)[0].upper()


def _entities(txt: str) -> List[Tuple[str, str]]:
    return atc_ner.annotate(txt)

# ─────────────────── FastAPI app ──────────────────────
app = FastAPI(title="ATC-Whisper")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.get("/", include_in_schema=False)
def root(): return RedirectResponse("/ui/")

@app.get("/health")
def health(): return {"status": "ok"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    wav   = _load_audio(file)
    text  = _transcribe(wav, proc_ft, model_ft, PROMPT_IDS_FT, DEVICE_FINE)
    ents  = _entities(text)
    return JSONResponse({"text": text, "entities": ents})


@app.post("/compare")
async def compare(file: UploadFile = File(...)):
    wav   = _load_audio(file)
    txt_ft  = _transcribe(wav, proc_ft,  model_ft,  PROMPT_IDS_FT,   DEVICE_FINE)
    txt_b   = _transcribe(wav, proc_base, model_base, PROMPT_IDS_BASE, DEVICE_BASE)
    ents    = _entities(txt_ft)          # entities only from fine-tuned transcript
    return JSONResponse({
        "custom_transcript":   txt_ft,
        "baseline_transcript": txt_b,
        "entities":            ents,
    })

# ───────────── static single-page app ─────────────────
if not UI_DIR.joinpath("index.html").exists():
    raise RuntimeError(f"Front-end not found at {UI_DIR}")
app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="frontend")