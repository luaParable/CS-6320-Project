"""
CPU-only Faster-Whisper transcription with automatic compute-type fallback.
"""
from __future__ import annotations
from pathlib import Path
import ctranslate2  # ← query supported types
import datasets
from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/chunks_dataset"; SRC.mkdir(parents=True, exist_ok=True)
DST = ROOT / "data/chunks_dataset_labeled"; DST.mkdir(parents=True, exist_ok=True)
MODEL_ID = "tiny"  # CTranslate2-converted model

# ───────────────────── compute-type ──────────────────────
supported = ctranslate2.get_supported_compute_types("cpu")
for ct in ("int8_float16", "int8_bfloat16", "int8", "float32"):
    if ct in supported:
        COMPUTE_TYPE = ct
        break
print("🖥️  using", COMPUTE_TYPE, "on CPU")

# ───────────────────────── model ─────────────────────────
model = WhisperModel(
    MODEL_ID,
    device="cpu",
    compute_type=COMPUTE_TYPE,
    cpu_threads=0,
)


def _transcribe(path: str) -> str:
    segs, _ = model.transcribe(
        path,
        beam_size=1,
        vad_filter=False,
        word_timestamps=False,
    )
    return "".join(s.text for s in segs).strip().upper()


def _proc(batch):
    batch["text"] = [_transcribe(p) for p in batch["path"]]
    return batch


print("🔗  loading dataset …")
ds = datasets.load_from_disk(str(SRC))

print("📝  generating transcripts …")
for split in ds:
    ds[split] = ds[split].map(
        _proc,
        batched=True,
        batch_size=8,
        desc=f"ASR {split}",
    )

print("💾  saving …")
DST.mkdir(parents=True, exist_ok=True)
ds.save_to_disk(str(DST))
print("✅  done –", DST)
