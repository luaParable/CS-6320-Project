"""
1. Slice every WAV in ./data/wav/ into silence-based chunks
2. Save the chunks into ./data/chunks/
3. Build a HuggingFace dataset with TRAIN / DEV splits
   and store it at ./data/chunks_dataset/

Run this AFTER you converted mp3 → wav (`input/convert_mp3_to_wav.py`).
"""
from __future__ import annotations
from pathlib import Path

import datasets
from pydub import AudioSegment
from pydub.silence import split_on_silence

# ---------------------------------------------------------------------- #
WAV_DIR       = Path(__file__).parent / "wav"
CHUNK_DIR     = Path(__file__).parent / "chunks"
DATASET_OUT   = Path(__file__).parent / "chunks_dataset"

CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------- #
print("🔪  Splitting WAV files into chunks …")
for wav_file in WAV_DIR.glob("*.wav"):
    audio = AudioSegment.from_wav(wav_file)
    pieces = split_on_silence(
        audio,
        min_silence_len=100,            # ms
        silence_thresh=audio.dBFS - 14,
        keep_silence=100,               # ms
    )
    for i, segment in enumerate(pieces):
        out = CHUNK_DIR / f"{wav_file.stem}-chunk{i}.wav"
        segment.export(out, format="wav")

# ---------------------------------------------------------------------- #
print("📦  Building HuggingFace dataset (train / dev) …")
paths = sorted(str(p.resolve()) for p in CHUNK_DIR.glob("*.wav"))
base_ds = datasets.Dataset.from_dict({"path": paths, "text": [""] * len(paths)})
splits  = base_ds.train_test_split(test_size=0.1, seed=42)
ds      = datasets.DatasetDict({"train": splits["train"], "dev": splits["test"]})

DATASET_OUT.mkdir(parents=True, exist_ok=True)
ds.save_to_disk(str(DATASET_OUT))
print(
    f"✅  Dataset saved to {DATASET_OUT} "
    f"({len(ds['train'])} train / {len(ds['dev'])} dev examples)"
)
