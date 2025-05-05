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
print("🔪  Splitting wav files into chunks …")
for wav_file in WAV_DIR.glob("*.wav"):
    audio = AudioSegment.from_wav(wav_file)
    pieces = split_on_silence(
        audio,
        min_silence_len=100,
        silence_thresh=audio.dBFS - 14,
        keep_silence=100,
    )
    for i, seg in enumerate(pieces):
        out = CHUNK_DIR / f"{wav_file.stem}-chunk{i}.wav"
        (
            seg.set_frame_rate(16_000)
            .set_channels(1)
            .set_sample_width(2)          # 16-bit
            .export(out, format="wav", parameters=["-acodec", "pcm_s16le"])
        )

print("📦  Building HF dataset …")
paths  = sorted(str(p) for p in CHUNK_DIR.glob("*.wav"))
base   = datasets.Dataset.from_dict({"path": paths, "text": [""] * len(paths)})
splits = base.train_test_split(test_size=0.1, seed=42)
ds     = datasets.DatasetDict({"train": splits["train"], "dev": splits["test"]})
DATASET_OUT.mkdir(parents=True, exist_ok=True)
ds.save_to_disk(DATASET_OUT)
print(f"✅  Saved {len(ds['train'])}+{len(ds['dev'])} examples → {DATASET_OUT}")