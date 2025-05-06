"""
Minimal-impact Whisper-tiny fine-tune on ATCOSIM corpus.

•  change behaviour by editing the CONFIG block – no CLI flags.
"""
from __future__ import annotations
import inspect, os, random
from functools import lru_cache
from multiprocessing import freeze_support
from pathlib import Path
from typing import Dict, List

import datasets
import numpy as np
import torch, torchaudio
import whisper.audio as wa
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ────────── CONFIG – tweak here, nothing else ──────────
FAST_MODE        = True      # True  → subset + few steps
UNFREEZE_LAYERS  = 4         # 0=head-only, 1=last block, 2=last-2 blocks …
STEPS_FAST       = 20        # max update steps if FAST_MODE
STEPS_FULL       = 400       # max update steps otherwise
SUBSET_TRAIN     = 2_000     # rows when FAST_MODE
SUBSET_DEV       = 500
BATCH_SIZE       = 64        # per-device batch
LEARNING_RATE    = 1e-5
OUTPUT_DIR       = "model/whisper-atc-11"
DATASET_ID       = "luigisaetta/atco2_atcosim"
SR               = 16_000
MAX_LEN          = SR * 30
# ───────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / OUTPUT_DIR; OUTDIR.mkdir(parents=True, exist_ok=True)
SPLITS = {"train": "train", "dev": "test"}


@lru_cache
def proc() -> WhisperProcessor:
    return WhisperProcessor.from_pretrained("openai/whisper-tiny")


def _valid(ex):  # at least 400 samples
    arr = ex["audio"]["array"]
    return isinstance(arr, np.ndarray) and arr.ndim == 1 and arr.size >= 400


def _pad_trim(x: np.ndarray) -> np.ndarray:
    x = x.flatten() if x.ndim else np.zeros(1, np.float32)
    return np.pad(x, (0, MAX_LEN - x.size)) if x.size < MAX_LEN else x[:MAX_LEN]


def _preprocess(ex: Dict):
    p       = proc()
    trg_sr  = p.feature_extractor.sampling_rate

    wav = torch.tensor(ex["audio"]["array"], dtype=torch.float32)
    if ex["audio"]["sampling_rate"] != trg_sr:
        wav = torchaudio.functional.resample(wav.unsqueeze(0),
                                             ex["audio"]["sampling_rate"],
                                             trg_sr).squeeze(0)
    wav   = wav.mean(0)
    feats = wa.log_mel_spectrogram(torch.from_numpy(_pad_trim(wav.numpy())))

    text   = ex.get("text") or ex.get("sentence") or ex.get("transcription") or ""
    labels = p.tokenizer(text).input_ids            # ← replaces as_target_processor

    return {"input_features": feats, "labels": labels}


def collate(batch: List[Dict]):
    p = proc()
    feats = torch.stack([b["input_features"] for b in batch])
    labels = p.tokenizer.pad({"input_ids": [b["labels"] for b in batch]},
                             padding=True, return_tensors="pt").input_ids
    labels[labels == p.tokenizer.pad_token_id] = -100
    return {"input_features": feats, "labels": labels}


def main():
    freeze_support(); set_seed(42); random.seed(42); np.random.seed(42)

    # model
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.requires_grad_(False)                           # freeze all
    for head in ("proj_out", "lm_head"):                  # unfreeze LM head
        if hasattr(model, head):
            getattr(model, head).requires_grad_(True)
    if UNFREEZE_LAYERS > 0:                               # optional extra layers
        for blk in model.model.decoder.layers[-UNFREEZE_LAYERS:]:
            blk.requires_grad_(True)

    # dataset (cached)
    cache = Path("data/mel_cache_atcosim")
    if cache.exists():
        ds = datasets.load_from_disk(cache)
    else:
        ds = {k: datasets.load_dataset(DATASET_ID, split=v) for k, v in SPLITS.items()}
        ds = datasets.DatasetDict(ds).cast_column("audio", datasets.Audio())
        for s in ds:
            ds[s] = (ds[s].filter(_valid).map(_preprocess,
                                              remove_columns=ds[s].column_names,
                                              desc=f"mel {s}"))
        ds.save_to_disk(cache)

    if FAST_MODE:
        ds["train"] = ds["train"].shuffle(42).select(range(SUBSET_TRAIN))
        ds["dev"]   = ds["dev"].shuffle(42).select(range(SUBSET_DEV))
        max_steps   = STEPS_FAST
    else:
        max_steps   = STEPS_FULL
    ds.set_format("torch")

    # TrainingArguments (only supported keys)
    base = dict(
        output_dir=str(OUTDIR),
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        max_steps=max_steps,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    args = TrainingArguments(**{k: v for k, v in base.items()
                                if k in inspect.signature(TrainingArguments).parameters})

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["dev"],
        tokenizer=proc().feature_extractor,
        data_collator=collate,
        optimizers=(torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE), None),
    )

    trainer.train()
    model.save_pretrained(OUTDIR); proc().save_pretrained(OUTDIR)
    print("✓ finished – model saved to", OUTDIR)


if __name__ == "__main__":
    main()