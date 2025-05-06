"""
Fast Whisper-tiny fine-tune for ATC data.

Optimisations over the “vanilla” script:
  1. Freeze encoder (we adapt only the decoder).
  2. 8-bit Adam (bitsandbytes) – lower VRAM, bigger batch.
  3. torch.compile() – kernel fusion (~10 % speed-up).
  4. Larger batch, no gradient accumulation.

Keeps   • Windows-safe multiprocessing   • Robust audio loader.
"""
from __future__ import annotations

import random
import warnings
from multiprocessing import freeze_support
from pathlib import Path

import datasets
import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import (
    WhisperForConditionalGeneration, WhisperProcessor,
    TrainingArguments, Trainer, set_seed,
)

from data.augment import augment


# ---------------------------------------------------------------------- #
# 1. helpers
# ---------------------------------------------------------------------- #
def safe_load(path: str):
    """torchaudio first; fall back to soundfile (handles edge-case WAVs)."""
    try:
        return torchaudio.load(path)
    except RuntimeError:
        data, sr = sf.read(path, always_2d=True)
        return torch.from_numpy(data.T), sr


# These globals are populated in main()
processor: WhisperProcessor
SAMPLE_RATE: int


def _preprocess(batch):
    wav, sr = safe_load(batch["path"])
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    wav = wav.mean(0)               # mono
    wav = augment(wav, SAMPLE_RATE)

    feats = processor.feature_extractor(
        wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
    ).input_features[0]

    txt = batch["text"] or ""
    if hasattr(processor, "as_target_processor"):
        with processor.as_target_processor():
            labels = processor(txt).input_ids
    else:  # transformers < 4.28 fallback
        labels = processor.tokenizer(txt).input_ids

    return {"input_features": feats, "labels": labels}


def collate(batch):
    feats = torch.stack([x["input_features"] for x in batch])
    labels = processor.tokenizer.pad(
        {"input_ids": [x["labels"] for x in batch]}, padding=True, return_tensors="pt"
    ).input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return {"input_features": feats, "labels": labels}


# ---------------------------------------------------------------------- #
# 2. main
# ---------------------------------------------------------------------- #
def main():
    freeze_support()  # Windows fix

    # Repro.
    SEED = 42
    set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    random.seed(SEED); np.random.seed(SEED)

    # Model & processor
    BASE = "openai/whisper-tiny"
    global processor, SAMPLE_RATE
    processor = WhisperProcessor.from_pretrained(BASE)
    model = WhisperForConditionalGeneration.from_pretrained(BASE)
    SAMPLE_RATE = processor.feature_extractor.sampling_rate

    # ---- Speed tricks -------------------------------------------------
    # 2.1 freeze encoder
    for p in model.model.encoder.parameters():
        p.requires_grad = False

    # 2.2 torch.compile (PyTorch >= 2)
    if hasattr(torch, "compile"):
        model = torch.compile(model)

    # 2.3 try 8-bit Adam
    try:
        from bitsandbytes.optim import Adam8bit
        optim = Adam8bit(model.parameters(), lr=1e-5)
        print("🚀  Using bitsandbytes Adam8bit optimiser")
    except Exception as e:
        from torch.optim import AdamW
        optim = AdamW(model.parameters(), lr=1e-5)
        warnings.warn(f"bitsandbytes unavailable ({e}); using AdamW")

    # Dataset
    dpath = Path("data/chunks_dataset")
    if not dpath.exists():
        raise SystemExit("❌  dataset missing – run data/chunk_creation.py first.")
    ds = datasets.load_from_disk(dpath)

    num_proc = 1
    ds = ds.map(
        _preprocess,
        remove_columns=ds["train"].column_names,
        num_proc=num_proc,
        desc="Extracting Whisper features",
    ).with_format("torch")

    # TrainingArguments
    args = TrainingArguments(
        output_dir="model/whisper-atc",
        per_device_train_batch_size=32,      # ↑ batch
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,       # ↓ accum
        learning_rate=1e-5,
        max_steps=3000,
        warmup_steps=200,
        evaluation_strategy="steps",
        eval_steps=300,
        save_steps=300,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["dev"],
        tokenizer=processor.feature_extractor,
        data_collator=collate,
        optimizers=(optim, None),
    )

    trainer.train()
    trainer.save_model("model/whisper-atc")
    processor.save_pretrained("model/whisper-atc")
    print("✅  Training finished – model saved to  model/whisper-atc")


if __name__ == "__main__":
    main()