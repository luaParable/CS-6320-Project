"""
Fast Whisper-tiny fine-tune without PEFT.
  • 8-bit load if possible, else fp16/fp32
  • freeze everything except last 2 decoder blocks + head
  • cached mel dataset
  • --fast flag  (10 k / 2 k subset, 400 steps)

No dependency newer than your current environment is strictly required.
"""
from __future__ import annotations

import argparse
import os
import platform
import random
from functools import lru_cache
from multiprocessing import freeze_support
from pathlib import Path
from typing import Dict, List

import datasets
import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ─────────────────────────────────── CLI ──────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--fast", action="store_true", help="10 k / 2 k subset, 400 steps")
CFG = ap.parse_args()

# ────────────────────────── helpers ───────────────────────────────────────
@lru_cache(maxsize=1)
def proc() -> WhisperProcessor:
    return WhisperProcessor.from_pretrained("openai/whisper-tiny")


def safe_load(path: str):
    try:
        return torchaudio.load(path)
    except RuntimeError:
        data, sr = sf.read(path, always_2d=True)
        return torch.from_numpy(data.T), sr


def _preprocess(ex: Dict):
    p = proc()
    trg_sr = p.feature_extractor.sampling_rate
    wav, sr = safe_load(ex["path"])
    if sr != trg_sr:
        wav = torchaudio.functional.resample(wav, sr, trg_sr)
    wav = wav.mean(0)

    feats = p.feature_extractor(
        wav.numpy(), sampling_rate=trg_sr, return_tensors="pt"
    ).input_features[0]

    txt = ex["text"] or ""
    if hasattr(p, "as_target_processor"):          # transformers ≥4.28
        with p.as_target_processor():
            labels = p(txt).input_ids
    else:                                          # fallback
        labels = p.tokenizer(txt).input_ids
    return {"input_features": feats, "labels": labels}


def collate(batch: List[Dict]):
    p = proc()
    feats = torch.stack([b["input_features"] for b in batch])
    labels = p.tokenizer.pad(
        {"input_ids": [b["labels"] for b in batch]}, padding=True, return_tensors="pt"
    ).input_ids
    labels[labels == p.tokenizer.pad_token_id] = -100
    return {"input_features": feats, "labels": labels}


# ─────────────────────────── main ─────────────────────────────────────────
def main():
    freeze_support()  # Windows
    SEED = 42
    set_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    # 1. 8-bit load if bitsandbytes GPU wheel present
    try:
        import bitsandbytes as bnb  # noqa
        use8 = bnb.cuda.is_available()
    except Exception:
        use8 = False

    print(f"🔗  loading Whisper-tiny ({'8-bit' if use8 else 'fp16/fp32'})")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny",
        load_in_8bit=use8,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if use8 else None,
    )

    # freeze all
    model.requires_grad_(False)
    # unfreeze last 2 decoder layers
    for blk in model.model.decoder.layers[-2:]:
        blk.requires_grad_(True)
    # decoder LN (name varies)
    for ln_name in ("layer_norm", "final_layer_norm", "ln_post"):
        if hasattr(model.model.decoder, ln_name):
            getattr(model.model.decoder, ln_name).requires_grad_(True)
    # output head
    if hasattr(model, "proj_out"):
        model.proj_out.requires_grad_(True)
    elif hasattr(model, "lm_head"):
        model.lm_head.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"🧮  trainable params: {trainable/1e6:.2f} M / {total/1e6:.1f} M")

    # 2. dataset with cached mels
    RAW = Path("data/chunks_dataset")
    MEL = RAW.parent / "mel_cache_freeze"
    if MEL.exists():
        print("⚡  using cached mel dataset")
        ds = datasets.load_from_disk(MEL)
    else:
        print("🎧  extracting mel spectrograms (one-time)…")
        ds = datasets.load_from_disk(RAW)
        num_proc = 1 if platform.system() == "Windows" else os.cpu_count()
        for split in ds:
            ds[split] = ds[split].map(
                _preprocess,
                remove_columns=ds[split].column_names,
                num_proc=num_proc,
                desc=f"mel {split}",
            )
        ds.save_to_disk(MEL)
        print("✅  cached to", MEL)

    if CFG.fast:
        ds["train"] = ds["train"].shuffle(SEED).select(range(10_000))
        ds["dev"]   = ds["dev"].shuffle(SEED).select(range(2_000))
        max_steps = 400
    else:
        max_steps = 4_000

    ds.set_format("torch")

    # 3. build TrainingArguments dict only with supported keys
    bs = 128 if torch.cuda.is_available() else 32
    targs_kwargs = dict(
        output_dir="model/whisper-atc",
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        gradient_accumulation_steps=1,
        learning_rate=3e-4,
        max_steps=max_steps,
        warmup_steps=int(max_steps * 0.1),
        logging_steps=20,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    # add keys present in newer transformers
    if "evaluation_strategy" in TrainingArguments.__init__.__code__.co_varnames:
        targs_kwargs["evaluation_strategy"] = "no"

    train_args = TrainingArguments(**targs_kwargs)

    # 4. optimiser
    if use8:
        from bitsandbytes.optim import Adam8bit as Optim
    else:
        from torch.optim import AdamW as Optim

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=ds["dev"],
        tokenizer=proc().feature_extractor,
        data_collator=collate,
        optimizers=(Optim(model.parameters(), lr=3e-4), None),
    )

    print("🚀  training…")
    trainer.train()
    model.save_pretrained("model/whisper-atc")
    proc().save_pretrained("model/whisper-atc")
    print("🏁 done – model stored in  model/whisper-atc")


if __name__ == "__main__":
    main()