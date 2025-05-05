"""
Fine-tune Whisper-tiny on ATC chunks – Windows-safe and with a robust loader.

Run after:
  • `input/convert_mp3_to_wav.py`
  • `data/chunk_creation.py`
"""
from __future__ import annotations

import os
import platform
import random
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


# ────────────────────────────────────────────────────────────────────────
def safe_load(path: str):
    """
    Try torchaudio; fall back to SoundFile if torchaudio cannot decode.
    Returns   waveform (channels×samples), sample_rate.
    """
    try:
        return torchaudio.load(path)
    except RuntimeError:
        data, sr = sf.read(path, always_2d=True)       # samples×channels
        return torch.from_numpy(data.T), sr            # → channels×samples


# globals filled in main()
processor: WhisperProcessor
SAMPLE_RATE: int


def _preprocess(batch):
    wav, sr = safe_load(batch["path"])
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    wav = wav.mean(0)                     # mono
    wav = augment(wav, SAMPLE_RATE)

    inp = processor.feature_extractor(
        wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
    ).input_features[0]

    with processor.as_target_processor():
        lab = processor(batch["text"]).input_ids
    return {"input_features": inp, "labels": lab}


def collate(ex):
    feats = torch.stack([x["input_features"] for x in ex])
    labels = processor.tokenizer.pad(
        {"input_ids": [x["labels"] for x in ex]}, padding=True, return_tensors="pt"
    ).input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return {"input_features": feats, "labels": labels}


# ────────────────────────────────────────────────────────────────────────
def main():
    freeze_support()                       # Windows multiprocessing fix
    SEED = 42
    set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    random.seed(SEED); np.random.seed(SEED)

    global processor, SAMPLE_RATE
    BASE = "openai/whisper-tiny"
    processor = WhisperProcessor.from_pretrained(BASE)
    model     = WhisperForConditionalGeneration.from_pretrained(BASE)
    SAMPLE_RATE = processor.feature_extractor.sampling_rate

    dpath = Path("data/chunks_dataset")
    if not dpath.exists():
        raise SystemExit("❌ dataset missing – run data/chunk_creation.py first.")
    ds = datasets.load_from_disk(dpath)
    num_proc = 1 if platform.system() == "Windows" else os.cpu_count()

    ds = ds.map(
        _preprocess,
        remove_columns=ds["train"].column_names,
        num_proc=num_proc,
        desc="Extracting Whisper features",
    ).with_format("torch")

    args = TrainingArguments(
        output_dir="model/whisper-atc",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=250,
        max_steps=4000,
        evaluation_strategy="steps",
        eval_steps=250,
        save_steps=250,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        report_to="none",
    )

    Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["dev"],
        tokenizer=processor.feature_extractor,
        data_collator=collate,
    ).train()

    model.save_pretrained("model/whisper-atc")
    processor.save_pretrained("model/whisper-atc")
    print("✅  Training finished – model stored in model/whisper-atc")


if __name__ == "__main__":
    main()