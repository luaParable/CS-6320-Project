"""
Fine-tune Whisper-tiny on the chunks prepared in `data/chunk_creation.py`.
"""
from __future__ import annotations
import os
import random
from pathlib import Path

import datasets
import numpy as np
import torch
import torchaudio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    TrainingArguments,
    Trainer,
    set_seed,
)

from data.augment import augment

# ---------------------------------------------------------------------- #
SEED = 42
set_seed(SEED)
torch.backends.cudnn.deterministic = True
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------- #
BASE_MODEL = "openai/whisper-tiny"
processor = WhisperProcessor.from_pretrained(BASE_MODEL)
model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
SAMPLE_RATE = processor.feature_extractor.sampling_rate  # 16 000

HF_DATASET_DIR = Path("data/chunks_dataset")
assert HF_DATASET_DIR.exists(), (
    "Dataset missing – run data/chunk_creation.py first."
)


# ---------------------------------------------------------------------- #
def _preprocess(batch):
    waveform, sr = torchaudio.load(batch["path"])
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    waveform = waveform.mean(0)  # mono

    waveform = augment(waveform, SAMPLE_RATE)

    input_features = processor.feature_extractor(
        waveform.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
    ).input_features[0]

    with processor.as_target_processor():
        labels = processor(batch["text"]).input_ids

    return {"input_features": input_features, "labels": labels}


print("🔄  Loading dataset …")
ds = datasets.load_from_disk(str(HF_DATASET_DIR))
assert "train" in ds and "dev" in ds

ds = ds.map(
    _preprocess,
    remove_columns=ds["train"].column_names,
    num_proc=os.cpu_count(),
    desc="Preparing Whisper features",
)

ds.set_format(type="torch", columns=["input_features", "labels"])


# ---------------------------------------------------------------------- #
def data_collator(batch):
    input_features = torch.stack([x["input_features"] for x in batch])
    labels = processor.tokenizer.pad(
        {"input_ids": [x["labels"] for x in batch]}, padding=True, return_tensors="pt"
    ).input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return {"input_features": input_features, "labels": labels}


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

if __name__ == "__main__":
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["dev"],
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model("model/whisper-atc")
    processor.save_pretrained("model/whisper-atc")
