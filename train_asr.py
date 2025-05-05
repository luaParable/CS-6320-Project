import datasets
import whisper

from data.augment import augment

model = whisper.load_model("small")
ds = datasets.load_from_disk("data/whisper_chunks")


def preprocess(batch):
    audio = augment(batch["audio"])
    batch["input_features"] = whisper.log_mel_spectrogram(audio)
    batch["labels"] = whisper.tokenizer.encode(batch["text"])
    return batch


ds = ds.map(preprocess, remove_columns=ds.column_names)
trainer = whisper.get_trainer(model, ds["train"], ds["dev"])
trainer.train(max_steps=500)
model.save_pretrained("model/whisper-atc")
