import nlp.atc_ner as atc_ner
import tempfile
import torchaudio
import whisper
from fastapi import FastAPI, UploadFile

model = whisper.load_model("model/whisper-atc")
app = FastAPI()


@app.post("/transcribe")
async def transcribe(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
    audio, _ = torchaudio.load(tmp.name)
    text = model.transcribe(audio)["text"]
    entities = atc_ner.annotate(text)
    return {"text": text, "entities": entities}
