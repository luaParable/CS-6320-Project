# ATC-Whisper 📡✈️
Automated Transcription & Entity Extraction for Air-Traffic-Control Speech

---

## 1 What this project does
* Converts raw tower / approach recordings (MP3 or WAV) into  
  – a clean transcript, and  
  – a colour-coded table of callsigns, headings, runways, etc.
* Runs locally on Windows or Linux; GPU strongly recommended (RTX-20xx+).
* “Light-touch” fine-tuning: only the language-model head (and _N_ last
  decoder blocks if you want) is updated, so training takes **minutes**, not
  hours.
* Ships with a FastAPI service + single-page web UI **and** a CLI that writes
  Markdown files ready for incident logs.

---

## 2 Repository layout

```
pv2/
├── api.py                  ← FastAPI service (JSON + Markdown)
├── CHANGELOG.md
├── README.md               ← this file
├── requirements.txt
├── train_asr_public_corpus.py
├── training_text.py        ← (optional experiment)
├── transcribe_md.py        ← Markdown helper (used by api.py & CLI)
│
├── input/
│   ├── *.mp3               ← raw archive
│   └── convert_mp3_to_wav.py
│
├── data/
│   ├── augment.py
│   ├── chunk_creation.py
│   ├── wav/                ← 16-kHz mono WAVs (auto-created)
│   ├── chunks/             ← silence chunks       "
│   ├── chunks_dataset/     ← HF dataset           "
│   └── mel_cache_freeze/   ← cached mel features  "
│
├── model/
│   └── whisper-atc-*       ← fine-tuned checkpoints
│
├── nlp/
│   ├── __init__.py
│   └── atc_ner.py          ← rule-based, low-FP NER
│
├── output/                 ← Markdown transcripts created by CLI
│
└── ui/
    └── frontend/
        ├── index.html
        ├── app.js
        └── style.css
```

---

## 3 Quick-start (Windows / PowerShell)

```powershell
# 0) clone & create venv
git clone <repo-url>
cd pv2
python -m venv .venv
.\.venv\Scripts\activate

# 1) install deps
pip install -r requirements.txt

# 2) convert MP3 → 16-kHz WAV
python input\convert_mp3_to_wav.py          # writes to data\wav\

# 3) split long WAVs into chunks + HF dataset
python data\chunk_creation.py

# 4) light-touch fine-tune (FAST_MODE = True in script → 20 steps)
python train_asr_public_corpus.py

# 5) start the API (auto-picks free port, serves UI)
python api.py
# console prints e.g.  🚀  http://127.0.0.1:8000/ui/

# 6) open that URL, upload an audio file, enjoy the transcript
```

---

## 4 Command-line transcription to Markdown

```powershell
python transcribe_md.py  path\to\audio.wav
# writes  output/<name>.md
```

Sample excerpt:

```md
## Fine-Tuned Whisper-ATC
AMERICAN FIVE SIX FOUR HEAVY DESCEND AND MAINTAIN ONE THREE THOUSAND.

| text                     | label      |
|--------------------------|------------|
| AMERICAN FIVE SIX FOUR   | CALLSIGN   |
| ONE THREE THOUSAND       | ALTITUDE   |
| DENVER APPROACH          | ATC_UNIT   |
```

---

## 5 Tweaking fine-tuning effort
Open `train_asr_public_corpus.py` and edit the **CONFIG block**:

```python
FAST_MODE        = True      # subset / full corpus
UNFREEZE_LAYERS  = 0         # 0=head only, 1=last block, 2=last-2 …
STEPS_FAST       = 20        # optimisation steps
STEPS_FULL       = 100
```

Set `UNFREEZE_LAYERS = 2` and `FAST_MODE = False` for a still-quick
15-minute run that improves WER ≈ 5 pp over frozen Whisper-tiny.

---

## 6 Named-Entity Recognition details

`nlp/atc_ner.py` combines spaCy **Matcher** patterns with a few precise
regexes.  Labels emitted:

| Label        | Example                      |
|--------------|------------------------------|
| CALLSIGN     | AMERICAN 564 / N123AB        |
| ATC_UNIT     | DENVER APPROACH              |
| ALTITUDE     | ONE THREE THOUSAND           |
| FLIGHT_LEVEL | FL TWO FOUR ZERO             |
| HEADING      | HEADING 225                  |
| RUNWAY       | RUNWAY 21L                   |
| FREQUENCY    | 118.3                        |
| AIRPORT      | KDEN                         |
| WAYPOINT     | EMPYR                        |
| COMMAND      | DESCEND AND MAINTAIN         |

False positives were reduced by  
① whitelisting real airline telephony words,  
② requiring 3-digit headings / 2-digit runways.

---

## 7 Testing
* `pytest` ensures every dev JSON has a callsign and that NER still
  finds it after each change.
* `/health` returns `{"status":"ok"}` – used by GitHub CI.

---

## 8 Demo video
[Watch the video](https://youtu.be/<your-id>)  
Runtime ≤ 2 minutes: overview → fast fine-tune → live UI demo.

---

## 9 License / Credits
Code MIT.  Audio data © ATCOSIM / ATCO-2 (ELRA-S0484) used under the
project licence.  Logo icons from Twemoji CC-BY-4.0.

---

*Last updated · see CHANGELOG.md for incremental steps.*