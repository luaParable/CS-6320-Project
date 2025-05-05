```markdown
ATC-Whisper - End-to-End Transcription Demo
==========================================

This repo fine-tunes Whisper-tiny on Air-Traffic-Control recordings,
tags entities (callsigns, controller units) and offers a small web UI
to compare the custom model with a vanilla Whisper model.

Project tree
------------

pv2/
├── README.md                ← **you are here**
├── requirements.txt
├── train_asr.py
├── serve.py
│
├── input/
│   ├── *.mp3                ← ORIGINAL MP3 files
│   └── convert_mp3_to_wav.py
│
├── data/
│   ├── wav/                 ← 16 kHz mono WAV files (auto–created)
│   ├── chunks/              ← silence-based chunks   (auto–created)
│   ├── chunks_dataset/      ← HF dataset w/ train & dev (auto–created)
│   ├── augment.py
│   └── chunk_creation.py
│
├── model/                   ← fine-tuned model gets stored here
│
├── nlp/
│   ├── __init__.py
│   └── atc_ner.py
│
└── ui/
    └── frontend/
        └── index.html

Quick-start
-----------

1. Install system packages (Ubuntu example – adjust to your OS):

   ```bash
   sudo apt-get update && sudo apt-get install -y ffmpeg
   ```

2. Create a virtual-env and install Python deps:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Convert the original MP3s to 16 kHz WAV:

   ```bash
   python input/convert_mp3_to_wav.py
   ```

4. Split WAV files into chunks & build the HuggingFace dataset:

   ```bash
   python data/chunk_creation.py
   ```

5. Fine-tune Whisper-tiny (GPU recommended, ~few hours):

   ```bash
   python train_asr.py
   ```

   The resulting model is written to `model/whisper-atc/`.

6. Run the API + Web UI:

   ```bash
   uvicorn serve:app --reload
   ```

   Open http://localhost:8000/ui/ in your browser,
   upload an audio file and click the buttons.

Notes
-----

*   If you already have your own transcripts, place them in
    `data/chunks_dataset/` (after step 4) before running `train_asr.py`.
*   `train_asr.py` expects ≈ ≥ 2 GB of GPU RAM; reduce `per_device_*_batch_size`
    if you hit OOM.
```