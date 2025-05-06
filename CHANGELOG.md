────────────────────────────────────────────────────────────────────────────
1.  Initial goal / baseline
────────────────────────────────────────────────────────────────────────────
    • Target: build an ATC-specific Whisper system (training, NER, REST API + UI).  
    • Provided starting files:  
    − atc_ner.py (regex-based NER)  
    − train_asr.py (Whisper-tiny fine-tune on local chunks)  
    − serve.py (FastAPI; fine-tuned + baseline)  
    − HTML front-end.

────────────────────────────────────────────────────────────────────────────
2.  Expanded NER
────────────────────────────────────────────────────────────────────────────
    • Added robust regexes: callsigns, controller units, heavy/super suffixes.  
    • Added annotate() helper.  
    • Made package-ready with `__init__.py`.

────────────────────────────────────────────────────────────────────────────
3.  Audio processing & augmentation
────────────────────────────────────────────────────────────────────────────
    • augment.py introduced: noise, speed-perturb, gain.  
    • chunk_creation.py: silence-based WAV splitter, HF dataset with train/dev.  
    • Added `convert_mp3_to_wav.py` (FFmpeg helper).

────────────────────────────────────────────────────────────────────────────
4.  Training pipeline(s)
────────────────────────────────────────────────────────────────────────────
    • train_asr.py refactor:  
    − HuggingFace Trainer, fp16, cache-friendly preprocessing.  
    − Windows-safe (freeze_support, num_proc=1).  
    • “Turbo” version: encoder frozen, torch.compile, optional 8-bit Adam; batch 64.  
    • Public-corpus script (train_asr_public_corpus.py):  
    − Downloads `jacktol/atc-dataset` → later switched to `luigisaetta/atco2_atcosim`.  
    − Added log-mel extraction, pad_or_trim, empty-clip filter.  
    − Added `--fast` flag → subset + few steps.  
    − Multiple bug-fix rounds (NumPy broadcast, STFT padding, empty rows).  
    − Final constants-only CONFIG block with:  
    FAST_MODE, UNFREEZE_LAYERS, STEPS, SUBSET sizes, LR, BATCH_SIZE.

────────────────────────────────────────────────────────────────────────────
5.  Serve / API evolution
────────────────────────────────────────────────────────────────────────────
    • Re-wrote serve.py multiple times:  
    − Switched to Transformers model loading (baseline stayed openai-whisper).  
    − Fixed Windows audio-decoding (exclusive temp-file, torchaudio → FFmpeg fallback).  
    − Removed forced_decoder_ids clash with TF-4.38.  
    − Added CORS, /health endpoint, redirect / → /ui.  
    − Final simplified version: only fine-tuned model, /transcribe endpoint,
    static SPA, entity extraction.

────────────────────────────────────────────────────────────────────────────
6.  Command-line transcription tool
────────────────────────────────────────────────────────────────────────────
    • tools/transcribe_to_md.py:  
    − Reads WAV(s) → Markdown transcript + entity table.

────────────────────────────────────────────────────────────────────────────
7.  Fine-tune scope dial-back
────────────────────────────────────────────────────────────────────────────
    • Progressively reduced the number of trainable parameters:  
    − Only lm_head + final LN unfrozen.  
    − Added CLI flags (--layers, --steps) → later replaced by CONFIG constants.  
    − FAST_MODE default True, 2 k / 500 rows, 20 steps → ultra-quick.

────────────────────────────────────────────────────────────────────────────
8.  Final minimal-impact training script
────────────────────────────────────────────────────────────────────────────
    • All hyper-parameters moved to a CONFIG block.  
    • No argparse; edit constants in the file instead.  
    • Works on Windows without bitsandbytes; falls back to AdamW.  
    • Ready for quick experimentation:  
    UNFREEZE_LAYERS, STEPS_FAST / STEPS_FULL, LR, BATCH_SIZE etc.

────────────────────────────────────────────────────────────────────────────
Current state
────────────────────────────────────────────────────────────────────────────
• Fine-tuned model can be (lightly) updated in minutes.  
• Single CLI tool generates `.md` transcripts.  
• REST API serves `/transcribe` with robust audio loader and entity output.  
• NER covers common ATC phraseology.