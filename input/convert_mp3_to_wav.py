"""
Convert every *.mp3 in ./input/ to 16 kHz mono WAV and write the result to
../data/wav/.

Requires `ffmpeg` to be installed on the system (tested with ffmpeg ≥ 4.0).
"""
from __future__ import annotations
import subprocess
from pathlib import Path

INPUT_DIR = Path(__file__).parent
WAV_DIR   = INPUT_DIR.parent / "data" / "wav"
WAV_DIR.mkdir(parents=True, exist_ok=True)

MP3S = list(INPUT_DIR.glob("*.mp3"))
if not MP3S:
    print("No .mp3 files found in input/.  Nothing to do.")
    raise SystemExit(0)

print(f"Converting {len(MP3S)} mp3 files to wav …")
for mp3 in MP3S:
    wav = WAV_DIR / f"{mp3.stem}.wav"
    cmd = [
        "ffmpeg",
        "-y",                # overwrite
        "-i",  str(mp3),
        "-ar", "16000",      # sample-rate
        "-ac", "1",          # mono
        str(wav),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("✓", wav)
print("All done – WAVs available in data/wav/")
