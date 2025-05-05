"""
Light-weight, stateless audio augmentation for ATC training data.
Returns 16 kHz mono tensors – exactly what Whisper expects.
"""
from __future__ import annotations
import random
from typing import Tuple

import torch
import torchaudio

TARGET_SR = 16_000


# ---------------------------------------------------------------------- #
def _maybe(fn, p: float = 0.5):
    return fn if random.random() < p else (lambda x, *a, **kw: x)


def _random_resample(wave: torch.Tensor, sr: int = TARGET_SR) -> torch.Tensor:
    """Speed-perturbation between 0.9× and 1.1×, then back to TARGET_SR."""
    factor = random.uniform(0.9, 1.1)
    new_sr = int(sr * factor)
    return torchaudio.functional.resample(
        torchaudio.functional.resample(wave, sr, new_sr), new_sr, TARGET_SR
    )


def _add_noise(wave: torch.Tensor, snr_db: Tuple[int, int] = (15, 25)) -> torch.Tensor:
    """Add white noise at a random SNR."""
    snr = random.uniform(*snr_db)
    noise = torch.randn_like(wave)
    wave_power = wave.norm(p=2)
    noise_power = noise.norm(p=2)
    factor = (wave_power / (10 ** (snr / 20))) / (noise_power + 1e-6)
    return wave + factor * noise


def augment(wave: torch.Tensor, sr: int = TARGET_SR) -> torch.Tensor:
    """
    50 %  speed-perturb
    50 %  add gaussian noise (15–25 dB SNR)
    50 %  random gain (-6 … +6 dB)
    """
    wave = _maybe(lambda w: _random_resample(w, sr), 0.5)(wave)
    wave = _maybe(_add_noise, 0.5)(wave)
    wave = _maybe(lambda w: torchaudio.functional.gain(w, random.uniform(-6, 6)), 0.5)(
        wave
    )
    return wave.clamp_(-1.0, 1.0)