import torch
import torchaudio
from numpy.random import random


def augment(wave):
    if random() < 0.5:
        wave = torchaudio.transforms.Resample(16000, int(16000 * random.uniform(0.9, 1.1)))(wave)
    if random() < 0.5:
        wave += 0.05 * torch.randn_like(wave)
    return wave
