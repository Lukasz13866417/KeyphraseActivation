from __future__ import annotations

import random
from typing import Optional

import torch


def _random_time_mask(
    spec: torch.Tensor,
    max_width_frac: float = 0.15,
    num_masks: int = 1,
) -> torch.Tensor:
    """
    Remove random blocks of time from the spectrogram. (make them silent)
    """
    if spec.dim() != 2:
        raise ValueError("Expected 2D spectrogram [n_mels, T].")

    n_mels, T = spec.shape
    if T <= 1:
        return spec

    out = spec
    max_width = max(1, int(T * max_width_frac))
    for _ in range(max(0, int(num_masks))):
        width = random.randint(1, max_width)
        if width >= T:
            start = 0
        else:
            start = random.randint(0, T - width)
        end = min(T, start + width)
        out[:, start:end] = 0.0
    return out


def _random_freq_mask(
    spec: torch.Tensor,
    max_width_frac: float = 0.2,
    num_masks: int = 1,
) -> torch.Tensor:
    """
    Remove random blocks of frequency from the spectrogram. (make these frequencies silent)
    """
    if spec.dim() != 2:
        raise ValueError("Expected 2D spectrogram [n_mels, T].")

    n_mels, T = spec.shape
    if n_mels <= 1:
        return spec

    out = spec
    max_width = max(1, int(n_mels * max_width_frac))
    for _ in range(max(0, int(num_masks))):
        width = random.randint(1, max_width)
        if width >= n_mels:
            start = 0
        else:
            start = random.randint(0, n_mels - width)
        end = min(n_mels, start + width)
        out[start:end, :] = 0.0
    return out


def _additive_noise(
    spec: torch.Tensor,
    std: float = 0.03,
) -> torch.Tensor:
    """
    Add small Gaussian noise.
    """
    if std <= 0.0:
        return spec
    noise = torch.randn_like(spec) * float(std)
    return spec + noise


def apply_training_augmentations(
    spec: torch.Tensor,
    *,
    time_mask_prob: float = 0.8,
    freq_mask_prob: float = 0.8,
    noise_prob: float = 0.5,
) -> torch.Tensor:
    """
    Apply lightweight, training-time augmentations to a single log-mel spectrogram.

    Expects 2D tensor [n_mels, T] (the spectrogram) and returns an augmented tensor of the same shape.
    """
    if spec.dim() != 2:
        raise ValueError("Expected 2D spectrogram [n_mels, T].")

    out = spec

    if random.random() < time_mask_prob:
        out = _random_time_mask(out, max_width_frac=0.15, num_masks=1)

    if random.random() < freq_mask_prob:
        out = _random_freq_mask(out, max_width_frac=0.2, num_masks=1)

    if random.random() < noise_prob:
        out = _additive_noise(out, std=0.03)

    return out


