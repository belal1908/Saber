from __future__ import annotations

import numpy as np
from scipy.fft import rfft, rfftfreq

from holo.config import SAMPLE_RATE

N_MEL_FILTERS = 20
N_MFCC = 13


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(n_fft: int, sample_rate: int, n_filters: int) -> np.ndarray:
    freqs = rfftfreq(n_fft, d=1.0 / sample_rate)
    mel_min, mel_max = _hz_to_mel(np.array([0.0, sample_rate / 2]))
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    hz_points = _mel_to_hz(mel_points)

    bank = np.zeros((n_filters, len(freqs)))
    for i in range(1, n_filters + 1):
        left, center, right = hz_points[i - 1], hz_points[i], hz_points[i + 1]
        left_slope = (freqs - left) / max(center - left, 1e-6)
        right_slope = (right - freqs) / max(right - center, 1e-6)
        bank[i - 1] = np.clip(np.minimum(left_slope, right_slope), 0, None)
    return bank


def extract_features(window: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """FFT -> mel filterbank -> log -> DCT (MFCC-style) plus spectral shape features."""
    windowed = window * np.hanning(len(window))
    spectrum = np.abs(rfft(windowed))
    n_fft = len(window)

    bank = _mel_filterbank(n_fft, sample_rate, N_MEL_FILTERS)
    mel_energies = bank @ spectrum
    log_mel = np.log(mel_energies + 1e-8)

    # DCT-II for MFCCs, no external dependency
    n = np.arange(N_MEL_FILTERS)
    mfcc = np.zeros(N_MFCC)
    for k in range(N_MFCC):
        mfcc[k] = np.sum(log_mel * np.cos(np.pi * k * (2 * n + 1) / (2 * N_MEL_FILTERS)))

    freqs = rfftfreq(n_fft, d=1.0 / sample_rate)
    power = spectrum**2
    total_power = power.sum() + 1e-8
    centroid = float(np.sum(freqs * power) / total_power)
    rolloff_threshold = 0.85 * total_power
    cumulative = np.cumsum(power)
    rolloff_idx = int(np.searchsorted(cumulative, rolloff_threshold))
    rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    return np.concatenate([mfcc, [centroid, rolloff]])
