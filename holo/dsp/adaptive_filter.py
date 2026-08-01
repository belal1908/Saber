"""Adaptive noise filtering via spectral subtraction.

Tracks a running estimate of the ambient noise spectrum from non-impulse
blocks (fan noise, typing, music) and subtracts it from impulse-window
spectra before feature extraction, so classification stays stable as
background conditions drift over a session.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.fft import rfft

from holo.config import SAMPLE_RATE


def _pad_or_trim(block: np.ndarray, n: int) -> np.ndarray:
    if len(block) == n:
        return block
    if len(block) > n:
        return block[:n]
    return np.pad(block, (0, n - len(block)))


@dataclass
class AdaptiveNoiseFilter:
    fft_size: int
    sample_rate: int = SAMPLE_RATE
    alpha: float = 0.05  # EMA rate for the noise spectrum estimate
    subtraction_factor: float = 1.0
    floor_ratio: float = 0.05  # keep at least this fraction of original magnitude

    _noise_spectrum: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        n_bins = self.fft_size // 2 + 1
        self._noise_spectrum = np.zeros(n_bins)

    def update(self, background_window: np.ndarray) -> None:
        """Feed a non-impulse window to refine the ambient noise estimate."""
        padded = _pad_or_trim(background_window, self.fft_size)
        magnitude = np.abs(rfft(padded * np.hanning(self.fft_size)))
        self._noise_spectrum = (1 - self.alpha) * self._noise_spectrum + self.alpha * magnitude

    def apply(self, spectrum: np.ndarray) -> np.ndarray:
        """Spectral-subtract the tracked noise floor from an impulse spectrum."""
        if spectrum.shape != self._noise_spectrum.shape:
            raise ValueError(
                f"spectrum length {spectrum.shape} does not match tracked fft_size "
                f"{self._noise_spectrum.shape}"
            )
        subtracted = spectrum - self.subtraction_factor * self._noise_spectrum
        floor = self.floor_ratio * spectrum
        return np.maximum(subtracted, floor)

    @property
    def noise_spectrum(self) -> np.ndarray:
        return self._noise_spectrum.copy()
