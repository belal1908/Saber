"""Optional active acoustic probe: emit a short ultrasonic-ish chirp and
capture the desk's response, as a supplement to passive tap detection.

Unlike passive listening (continuous, via AudioCapture), a probe is a
discrete request/response action — played and recorded in one blocking
call via sd.playrec so playback and capture stay sample-aligned.
"""
from __future__ import annotations

import numpy as np
import sounddevice as sd
from scipy.signal import chirp

from holo.config import SAMPLE_RATE

CHIRP_F0 = 15_500  # Hz
CHIRP_F1 = 21_000  # Hz
CHIRP_DURATION_MS = 20
CAPTURE_PADDING_MS = 60  # trailing window to catch the desk's response


def generate_chirp(
    duration_ms: int = CHIRP_DURATION_MS,
    f0: int = CHIRP_F0,
    f1: int = CHIRP_F1,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    n_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples, endpoint=False)
    signal = chirp(t, f0=f0, f1=f1, t1=duration_ms / 1000, method="linear")
    tapered = signal * np.hanning(n_samples)  # avoid clicks at chirp edges
    return tapered.astype(np.float32)


def emit_and_capture(
    sample_rate: int = SAMPLE_RATE,
    device: int | str | None = None,
    capture_padding_ms: int = CAPTURE_PADDING_MS,
) -> np.ndarray:
    """Play the probe chirp and simultaneously record the response.

    Returns the full sample-aligned recording (chirp + trailing response
    window), single channel, ready to hand to holo.dsp.features.extract_features.
    """
    probe = generate_chirp(sample_rate=sample_rate)
    pad_samples = int(sample_rate * capture_padding_ms / 1000)

    playback = np.zeros(len(probe) + pad_samples, dtype=np.float32)
    playback[: len(probe)] = probe

    recording = sd.playrec(
        playback.reshape(-1, 1),
        samplerate=sample_rate,
        channels=1,
        device=device,
        blocking=True,
    )
    return recording[:, 0]


def response_window(recording: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Slice off the emitted chirp, keeping only the desk's response."""
    chirp_len = int(sample_rate * CHIRP_DURATION_MS / 1000)
    return recording[chirp_len:]
