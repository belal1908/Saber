import numpy as np

from holo.audio.probe import CHIRP_DURATION_MS, generate_chirp, response_window
from holo.config import SAMPLE_RATE


def test_generate_chirp_length_and_range():
    signal = generate_chirp()
    expected_len = int(SAMPLE_RATE * CHIRP_DURATION_MS / 1000)
    assert signal.shape == (expected_len,)
    assert np.all(np.isfinite(signal))
    assert np.max(np.abs(signal)) <= 1.0


def test_generate_chirp_tapered_edges():
    signal = generate_chirp()
    assert abs(signal[0]) < 1e-3
    assert abs(signal[-1]) < 1e-3


def test_response_window_slices_off_chirp():
    chirp_len = int(SAMPLE_RATE * CHIRP_DURATION_MS / 1000)
    recording = np.concatenate([np.ones(chirp_len), np.full(500, 2.0)])
    response = response_window(recording)
    assert len(response) == 500
    assert np.all(response == 2.0)
