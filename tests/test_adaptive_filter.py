import numpy as np

from holo.dsp.adaptive_filter import AdaptiveNoiseFilter

FFT_SIZE = 1024
SAMPLE_RATE = 48_000


def _sine(freq_hz: float, amplitude: float, n: int, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(n) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def test_apply_is_noop_before_any_update():
    filt = AdaptiveNoiseFilter(fft_size=FFT_SIZE)
    spectrum = np.abs(np.fft.rfft(_sine(2000, 1.0, FFT_SIZE)))
    filtered = filt.apply(spectrum)
    assert np.allclose(filtered, spectrum)


def test_update_converges_noise_spectrum_toward_background():
    filt = AdaptiveNoiseFilter(fft_size=FFT_SIZE, alpha=0.2)
    background = _sine(3000, 0.5, FFT_SIZE)
    for _ in range(200):
        filt.update(background)

    expected = np.abs(np.fft.rfft(background * np.hanning(FFT_SIZE)))
    assert np.allclose(filt.noise_spectrum, expected, atol=1e-2)


def test_apply_suppresses_matched_noise_more_than_novel_signal():
    filt = AdaptiveNoiseFilter(fft_size=FFT_SIZE, alpha=0.2, floor_ratio=0.0)
    background = _sine(3000, 0.5, FFT_SIZE)
    for _ in range(200):
        filt.update(background)

    # A spectrum identical to the tracked noise should be almost fully removed.
    noise_only_spectrum = np.abs(np.fft.rfft(background * np.hanning(FFT_SIZE)))
    filtered_noise = filt.apply(noise_only_spectrum)
    assert np.max(filtered_noise) < 0.05 * np.max(noise_only_spectrum)

    # A strong impulse at a different frequency (little energy in the tracked
    # noise estimate there) should survive largely intact.
    impulse = _sine(8000, 1.0, FFT_SIZE)
    impulse_spectrum = np.abs(np.fft.rfft(impulse * np.hanning(FFT_SIZE)))
    filtered_impulse = filt.apply(impulse_spectrum)
    assert np.max(filtered_impulse) > 0.9 * np.max(impulse_spectrum)


def test_apply_rejects_mismatched_spectrum_length():
    filt = AdaptiveNoiseFilter(fft_size=FFT_SIZE)
    wrong_size = np.zeros(FFT_SIZE)  # should be fft_size // 2 + 1
    try:
        filt.apply(wrong_size)
        assert False, "expected ValueError"
    except ValueError:
        pass
