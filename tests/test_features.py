import numpy as np

from holo.dsp.features import extract_features
from holo.dsp.onset import OnsetDetector


def test_extract_features_shape():
    window = np.random.randn(4096).astype(np.float32) * 0.01
    features = extract_features(window)
    assert features.shape == (15,)  # 13 MFCCs + centroid + rolloff
    assert np.all(np.isfinite(features))


def test_onset_detector_fires_on_spike():
    detector = OnsetDetector()
    quiet = np.zeros(1024, dtype=np.float32)
    for _ in range(20):
        assert detector.process(quiet) is False

    loud = np.ones(1024, dtype=np.float32) * 0.5
    assert detector.process(loud) is True
