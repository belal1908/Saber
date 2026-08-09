import json

import numpy as np

from holo.model.classifier import ZoneClassifier

ZONES = ["rear-left", "rear-right", "front-left", "front-right"]


def _toy_dataset():
    rng = np.random.default_rng(0)
    X, y = [], []
    for i, zone in enumerate(ZONES):
        center = np.eye(len(ZONES))[i] * 5.0
        for _ in range(10):
            X.append(center + rng.normal(scale=0.1, size=len(ZONES)))
            y.append(zone)
    return np.array(X), y


def test_fit_defaults_to_passive_mode():
    X, y = _toy_dataset()
    clf = ZoneClassifier.fit(X, y)
    assert clf.mode == "passive"


def test_fit_records_probe_mode():
    X, y = _toy_dataset()
    clf = ZoneClassifier.fit(X, y, mode="probe")
    assert clf.mode == "probe"


def test_save_load_round_trip_preserves_mode_and_device(tmp_path):
    X, y = _toy_dataset()
    clf = ZoneClassifier.fit(X, y, mode="probe", device=2)
    path = tmp_path / "model.json"
    clf.save(path)

    loaded = ZoneClassifier.load(path)
    assert loaded.mode == "probe"
    assert loaded.device == 2
    assert loaded.classes == clf.classes
    assert np.allclose(loaded.coef, clf.coef)
    assert np.allclose(loaded.intercept, clf.intercept)


def test_save_load_round_trip_preserves_string_device_name(tmp_path):
    X, y = _toy_dataset()
    clf = ZoneClassifier.fit(X, y, device="USB Mic")
    path = tmp_path / "model.json"
    clf.save(path)

    assert ZoneClassifier.load(path).device == "USB Mic"


def test_load_defaults_to_passive_for_models_saved_before_mode_tracking(tmp_path):
    """Older model.json files won't have "mode"/"device" keys — must not crash, must use safe defaults."""
    path = tmp_path / "model.json"
    path.write_text(json.dumps({"classes": ZONES, "coef": [[0.0] * 4] * 4, "intercept": [0.0] * 4}))

    loaded = ZoneClassifier.load(path)
    assert loaded.mode == "passive"
    assert loaded.device is None


def test_predict_picks_highest_scoring_zone():
    X, y = _toy_dataset()
    clf = ZoneClassifier.fit(X, y)
    for i, zone in enumerate(ZONES):
        probe_features = np.eye(len(ZONES))[i] * 5.0
        assert clf.predict(probe_features) == zone


def test_scores_sum_to_one_and_agree_with_predict():
    X, y = _toy_dataset()
    clf = ZoneClassifier.fit(X, y)
    for i, zone in enumerate(ZONES):
        features = np.eye(len(ZONES))[i] * 5.0
        scores = clf.scores(features)
        assert set(scores) == set(ZONES)
        assert abs(sum(scores.values()) - 1.0) < 1e-9
        assert max(scores, key=scores.get) == clf.predict(features) == zone


def test_scores_reflect_uncertainty_for_ambiguous_input():
    """Equidistant features should come out close to uniform, not confidently
    wrong for one class — this is exactly what a "random-looking" real-world
    misclassification would look like under the hood."""
    X, y = _toy_dataset()
    clf = ZoneClassifier.fit(X, y)
    ambiguous = np.zeros(len(ZONES))
    scores = clf.scores(ambiguous)
    assert max(scores.values()) < 0.9
