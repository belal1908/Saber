"""CLI: record labeled taps per zone, fit the classifier, save weights.

Usage:
    python -m holo.model.train --samples-per-zone 15
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from holo.audio.capture import AudioCapture
from holo.config import IMPULSE_WINDOW_MS, SAMPLE_RATE, ZONES
from holo.dsp.features import extract_features
from holo.dsp.onset import OnsetDetector
from holo.model.classifier import ZoneClassifier


def record_zone_samples(capture: AudioCapture, detector: OnsetDetector, zone: str, n: int) -> list[np.ndarray]:
    print(f"\nTap the '{zone}' zone {n} times...")
    samples: list[np.ndarray] = []
    window_len = int(SAMPLE_RATE * IMPULSE_WINDOW_MS / 1000)

    while len(samples) < n:
        block = capture.next_block(timeout=5.0)
        if detector.process(block):
            time.sleep(IMPULSE_WINDOW_MS / 1000)  # let the impulse ring into the buffer
            window = capture.recent_audio()[-window_len:]
            if len(window) < window_len:
                continue
            samples.append(extract_features(window))
            print(f"  captured {len(samples)}/{n}")

    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-zone", type=int, default=15)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    capture = AudioCapture(device=args.device)
    detector = OnsetDetector()
    capture.start()

    X, y = [], []
    try:
        for zone in ZONES:
            input(f"Press Enter, then tap '{zone}'...")
            samples = record_zone_samples(capture, detector, zone, args.samples_per_zone)
            X.extend(samples)
            y.extend([zone] * len(samples))
    finally:
        capture.stop()

    clf = ZoneClassifier.fit(np.array(X), y)
    clf.save()
    print(f"\nSaved model to {clf}")


if __name__ == "__main__":
    main()
