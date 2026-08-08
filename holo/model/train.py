"""CLI: record labeled taps per zone, fit the classifier, save weights.

Usage:
    python -m holo.model.train --samples-per-zone 15
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from holo.audio.capture import AudioCapture
from holo.audio.probe import emit_and_capture, response_window
from holo.config import IMPULSE_WINDOW_MS, MODEL_PATH, SAMPLE_RATE, ZONES
from holo.dsp.adaptive_filter import AdaptiveNoiseFilter
from holo.dsp.features import extract_features
from holo.dsp.onset import OnsetDetector
from holo.model.classifier import ZoneClassifier


def record_zone_samples(
    capture: AudioCapture,
    detector: OnsetDetector,
    noise_filter: AdaptiveNoiseFilter,
    zone: str,
    n: int,
) -> list[np.ndarray]:
    print(f"\nTap the '{zone}' zone {n} times...")
    samples: list[np.ndarray] = []
    window_len = int(SAMPLE_RATE * IMPULSE_WINDOW_MS / 1000)

    while len(samples) < n:
        block = capture.next_block(timeout=5.0)
        is_onset = detector.process(block)

        if not is_onset:
            background = capture.recent_audio()[-window_len:]
            if len(background) == window_len:
                noise_filter.update(background)
            continue

        time.sleep(IMPULSE_WINDOW_MS / 1000)  # let the impulse ring into the buffer
        window = capture.recent_audio()[-window_len:]
        if len(window) < window_len:
            continue
        samples.append(extract_features(window, noise_filter=noise_filter))
        print(f"  captured {len(samples)}/{n}")

    return samples


def record_zone_samples_probe(zone: str, n: int, device: int | str | None) -> list[np.ndarray]:
    """Active-probe alternative: fire a chirp per sample instead of waiting for a tap."""
    print(f"\nRest a hand/object on '{zone}' and press Enter {n} times (or hold still for ambient probes).")
    samples: list[np.ndarray] = []
    for i in range(n):
        input(f"  probe {i + 1}/{n} — Enter to fire...")
        recording = emit_and_capture(device=device)
        samples.append(extract_features(response_window(recording)))
    return samples


def print_devices() -> None:
    for entry in AudioCapture.list_devices():
        print(f"[{entry['index']}] {entry['name']} (in: {entry['max_input_channels']} ch)")


def _parse_device(value: str) -> int | str:
    """--device accepts either a numeric index (see --list-devices) or a device
    name substring. sounddevice treats str devices as name matches, not indices,
    so a numeric-looking value must be converted to int or index lookups break."""
    try:
        return int(value)
    except ValueError:
        return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-zone", type=int, default=15)
    parser.add_argument(
        "--device",
        default=None,
        type=_parse_device,
        help="Input device index or name (see --list-devices).",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available input devices and exit.",
    )
    parser.add_argument(
        "--use-probe",
        action="store_true",
        help="Use the active chirp probe instead of passive tap detection.",
    )
    args = parser.parse_args()

    if args.list_devices:
        print_devices()
        return

    X, y = [], []

    if args.use_probe:
        for zone in ZONES:
            samples = record_zone_samples_probe(zone, args.samples_per_zone, args.device)
            X.extend(samples)
            y.extend([zone] * len(samples))
    else:
        capture = AudioCapture(device=args.device)
        detector = OnsetDetector()
        window_len = int(SAMPLE_RATE * IMPULSE_WINDOW_MS / 1000)
        noise_filter = AdaptiveNoiseFilter(fft_size=window_len)
        capture.start()
        try:
            for zone in ZONES:
                input(f"Press Enter, then tap '{zone}'...")
                samples = record_zone_samples(capture, detector, noise_filter, zone, args.samples_per_zone)
                X.extend(samples)
                y.extend([zone] * len(samples))
        finally:
            capture.stop()

    mode = "probe" if args.use_probe else "passive"
    clf = ZoneClassifier.fit(np.array(X), y, mode=mode, device=args.device)
    clf.save()
    print(f"\nSaved {mode}-mode model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
