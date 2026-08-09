"""CLI: live per-zone confidence scores for a trained model.

Useful when predictions look wrong or inconsistent — this shows the full
softmax distribution per tap/probe instead of just the winning zone, so you
can tell a close call (needs more/cleaner training data) apart from no
separation at all (the features aren't distinguishing these zones on this
desk/mic).

Usage:
    python -m holo.model.diagnose
    python -m holo.model.diagnose --device 0
"""
from __future__ import annotations

import argparse
import time

from holo.audio.capture import AudioCapture
from holo.audio.probe import emit_and_capture, response_window
from holo.config import IMPULSE_WINDOW_MS, SAMPLE_RATE
from holo.dsp.adaptive_filter import AdaptiveNoiseFilter
from holo.dsp.features import extract_features
from holo.dsp.onset import ImpulseCapture, OnsetDetector
from holo.model.classifier import ZoneClassifier
from holo.model.train import _parse_device


def format_scores(scores: dict[str, float], predicted: str) -> str:
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    parts = [f"{'*' if zone == predicted else ' '}{zone}: {prob:.0%}" for zone, prob in ranked]
    return "  ".join(parts)


def run_passive(clf: ZoneClassifier, device: int | str | None) -> None:
    capture = AudioCapture(device=device)
    detector = OnsetDetector()
    window_len = int(SAMPLE_RATE * IMPULSE_WINDOW_MS / 1000)
    noise_filter = AdaptiveNoiseFilter(fft_size=window_len)
    impulse_capture = ImpulseCapture(window_s=IMPULSE_WINDOW_MS / 1000)
    capture.start()
    print("Listening (passive mode) — tap your desk. Ctrl+C to stop.\n")
    try:
        while True:
            now = time.monotonic()
            try:
                block = capture.next_block(timeout=1.0)
            except Exception:
                continue
            is_onset = detector.process(block)
            impulse_capture.on_block(is_onset, now)

            if not is_onset and not impulse_capture.pending:
                background = capture.recent_audio()[-window_len:]
                if len(background) == window_len:
                    noise_filter.update(background)

            if impulse_capture.ready(now):
                impulse_capture.consume()
                window = capture.recent_audio()[-window_len:]
                if len(window) < window_len:
                    continue
                features = extract_features(window, noise_filter=noise_filter)
                print(format_scores(clf.scores(features), clf.predict(features)))
    finally:
        capture.stop()


def run_probe(clf: ZoneClassifier, device: int | str | None) -> None:
    print("Probe mode — press Enter to fire a chirp and see scores. Ctrl+C to stop.\n")
    while True:
        input("  probe — Enter to fire...")
        recording = emit_and_capture(device=device)
        features = extract_features(response_window(recording))
        print(format_scores(clf.scores(features), clf.predict(features)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default=None,
        type=_parse_device,
        help="Input device index or name; defaults to the device the model was trained on.",
    )
    args = parser.parse_args()

    clf = ZoneClassifier.load()
    device = args.device if args.device is not None else clf.device
    print(f"Loaded {clf.mode}-mode model (trained on device {clf.device!r}); using device {device!r}\n")

    if clf.mode == "probe":
        run_probe(clf, device)
    else:
        run_passive(clf, device)


if __name__ == "__main__":
    main()
