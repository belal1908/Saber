"""Menu-bar app: listens for taps and dispatches zone actions.

Run with: python -m holo.app
Requires a trained model at data/model.json (see holo/model/train.py).
"""
from __future__ import annotations

import rumps

from holo.actions.dispatch import dispatch
from holo.actions.registry import load_actions
from holo.audio.capture import AudioCapture
from holo.audio.probe import emit_and_capture, response_window
from holo.config import IMPULSE_WINDOW_MS, MODEL_PATH, PROBE_POLL_INTERVAL_S, SAMPLE_RATE
from holo.dsp.adaptive_filter import AdaptiveNoiseFilter
from holo.dsp.features import extract_features
from holo.dsp.onset import OnsetDetector
from holo.model.classifier import ZoneClassifier


class HoloApp(rumps.App):
    def __init__(self) -> None:
        super().__init__("Holo", title="◎")
        self.detector = OnsetDetector()
        self.actions = load_actions()
        self.classifier: ZoneClassifier | None = None
        self.listening = False
        self.window_len = int(SAMPLE_RATE * IMPULSE_WINDOW_MS / 1000)
        self.noise_filter = AdaptiveNoiseFilter(fft_size=self.window_len)

        if MODEL_PATH.exists():
            self.classifier = ZoneClassifier.load()
        else:
            rumps.alert("No trained model found. Run: python -m holo.model.train")

        # Use the same input device the model was trained against, so runtime
        # inference doesn't silently fall back to whatever the OS default happens to be.
        device = self.classifier.device if self.classifier is not None else None
        self.capture = AudioCapture(device=device)

        self.menu = ["Start Listening", "Stop Listening"]

    @rumps.clicked("Start Listening")
    def start(self, _) -> None:
        if self.listening or self.classifier is None:
            return
        self.listening = True
        self.title = "◉"
        if self.classifier.mode == "probe":
            self.poll_timer = rumps.Timer(self.poll_probe, PROBE_POLL_INTERVAL_S)
        else:
            self.capture.start()
            self.poll_timer = rumps.Timer(self.poll_passive, 0.02)
        self.poll_timer.start()

    @rumps.clicked("Stop Listening")
    def stop(self, _) -> None:
        if not self.listening:
            return
        self.poll_timer.stop()
        if self.classifier.mode != "probe":
            self.capture.stop()
        self.listening = False
        self.title = "◎"

    def poll_passive(self, _timer) -> None:
        try:
            block = self.capture.next_block(timeout=0.0)
        except Exception:
            return

        is_onset = self.detector.process(block)

        if not is_onset:
            background = self.capture.recent_audio()[-self.window_len :]
            if len(background) == self.window_len:
                self.noise_filter.update(background)
            return

        window = self.capture.recent_audio()[-self.window_len :]
        if len(window) < self.window_len:
            return
        features = extract_features(window, noise_filter=self.noise_filter)
        self._dispatch_for(features)

    def poll_probe(self, _timer) -> None:
        recording = emit_and_capture(device=self.capture.device)
        features = extract_features(response_window(recording))
        self._dispatch_for(features)

    def _dispatch_for(self, features) -> None:
        zone = self.classifier.predict(features)
        action = self.actions.get(zone)
        if action:
            dispatch(zone, action)


def main() -> None:
    HoloApp().run()


if __name__ == "__main__":
    main()
