"""Menu-bar app: listens for taps and dispatches zone actions.

Run with: python -m holo.app
Requires a trained model at data/model.json (see holo/model/train.py).
"""
from __future__ import annotations

import rumps

from holo.actions.dispatch import dispatch
from holo.actions.registry import load_actions
from holo.audio.capture import AudioCapture
from holo.config import IMPULSE_WINDOW_MS, MODEL_PATH, SAMPLE_RATE
from holo.dsp.adaptive_filter import AdaptiveNoiseFilter
from holo.dsp.features import extract_features
from holo.dsp.onset import OnsetDetector
from holo.model.classifier import ZoneClassifier


class HoloApp(rumps.App):
    def __init__(self) -> None:
        super().__init__("Holo", title="◎")
        self.capture = AudioCapture()
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

        self.menu = ["Start Listening", "Stop Listening"]

    @rumps.clicked("Start Listening")
    def start(self, _) -> None:
        if self.listening or self.classifier is None:
            return
        self.capture.start()
        self.listening = True
        self.title = "◉"
        self.poll_timer = rumps.Timer(self.poll, 0.02)
        self.poll_timer.start()

    @rumps.clicked("Stop Listening")
    def stop(self, _) -> None:
        if not self.listening:
            return
        self.poll_timer.stop()
        self.capture.stop()
        self.listening = False
        self.title = "◎"

    def poll(self, _timer) -> None:
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
        zone = self.classifier.predict(features)
        action = self.actions.get(zone)
        if action:
            dispatch(zone, action)


def main() -> None:
    HoloApp().run()


if __name__ == "__main__":
    main()
