from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from holo.config import ONSET_ENERGY_MULTIPLIER, SAMPLE_RATE


def rms(block: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(block)) + 1e-12))


@dataclass
class OnsetDetector:
    """Adaptive energy-threshold impulse detector.

    Tracks a slow-moving noise floor and fires when a block's RMS spikes
    above it by ONSET_ENERGY_MULTIPLIER, with a short refractory period so
    a single tap doesn't trigger multiple times.
    """

    sample_rate: int = SAMPLE_RATE
    noise_floor_alpha: float = 0.05
    refractory_blocks: int = 8

    _noise_floor: float = field(default=1e-4, init=False)
    _refractory_count: int = field(default=0, init=False)

    def process(self, block: np.ndarray) -> bool:
        level = rms(block)

        if self._refractory_count > 0:
            self._refractory_count -= 1
            self._noise_floor = (
                1 - self.noise_floor_alpha
            ) * self._noise_floor + self.noise_floor_alpha * level
            return False

        is_onset = level > self._noise_floor * ONSET_ENERGY_MULTIPLIER

        if is_onset:
            self._refractory_count = self.refractory_blocks
        else:
            self._noise_floor = (
                1 - self.noise_floor_alpha
            ) * self._noise_floor + self.noise_floor_alpha * level

        return is_onset


@dataclass
class ImpulseCapture:
    """Tracks the delay between detecting a tap onset and the point where a
    full impulse window has accumulated in the ring buffer.

    Feature extraction needs the window *following* an onset (the tap's
    ring/decay), not the window ending at the instant of detection. In a
    blocking context you'd just time.sleep(window_s) then read the buffer;
    this is the same idea as a non-blocking state machine for poll-driven
    callers (e.g. a GUI timer) that can't sleep on every tick.
    """

    window_s: float
    deadline: float | None = field(default=None, init=False)

    def on_block(self, is_onset: bool, now: float) -> None:
        if is_onset and self.deadline is None:
            self.deadline = now + self.window_s

    @property
    def pending(self) -> bool:
        return self.deadline is not None

    def ready(self, now: float) -> bool:
        return self.deadline is not None and now >= self.deadline

    def consume(self) -> None:
        self.deadline = None
