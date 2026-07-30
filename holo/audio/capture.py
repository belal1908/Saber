from __future__ import annotations

import queue
from collections import deque
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from holo.config import BLOCK_SIZE, SAMPLE_RATE


@dataclass
class AudioCapture:
    """Continuous mic capture into a ring buffer, with an onset callback hook."""

    sample_rate: int = SAMPLE_RATE
    block_size: int = BLOCK_SIZE
    ring_seconds: float = 1.0
    device: int | str | None = None

    def __post_init__(self) -> None:
        ring_len = int(self.sample_rate * self.ring_seconds)
        self._ring: deque[np.ndarray] = deque(maxlen=ring_len // self.block_size + 1)
        self._blocks: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream: sd.InputStream | None = None

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            pass  # xruns etc. — surface via logging if needed
        mono = indata[:, 0].copy()
        self._ring.append(mono)
        self._blocks.put_nowait(mono)

    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def recent_audio(self) -> np.ndarray:
        """Flattened view of the ring buffer, most-recent last."""
        if not self._ring:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(list(self._ring))

    def next_block(self, timeout: float | None = None) -> np.ndarray:
        return self._blocks.get(timeout=timeout)

    @staticmethod
    def list_devices() -> list[dict]:
        return [d for d in sd.query_devices() if d["max_input_channels"] > 0]
