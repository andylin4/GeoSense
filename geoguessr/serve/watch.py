"""Continuous screen watching, without continuous inference.

Polling the screen and encoding every frame would pin the GPU for no benefit: a
GeoGuessr scene is static most of the time, and a CLIP encode costs ~120x more
than a screen grab. Sustained MPS load on a fanless Air thermally throttles,
so this matters in practice, not just in theory.

So this is change-triggered rather than interval-triggered:

    poll (10ms)  ->  signature changed?  ->  settled?  ->  cooled down?  -> encode

Three gates stand between a changed pixel and an encode:

* **change threshold** -- ignore compression noise and cursor movement.
* **settle time** -- while the player is panning, the frame changes every tick;
  encoding then would describe a motion-blurred intermediate view. Waiting for
  stillness means one encode per look, not one per frame.
* **minimum gap** -- a hard ceiling on encode rate regardless of activity, which
  is the thermal guard.

The signature is a small grayscale thumbnail. That deliberately ignores fine
detail: what matters is "did the scene change", not "did a leaf move".
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from PIL.Image import Image

    from .predictor import Predictor

__all__ = ["WatchConfig", "ScreenWatcher", "frame_signature", "signature_distance"]

SIGNATURE_SIZE = 32


def frame_signature(image: Image, size: int = SIGNATURE_SIZE) -> np.ndarray:
    """Cheap perceptual fingerprint: a small normalized grayscale thumbnail."""
    thumb = image.convert("L").resize((size, size))
    return np.asarray(thumb, dtype=np.float32) / 255.0


def signature_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Mean absolute difference in [0, 1]. Infinite if either side is missing."""
    if a is None or b is None:
        return float("inf")
    return float(np.abs(a - b).mean())


@dataclass
class WatchConfig:
    """Tuning for the change-detection loop. Defaults suit GeoGuessr."""

    poll_interval: float = 0.25   # seconds between cheap screen grabs
    change_threshold: float = 0.02  # mean abs diff that counts as "changed"
    settle_time: float = 0.4      # stillness required before encoding
    min_gap: float = 1.5          # hard floor between encodes (thermal guard)

    def __post_init__(self) -> None:
        if self.poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if not 0.0 < self.change_threshold < 1.0:
            raise ValueError("change_threshold must be in (0, 1)")
        if self.settle_time < 0 or self.min_gap < 0:
            raise ValueError("settle_time and min_gap must be non-negative")


class ScreenWatcher:
    """Watches the screen and predicts only when the scene meaningfully changes.

    The loop is expressed as :meth:`step`, a pure-ish state machine taking the
    current time, so the timing behaviour is testable without sleeping or
    touching a real display.
    """

    def __init__(
        self,
        predictor: Predictor,
        *,
        config: WatchConfig | None = None,
        monitor: int = 1,
        grab: Callable[[], Image] | None = None,
        on_result: Callable[[list, Image], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        top_k: int = 5,
    ):
        self.predictor = predictor
        self.config = config or WatchConfig()
        self.monitor = monitor
        self.top_k = top_k
        self.on_result = on_result
        self.on_error = on_error
        self._grab = grab or self._default_grab

        self.last_signature: np.ndarray | None = None
        self.changed_at: float | None = None
        self.predicted_at: float = -float("inf")
        self.dirty = False
        self.predictions = 0
        self.polls = 0

    def _default_grab(self) -> Image:
        from .capture import grab_screen

        return grab_screen(monitor=self.monitor)

    def step(self, now: float) -> str:
        """Advance one tick. Returns what happened, for logging and tests.

        One of: ``"changed"``, ``"predicted"``, ``"settling"``, ``"cooling"``,
        ``"idle"``, ``"error"``.
        """
        self.polls += 1

        try:
            frame = self._grab()
        except Exception as exc:
            if self.on_error:
                self.on_error(exc)
            return "error"

        # Compare the region the model actually sees, so HUD animations such as
        # a ticking timer outside the crop never trigger an encode.
        signature = frame_signature(self.predictor.crop.apply(frame))
        distance = signature_distance(signature, self.last_signature)

        if distance > self.config.change_threshold:
            self.last_signature = signature
            self.changed_at = now
            self.dirty = True
            return "changed"

        if not self.dirty:
            return "idle"

        # `is None`, not `or` -- a timestamp of 0.0 is falsy and would make this
        # compare now against itself, so the frame would never settle.
        changed_at = now if self.changed_at is None else self.changed_at
        if now - changed_at < self.config.settle_time:
            return "settling"

        if now - self.predicted_at < self.config.min_gap:
            return "cooling"

        try:
            guesses = self.predictor.predict(frame, top_k=self.top_k)
        except Exception as exc:
            self.dirty = False
            if self.on_error:
                self.on_error(exc)
            return "error"

        self.dirty = False
        self.predicted_at = now
        self.predictions += 1
        if self.on_result:
            self.on_result(guesses, frame)
        return "predicted"

    def run(self, stop: threading.Event | None = None) -> None:
        """Poll until ``stop`` is set. Intended to run on a worker thread."""
        stop = stop or threading.Event()
        while not stop.is_set():
            self.step(time.monotonic())
            stop.wait(self.config.poll_interval)

    def start(self) -> tuple[threading.Thread, threading.Event]:
        """Start watching on a daemon thread. Returns ``(thread, stop_event)``."""
        stop = threading.Event()
        thread = threading.Thread(target=self.run, args=(stop,), daemon=True)
        thread.start()
        return thread, stop
