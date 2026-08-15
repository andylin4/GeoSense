"""Tests for change-triggered screen watching.

Time is passed into step() rather than slept, so the whole timing state machine
is exercised deterministically and instantly.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from geoguessr.data.crop import FULL_FRAME, CropSpec
from geoguessr.serve.predictor import Guess
from geoguessr.serve.watch import (
    ScreenWatcher,
    WatchConfig,
    frame_signature,
    signature_distance,
)


class StubPredictor:
    def __init__(self, crop=FULL_FRAME, error=None):
        self.crop = crop
        self.error = error
        self.calls = 0

    def predict(self, image, top_k=5):
        self.calls += 1
        if self.error:
            raise self.error
        return [Guess("PL", "Poland", 0.34)]


def solid(value: int, size=(160, 90)) -> Image.Image:
    return Image.new("RGB", size, (value, value, value))


class Screen:
    """A mutable fake display."""

    def __init__(self, image):
        self.image = image
        self.grabs = 0

    def __call__(self):
        self.grabs += 1
        return self.image


class TestSignature:
    def test_identical_frames_have_zero_distance(self):
        a = frame_signature(solid(120))
        assert signature_distance(a, a) == 0.0

    def test_different_frames_are_far_apart(self):
        assert signature_distance(frame_signature(solid(0)),
                                  frame_signature(solid(255))) > 0.9

    def test_missing_previous_is_infinite(self):
        assert signature_distance(frame_signature(solid(10)), None) == float("inf")

    def test_is_small_and_bounded(self):
        signature = frame_signature(solid(120))
        assert signature.shape == (32, 32)
        assert 0.0 <= signature.min() <= signature.max() <= 1.0


class TestConfigValidation:
    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"poll_interval": 0}, "poll_interval"),
            ({"change_threshold": 0}, "change_threshold"),
            ({"change_threshold": 1.5}, "change_threshold"),
            ({"settle_time": -1}, "settle_time"),
        ],
    )
    def test_rejects_nonsense(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            WatchConfig(**kwargs)


class TestStateMachine:
    @pytest.fixture
    def setup(self):
        screen = Screen(solid(50))
        predictor = StubPredictor()
        watcher = ScreenWatcher(
            predictor, grab=screen,
            config=WatchConfig(settle_time=0.4, min_gap=1.5, change_threshold=0.02),
        )
        return watcher, screen, predictor

    def test_first_frame_registers_as_a_change(self, setup):
        watcher, _, _ = setup
        assert watcher.step(0.0) == "changed"

    def test_static_screen_never_encodes(self, setup):
        watcher, _, predictor = setup
        watcher.step(0.0)                       # first frame
        results = [watcher.step(t) for t in (0.25, 0.5, 0.75)]

        assert "predicted" in results           # settles once, then goes idle
        assert results[-1] == "idle"
        assert predictor.calls == 1

    def test_encodes_after_change_settles(self, setup):
        watcher, screen, predictor = setup
        watcher.step(0.0)
        watcher.step(1.0)  # settle + predict the initial frame
        assert predictor.calls == 1

        screen.image = solid(200)
        assert watcher.step(10.0) == "changed"
        assert watcher.step(10.2) == "settling"   # not yet settled
        assert watcher.step(10.5) == "predicted"
        assert predictor.calls == 2

    def test_panning_does_not_encode_every_frame(self, setup):
        watcher, screen, predictor = setup
        # Scene changes on every tick, as when the player is dragging the view.
        for i, t in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
            screen.image = solid(20 + i * 40)
            assert watcher.step(t) == "changed"

        assert predictor.calls == 0  # never stable long enough

    def test_min_gap_throttles_rapid_scene_changes(self, setup):
        watcher, screen, predictor = setup
        watcher.step(0.0)
        watcher.step(1.0)
        assert predictor.calls == 1

        # A new scene settles, but too soon after the last encode.
        screen.image = solid(200)
        watcher.step(1.1)
        assert watcher.step(1.6) == "cooling"
        assert predictor.calls == 1

        # Once the gap has elapsed it goes through.
        assert watcher.step(3.0) == "predicted"
        assert predictor.calls == 2

    def test_small_changes_are_ignored(self, setup):
        watcher, screen, predictor = setup
        watcher.step(0.0)
        watcher.step(1.0)

        screen.image = solid(51)  # ~0.004 distance, below threshold
        assert watcher.step(5.0) == "idle"

    def test_polls_are_cheap_relative_to_predictions(self, setup):
        watcher, _, predictor = setup
        for t in range(40):
            watcher.step(t * 0.25)

        assert watcher.polls == 40
        assert predictor.calls == 1  # static screen -> one encode total


class TestCropAwareness:
    def test_changes_outside_the_crop_are_ignored(self):
        # A HUD timer ticking outside the crop must not trigger an encode.
        crop = CropSpec(name="top_half", bottom=0.5)
        predictor = StubPredictor(crop=crop)
        base = Image.new("RGB", (100, 100), (50, 50, 50))
        screen = Screen(base)

        watcher = ScreenWatcher(predictor, grab=screen,
                                config=WatchConfig(settle_time=0.0, min_gap=0.0))
        watcher.step(0.0)
        watcher.step(1.0)
        assert predictor.calls == 1

        changed = base.copy()
        changed.paste(Image.new("RGB", (100, 50), (255, 255, 255)), (0, 50))
        screen.image = changed

        assert watcher.step(2.0) == "idle"
        assert predictor.calls == 1


class TestCallbacksAndErrors:
    def test_on_result_receives_guesses(self):
        seen = []
        watcher = ScreenWatcher(
            StubPredictor(), grab=Screen(solid(50)),
            config=WatchConfig(settle_time=0.0, min_gap=0.0),
            on_result=lambda guesses, frame: seen.append(guesses),
        )
        watcher.step(0.0)
        watcher.step(1.0)
        assert seen[0][0].name == "Poland"

    def test_grab_failure_is_reported_not_raised(self):
        errors = []

        def broken():
            raise PermissionError("no screen recording")

        watcher = ScreenWatcher(StubPredictor(), grab=broken,
                                on_error=errors.append)
        assert watcher.step(0.0) == "error"
        assert "no screen recording" in str(errors[0])

    def test_prediction_failure_clears_dirty_so_it_does_not_spin(self):
        errors = []
        predictor = StubPredictor(error=RuntimeError("head exploded"))
        watcher = ScreenWatcher(
            predictor, grab=Screen(solid(50)),
            config=WatchConfig(settle_time=0.0, min_gap=0.0),
            on_error=errors.append,
        )
        watcher.step(0.0)
        assert watcher.step(1.0) == "error"
        assert watcher.dirty is False
        assert watcher.step(2.0) == "idle"  # does not retry forever
        assert predictor.calls == 1
