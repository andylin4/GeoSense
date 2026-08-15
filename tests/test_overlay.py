"""Tests for the overlay's threading contract.

The window itself is not exercised -- Tk needs a display and macOS demands the
main thread. What is tested is the part that would actually break in use: the
worker never touches a widget, errors reach the queue instead of killing the
thread, and a second hotkey press while busy is ignored.
"""

from __future__ import annotations

import queue

import pytest

from geoguessr.serve.overlay import DEFAULT_HOTKEY, Overlay
from geoguessr.serve.predictor import Guess


class StubPredictor:
    scaler = None

    def __init__(self, guesses=None, error=None):
        self.guesses = guesses or [Guess("PL", "Poland", 0.34)]
        self.error = error
        self.calls = 0

    def predict(self, image, top_k=5):
        self.calls += 1
        if self.error:
            raise self.error
        return self.guesses[:top_k]


@pytest.fixture
def fake_capture(monkeypatch):
    """Replace screen capture so no permission or display is needed."""
    from PIL import Image

    import geoguessr.serve.capture as capture

    monkeypatch.setattr(capture, "grab_screen",
                        lambda **kwargs: Image.new("RGB", (64, 36), "green"))


def drain(overlay) -> list[tuple[str, object]]:
    events = []
    try:
        while True:
            events.append(overlay._events.get_nowait())
    except queue.Empty:
        pass
    return events


class TestWorker:
    def test_successful_prediction_lands_on_the_queue(self, fake_capture):
        predictor = StubPredictor()
        overlay = Overlay(predictor)

        overlay._capture_and_predict()

        kind, payload = drain(overlay)[0]
        assert kind == "result"
        assert payload[0].name == "Poland"
        assert predictor.calls == 1

    def test_capture_failure_becomes_an_event_not_a_crash(self, monkeypatch):
        import geoguessr.serve.capture as capture

        monkeypatch.setattr(
            capture, "grab_screen",
            lambda **kwargs: (_ for _ in ()).throw(PermissionError("no permission")),
        )
        overlay = Overlay(StubPredictor())

        overlay._capture_and_predict()  # must not raise

        kind, payload = drain(overlay)[0]
        assert kind == "error"
        assert "no permission" in str(payload)

    def test_prediction_failure_becomes_an_event(self, fake_capture):
        overlay = Overlay(StubPredictor(error=RuntimeError("head exploded")))
        overlay._capture_and_predict()

        kind, payload = drain(overlay)[0]
        assert kind == "error"
        assert "head exploded" in str(payload)

    def test_respects_top_k(self, fake_capture):
        guesses = [Guess(f"C{i}", f"Country{i}", 0.1) for i in range(10)]
        overlay = Overlay(StubPredictor(guesses), top_k=3)

        overlay._capture_and_predict()
        assert len(drain(overlay)[0][1]) == 3


class TestHotkeyDebounce:
    def test_press_enqueues_working_then_runs(self, fake_capture):
        overlay = Overlay(StubPredictor())
        overlay._on_hotkey()

        # Worker runs on a thread; wait for it to finish via the queue.
        kinds = []
        for _ in range(200):
            kinds += [k for k, _ in drain(overlay)]
            if "result" in kinds:
                break
        assert kinds[0] == "working"
        assert "result" in kinds

    def test_second_press_while_busy_is_ignored(self, fake_capture):
        predictor = StubPredictor()
        overlay = Overlay(predictor)
        overlay._busy = True  # pretend a prediction is in flight

        overlay._on_hotkey()

        assert predictor.calls == 0
        assert drain(overlay) == []


class TestConfiguration:
    def test_default_hotkey(self):
        assert Overlay(StubPredictor()).hotkey == DEFAULT_HOTKEY

    def test_hotkey_is_overridable(self):
        assert Overlay(StubPredictor(), hotkey="<ctrl>+g").hotkey == "<ctrl>+g"

    def test_starts_idle(self):
        overlay = Overlay(StubPredictor())
        assert overlay._busy is False
        assert overlay._root is None
