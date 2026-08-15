"""Always-on-top overlay driven by a global hotkey.

Three things share this process and none of them may block the others:

* **Tkinter** must own the main thread -- that is a hard macOS requirement.
* **pynput**'s hotkey listener runs on its own thread and only ever enqueues an
  event; it must never touch a widget directly.
* **Inference** takes ~1.2s on MPS, so it runs on a worker thread. Doing it
  inline would freeze the window mid-round, which is exactly when it matters.

The three communicate through one queue that the Tk event loop drains on a
timer. That keeps every widget mutation on the main thread.

macOS permissions, both required and both silently failing without a restart of
the host terminal:

* **Accessibility** -- for the global hotkey (System Settings -> Privacy &
  Security -> Accessibility)
* **Screen Recording** -- for the capture itself
"""

from __future__ import annotations

import queue
import threading
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .predictor import Predictor

__all__ = ["Overlay", "DEFAULT_HOTKEY"]

DEFAULT_HOTKEY = "<cmd>+<shift>+g"

_BG = "#11141a"
_FG = "#e8ecf2"
_ACCENT = "#7fd1ff"
_MUTED = "#7c8798"


class Overlay:
    """A small borderless window that ranks countries when you press a key."""

    def __init__(
        self,
        predictor: Predictor,
        *,
        hotkey: str = DEFAULT_HOTKEY,
        monitor: int = 1,
        top_k: int = 5,
        position: tuple[int, int] = (40, 60),
        watch: bool = False,
        watch_config=None,
    ):
        self.predictor = predictor
        self.hotkey = hotkey
        self.monitor = monitor
        self.top_k = top_k
        self.position = position
        self.watch = watch
        self.watch_config = watch_config

        self._events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._busy = False
        self._root = None
        self._body = None
        self._status = None
        self._watcher = None
        self._watch_stop = None

    # --- worker side (never touches widgets) --------------------------------

    def _capture_and_predict(self) -> None:
        from .capture import grab_screen

        try:
            image = grab_screen(monitor=self.monitor)
            guesses = self.predictor.predict(image, top_k=self.top_k)
            self._events.put(("result", guesses))
        except Exception as exc:  # surfaced in the window, not the terminal
            traceback.print_exc()
            self._events.put(("error", exc))

    def _on_hotkey(self) -> None:
        if self._busy:
            return
        self._busy = True
        self._events.put(("working", None))
        threading.Thread(target=self._capture_and_predict, daemon=True).start()

    # --- main thread --------------------------------------------------------

    def _drain(self) -> None:
        try:
            while True:
                kind, payload = self._events.get_nowait()

                if kind == "working":
                    self._status.config(text="reading screen...", fg=_ACCENT)
                elif kind == "result":
                    self._render(payload)
                    self._busy = False
                elif kind == "error":
                    self._body.config(text=str(payload), fg="#ff9b9b")
                    self._status.config(text="error", fg="#ff9b9b")
                    self._busy = False
        except queue.Empty:
            pass

        self._root.after(50, self._drain)

    def _render(self, guesses) -> None:
        width = max(len(g.name) for g in guesses) if guesses else 10
        lines = [f"{g.name:<{width}}  {g.probability * 100:5.1f}%" for g in guesses]
        self._body.config(text="\n".join(lines), fg=_FG)

        suffix = "" if self.predictor.scaler else "  (uncalibrated)"
        mode = "watching" if self.watch else self.hotkey
        if self._watcher is not None:
            mode += f"  {self._watcher.predictions} reads"
        self._status.config(text=f"{mode}{suffix}", fg=_MUTED)

    def run(self) -> None:
        """Show the overlay and block until it is closed."""
        import tkinter as tk

        self._root = tk.Tk()
        self._root.title("geoguessr")
        self._root.configure(bg=_BG)
        self._root.attributes("-topmost", True)
        self._root.overrideredirect(True)  # no title bar
        self._root.geometry(f"+{self.position[0]}+{self.position[1]}")
        try:
            self._root.attributes("-alpha", 0.92)
        except tk.TclError:
            pass

        frame = tk.Frame(self._root, bg=_BG, padx=14, pady=10)
        frame.pack()

        self._body = tk.Label(
            frame, text=f"press {self.hotkey}", font=("Menlo", 13), bg=_BG,
            fg=_MUTED, justify="left", anchor="w",
        )
        self._body.pack(anchor="w")

        self._status = tk.Label(
            frame, text=self.hotkey, font=("Menlo", 10), bg=_BG, fg=_MUTED,
            justify="left", anchor="w",
        )
        self._status.pack(anchor="w", pady=(8, 0))

        # Escape quits; drag anywhere to move a borderless window.
        self._root.bind("<Escape>", lambda _e: self._root.destroy())
        self._bind_drag()

        listener = self._start_hotkey_listener()
        if self.watch:
            self._start_watcher()

        self._root.after(50, self._drain)
        try:
            self._root.mainloop()
        finally:
            if listener is not None:
                listener.stop()
            if self._watch_stop is not None:
                self._watch_stop.set()

    def _start_watcher(self) -> None:
        """Run change-triggered watching alongside the hotkey.

        The watcher pushes onto the same queue as the hotkey path, so the Tk
        loop stays the single place widgets are touched.
        """
        from .watch import ScreenWatcher

        self._watcher = ScreenWatcher(
            self.predictor,
            config=self.watch_config,
            monitor=self.monitor,
            top_k=self.top_k,
            on_result=lambda guesses, _frame: self._events.put(("result", guesses)),
            on_error=lambda exc: self._events.put(("error", exc)),
        )
        _thread, self._watch_stop = self._watcher.start()
        self._status.config(text="watching screen...", fg=_ACCENT)

    def _bind_drag(self) -> None:
        state = {"x": 0, "y": 0}

        def press(event):
            state["x"], state["y"] = event.x, event.y

        def drag(event):
            x = self._root.winfo_x() + event.x - state["x"]
            y = self._root.winfo_y() + event.y - state["y"]
            self._root.geometry(f"+{x}+{y}")

        self._root.bind("<Button-1>", press)
        self._root.bind("<B1-Motion>", drag)

    def _start_hotkey_listener(self):
        try:
            from pynput import keyboard
        except ImportError:
            self._status.config(text="pynput missing -- hotkey disabled")
            return None

        try:
            listener = keyboard.GlobalHotKeys({self.hotkey: self._on_hotkey})
            listener.start()
            return listener
        except Exception:
            # Usually missing Accessibility permission.
            self._status.config(
                text="hotkey unavailable -- grant Accessibility permission",
                fg="#ff9b9b",
            )
            return None
