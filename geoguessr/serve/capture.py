"""Screen capture for the live path.

``mss`` grabs the primary display in well under 10ms, which is why the design
picked it over ``ImageGrab``. The grab is deliberately separated from the crop:
capture returns the raw frame, and the caller applies the same
:class:`~geoguessr.data.crop.CropSpec` used everywhere else, so there is
exactly one definition of "what the model sees".

macOS requires **Screen Recording** permission (System Settings -> Privacy &
Security -> Screen Recording) for the terminal or IDE running this. Without it
macOS silently returns a black or desktop-only image rather than raising, so
:func:`grab_screen` checks for that and says so.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image

__all__ = ["grab_screen", "list_monitors", "save_screenshot"]


def list_monitors() -> list[dict]:
    """All displays mss can see. Index 0 is the union of every monitor."""
    import mss

    with mss.mss() as sct:
        return [dict(m) for m in sct.monitors]


def grab_screen(monitor: int = 1, *, check_permission: bool = True) -> Image:
    """Capture one display as a PIL image.

    Args:
        monitor: 1 is the primary display. 0 would be all monitors stitched
            together, which is almost never what you want.
        check_permission: raise a useful error if the frame comes back uniform,
            which on macOS means Screen Recording permission was denied.
    """
    import mss
    from PIL import Image as PILImage

    with mss.mss() as sct:
        if monitor >= len(sct.monitors):
            raise ValueError(
                f"monitor {monitor} does not exist; "
                f"{len(sct.monitors) - 1} display(s) available"
            )
        shot = sct.grab(sct.monitors[monitor])

    image = PILImage.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

    if check_permission:
        extrema = image.convert("L").getextrema()
        if extrema[0] == extrema[1]:
            raise PermissionError(
                "captured a uniform image, which on macOS means Screen Recording "
                "permission is missing. Grant it in System Settings -> Privacy & "
                "Security -> Screen Recording for this terminal, then restart the "
                "terminal (the permission is not picked up until relaunch)."
            )
    return image


def save_screenshot(path: str, *, monitor: int = 1) -> str:
    """Grab and save a raw screenshot -- the fastest way to start an eval set.

    Every saved frame plus its true country is one row of a self-captured
    evaluation set, which is the workaround while GeoGuessr-50k is unreachable.
    """
    from pathlib import Path

    image = grab_screen(monitor=monitor)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return str(destination)
