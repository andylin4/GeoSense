"""Screen crop specification.

Design decision #6: cropping is an inference-side concern only. OSV-5M training
images are full-frame street photos with no game UI and are used uncultivated
by this module. What gets cropped is (a) live screen captures and (b)
GeoGuessr-50k at evaluation time -- and those two must use the *same* rule, or
the eval number describes a distribution the live tool never sees.

Regions are stored as fractions of image size, not pixels, so one spec applies
to any resolution. The capture surface is still expected to be a fullscreen
browser at a known aspect ratio (decision #6); fractions make the spec robust
to display scaling, not a substitute for that constraint.

WARNING: ``GEOGUESSR_16_9`` below is a starting estimate of where the HUD sits.
It must be checked against a real screenshot before any number produced with it
is trusted -- use ``preview()`` to render the crop and look at it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image

__all__ = ["CropSpec", "FULL_FRAME", "GEOGUESSR_16_9", "CROP_PRESETS", "get_preset"]


@dataclass(frozen=True)
class CropSpec:
    """A crop rectangle in fractional coordinates, origin at top-left.

    ``left=0.0, top=0.0, right=1.0, bottom=1.0`` is the identity crop.
    """

    name: str
    left: float = 0.0
    top: float = 0.0
    right: float = 1.0
    bottom: float = 1.0
    note: str = ""

    def __post_init__(self) -> None:
        for field_name in ("left", "top", "right", "bottom"):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{field_name}={value} must be a fraction in [0, 1]"
                )
        if self.left >= self.right:
            raise ValueError(f"left ({self.left}) must be < right ({self.right})")
        if self.top >= self.bottom:
            raise ValueError(f"top ({self.top}) must be < bottom ({self.bottom})")

    @property
    def is_identity(self) -> bool:
        return (self.left, self.top, self.right, self.bottom) == (0.0, 0.0, 1.0, 1.0)

    @property
    def kept_area(self) -> float:
        """Fraction of the original frame this crop retains."""
        return (self.right - self.left) * (self.bottom - self.top)

    def box(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Pixel box ``(left, top, right, bottom)`` for a given frame size."""
        return (
            int(round(self.left * width)),
            int(round(self.top * height)),
            int(round(self.right * width)),
            int(round(self.bottom * height)),
        )

    def apply(self, image: Image) -> Image:
        """Crop a PIL image. The identity crop returns the image untouched."""
        if self.is_identity:
            return image
        return image.crop(self.box(image.width, image.height))

    def preview(self, image: Image, outline: str = "red", width: int = 6) -> Image:
        """Draw the crop rectangle on a copy of the image, for eyeballing it.

        This is how ``GEOGUESSR_16_9`` gets validated: capture one real
        screenshot, run it through here, and confirm no HUD survives and no
        scene is needlessly discarded.
        """
        from PIL import ImageDraw

        annotated = image.convert("RGB").copy()
        ImageDraw.Draw(annotated).rectangle(
            self.box(image.width, image.height), outline=outline, width=width
        )
        return annotated

    def fingerprint(self) -> str:
        """Stable string identifying this crop, for tagging embedding caches."""
        return (
            f"{self.name}:{self.left:.4f},{self.top:.4f},"
            f"{self.right:.4f},{self.bottom:.4f}"
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, payload: str) -> CropSpec:
        return cls(**json.loads(payload))


FULL_FRAME = CropSpec(
    name="full_frame",
    note="Identity. Used for OSV-5M training images, which carry no game UI.",
)

GEOGUESSR_16_9 = CropSpec(
    name="geoguessr_16_9",
    top=0.06,
    bottom=0.78,
    note=(
        "UNVERIFIED ESTIMATE. Cuts the top status/score strip and the bottom "
        "band holding the minimap, compass, and guess button. Full width is "
        "kept because the HUD spans both bottom corners, so trimming sides "
        "would discard scene without removing UI. Validate with preview() "
        "against a real fullscreen 16:9 screenshot before trusting results."
    ),
)

CROP_PRESETS: dict[str, CropSpec] = {
    spec.name: spec for spec in (FULL_FRAME, GEOGUESSR_16_9)
}


def get_preset(name: str) -> CropSpec:
    if name not in CROP_PRESETS:
        raise KeyError(f"unknown crop preset {name!r}; have {sorted(CROP_PRESETS)}")
    return CROP_PRESETS[name]
