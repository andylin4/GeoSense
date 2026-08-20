"""Phase 7: the live prediction path.

Loads a frozen backbone, a trained head, and a fitted temperature, then turns
one image into a ranked list of countries with calibrated percentages. No
training data is touched and nothing is looked up in a database -- this is two
small saved files and some matrix math.

The crop matters more here than anywhere else. Training embeddings are built
from full-frame OSV-5M photos that have no game UI; a live screenshot has a
HUD that must be removed before the encoder sees it. The crop used at
inference is recorded on the predictor and surfaced in ``describe()`` so a
mismatch is visible rather than silent.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..calibrate.temperature import TemperatureScaler
from ..data.countries import display_name
from ..data.crop import GEOGUESSR_16_9, CropSpec
from ..train.head import CountryHead

if TYPE_CHECKING:
    from PIL.Image import Image

    from ..embed.backbone import Backbone

__all__ = ["Guess", "Predictor"]


@dataclass(frozen=True)
class Guess:
    """One ranked candidate country."""

    code: str
    name: str
    probability: float

    def __str__(self) -> str:
        return f"{self.name:<24} {self.probability * 100:5.1f}%"


class Predictor:
    """Backbone + head + calibration, wired for single-image inference."""

    def __init__(
        self,
        backbone: Backbone,
        head: CountryHead,
        *,
        scaler: TemperatureScaler | None = None,
        crop: CropSpec = GEOGUESSR_16_9,
        warn_unverified_crop: bool = True,
    ):
        self.backbone = backbone
        self.head = head
        self.scaler = scaler
        self.crop = crop

        if warn_unverified_crop and (
            "UNVERIFIED" in crop.note or "NOT re-validated against a live" in crop.note
        ):
            warnings.warn(
                f"crop preset {crop.name!r} is not validated against a live "
                "screen capture. Run CropSpec.preview() against a real "
                "screenshot before trusting these percentages.",
                stacklevel=2,
            )

    @classmethod
    def from_artifacts(
        cls,
        artifacts: str | Path = "artifacts",
        *,
        tag: str,
        backbone: Backbone | None = None,
        crop: CropSpec = GEOGUESSR_16_9,
    ) -> Predictor:
        """Load the artifacts a training run wrote, by its ``--limit`` tag."""
        from ..embed.backbone import Backbone as _Backbone

        artifacts = Path(artifacts)
        head_path = artifacts / f"head_{tag}.joblib"
        if not head_path.exists():
            raise FileNotFoundError(
                f"no head at {head_path}. Run scripts/run_pipeline.py --limit {tag}"
            )

        temperature_path = artifacts / f"temperature_{tag}.json"
        scaler = (
            TemperatureScaler.load(temperature_path)
            if temperature_path.exists()
            else None
        )

        return cls(
            backbone or _Backbone(),
            CountryHead.load(head_path),
            scaler=scaler,
            crop=crop,
        )

    def probabilities(self, image: Image) -> np.ndarray:
        """Calibrated probability over every class, in class-list order."""
        prepared = self.crop.apply(image.convert("RGB"))
        embedding = self.backbone.encode_images([prepared])
        probs = self.head.predict_proba(embedding)
        if self.scaler is not None:
            probs = self.scaler.apply(probs)
        return probs[0]

    def predict(self, image: Image, *, top_k: int = 5) -> list[Guess]:
        """Rank countries for one image, most likely first."""
        probs = self.probabilities(image)
        top_k = min(top_k, len(probs))

        order = np.argpartition(-probs, kth=top_k - 1)[:top_k]
        order = order[np.argsort(-probs[order])]

        return [
            Guess(
                code=self.head.class_names[i],
                name=display_name(self.head.class_names[i]),
                probability=float(probs[i]),
            )
            for i in order
        ]

    def describe(self) -> str:
        """What this predictor actually is -- printed at startup."""
        lines = [
            f"backbone     {self.backbone.model_id} (dim {self.backbone.embed_dim})",
            f"head         {type(self.head.model).__name__}, "
            f"{self.head.n_classes} classes",
            f"crop         {self.crop.name} "
            f"(keeps {self.crop.kept_area * 100:.0f}% of frame)",
        ]
        if self.scaler is not None:
            lines.append(f"temperature  {self.scaler.temperature:.4f}")
        else:
            lines.append("temperature  none -- percentages are UNCALIBRATED")
        return "\n".join(lines)


def format_guesses(guesses: list[Guess]) -> str:
    """Render a ranked list, one country per line, most likely first."""
    return "\n".join(str(g) for g in guesses)
