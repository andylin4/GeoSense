"""Phase 3: the country head.

Consumes the embedding array and never touches an image. Per the design, a
``LogisticRegression`` probe comes first as a pipeline sanity check: if the
linear probe reaches reasonable accuracy the embeddings carry geography and an
MLP will add a few points; if it is near chance, something upstream is broken
and no architecture will rescue it.

Class imbalance is handled here by **reweighting the loss**, not resampling.
Every training image stays in play, and the correction is one uniform
transformation that temperature scaling can still undo cleanly.

The head exposes two adapters, both of which plug into the same
:func:`geoguessr.eval.harness.evaluate`:

* :meth:`CountryHead.predict_fn` -- consumes cached embedding rows. Fast, used
  for OSV-5M-internal validation.
* :meth:`CountryHead.image_predict_fn` -- consumes PIL images by running the
  backbone first. This is the production path and the one used against real
  game screenshots.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from ..data.crop import FULL_FRAME, CropSpec

if TYPE_CHECKING:
    from ..embed.backbone import Backbone

__all__ = [
    "BaseHead",
    "CountryHead",
    "train_linear_probe",
    "split_indices",
    "labels_from_manifest",
]


def labels_from_manifest(
    manifest: pl.DataFrame, class_names: Sequence[str]
) -> np.ndarray:
    """Map the manifest's country codes to integer class indices.

    Deriving this at train time rather than storing it is what makes relabeling
    free: the embedding array never changes when the class scheme does.
    """
    index = {code: i for i, code in enumerate(class_names)}
    codes = manifest["country"].to_list()

    unknown = sorted({c for c in codes if c not in index})
    if unknown:
        raise ValueError(
            f"manifest contains countries outside the class list: {unknown}. "
            "Rebuild the manifest, or pass the matching class_names."
        )
    return np.fromiter((index[c] for c in codes), dtype=np.int64, count=len(codes))


def split_indices(
    labels: np.ndarray, *, val_fraction: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/validation split over row indices.

    Stratified so rare countries appear on both sides; a random split at 100
    classes routinely leaves small classes absent from validation entirely,
    which quietly turns their per-class accuracy into nan.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []

    for cls in np.unique(labels):
        rows = np.flatnonzero(labels == cls)
        rng.shuffle(rows)
        # At least one row stays in train for every class present.
        n_val = min(len(rows) - 1, int(round(len(rows) * val_fraction)))
        val_parts.append(rows[:n_val])
        train_parts.append(rows[n_val:])

    train = np.sort(np.concatenate(train_parts))
    val = np.sort(np.concatenate(val_parts)) if val_parts else np.array([], dtype=int)
    return train, val


class BaseHead:
    """Shared plumbing for anything that maps embeddings to country scores.

    Subclasses implement :meth:`predict_proba`; everything else -- the two
    harness adapters, persistence -- is identical whether the model underneath
    is sklearn or torch. That is what lets a new head type drop into the
    existing harness and serving path unchanged.
    """

    def __init__(self, class_names: Sequence[str], *, meta: dict | None = None):
        self.class_names = list(class_names)
        self.meta = meta or {}

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_fn(self) -> Callable[[list[Any]], np.ndarray]:
        """Adapter for the harness when the dataset yields embedding rows."""

        def fn(items: list[Any]) -> np.ndarray:
            return self.predict_proba(np.asarray(items, dtype=np.float32))

        return fn

    def image_predict_fn(
        self, backbone: Backbone, *, crop: CropSpec = FULL_FRAME
    ) -> Callable[[list[Any]], np.ndarray]:
        """Adapter for the harness when the dataset yields PIL images.

        The crop must match the one used to build the training embeddings, or
        the fingerprints in the embedding cache were pointless.
        """

        def fn(images: list[Any]) -> np.ndarray:
            prepared = [crop.apply(img.convert("RGB")) for img in images]
            return self.predict_proba(backbone.encode_images(prepared))

        return fn


class CountryHead(BaseHead):
    """A trained sklearn classifier over CLIP embeddings, plus its class list."""

    def __init__(self, model: Any, class_names: Sequence[str], *,
                 meta: dict | None = None):
        super().__init__(class_names, meta=meta)
        self.model = model

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """``(N, dim)`` embeddings -> ``(N, n_classes)`` probabilities.

        sklearn only produces columns for classes it saw during training, so
        results are scattered back into the full class space; a country with no
        training rows gets probability 0 rather than shifting every column.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        raw = self.model.predict_proba(embeddings)

        full = np.zeros((embeddings.shape[0], self.n_classes), dtype=np.float64)
        full[:, self.model.classes_] = raw
        return full

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "class_names": self.class_names, "meta": self.meta},
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> CountryHead:
        import joblib

        payload = joblib.load(path)
        return cls(payload["model"], payload["class_names"], meta=payload["meta"])

    def __repr__(self) -> str:
        kind = type(self.model).__name__
        return f"CountryHead({kind}, classes={self.n_classes})"


def train_linear_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    *,
    class_weight: str | dict | None = "balanced",
    max_iter: int = 1000,
    C: float = 1.0,
    seed: int = 0,
    verbose: bool = False,
) -> CountryHead:
    """Fit multinomial logistic regression on cached embeddings.

    ``class_weight="balanced"`` handles imbalance in one argument: it scales
    each class's contribution by inverse frequency, so the US does not drown
    out Slovenia, without discarding or duplicating a single row.
    """
    from sklearn.linear_model import LogisticRegression

    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{embeddings.shape[0]} embeddings but {labels.shape[0]} labels"
        )

    model = LogisticRegression(
        max_iter=max_iter,
        C=C,
        class_weight=class_weight,
        random_state=seed,
        verbose=1 if verbose else 0,
    )
    model.fit(embeddings, labels)

    meta = {
        "kind": "logistic_regression",
        "n_train": int(embeddings.shape[0]),
        "embed_dim": int(embeddings.shape[1]),
        "class_weight": class_weight if isinstance(class_weight, str) else "custom",
        "C": C,
        "classes_seen": int(len(model.classes_)),
    }
    return CountryHead(model, class_names, meta=meta)
