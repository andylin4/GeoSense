"""Phase 5: the meta head.

Predicts how the imagery was captured -- camera generation -- rather than where
it was taken. Structurally this is the country head with a different label
column, which is exactly the payoff of keeping labels out of the embedding
cache: the same vectors serve both tracks.

The interesting part is not the classifier. It is
:func:`diagnose_meta_signal`, which answers **design decision #2**: does CLIP
retain camera-generation information at all?

CLIP is trained to match images to captions, so it has every incentive to
discard exactly the low-level artifacts that identify a camera generation --
sensor resolution, chromatic aberration, stitching seams. If it has, no
classifier on these embeddings can recover them, and the meta head needs its
own small CNN on raw pixels instead. That is an architecture change, so it
should be settled by measurement before Phase 4 imagery is paid for.

The diagnostic compares against a majority-class baseline rather than chance.
Generation labels are heavily imbalanced -- most modern coverage is gen4 -- so
a classifier that always answers "gen4" can look strong while having learned
nothing.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..data.streetview import GENERATIONS
from .head import BaseHead, split_indices, train_linear_probe
from .mlp import train_mlp_head

__all__ = ["train_meta_head", "diagnose_meta_signal", "MetaDiagnosis"]


def train_meta_head(
    embeddings: np.ndarray,
    generation_labels: np.ndarray,
    *,
    generations: Sequence[str] = GENERATIONS,
    kind: str = "linear",
    **kwargs,
) -> BaseHead:
    """Fit a camera-generation classifier on cached embeddings.

    Same machinery as the country head; only the label column differs. Class
    weighting is on by default for the same reason -- generation coverage is
    dominated by gen4.
    """
    if kind == "linear":
        return train_linear_probe(embeddings, generation_labels, generations, **kwargs)
    if kind == "mlp":
        return train_mlp_head(embeddings, generation_labels, generations, **kwargs)
    raise ValueError(f"unknown head kind {kind!r}; expected 'linear' or 'mlp'")


class MetaDiagnosis:
    """Verdict on whether CLIP embeddings carry camera-generation signal."""

    def __init__(
        self,
        accuracy: float,
        macro_accuracy: float,
        majority_accuracy: float,
        n_train: int,
        n_val: int,
        margin_threshold: float,
    ):
        self.accuracy = accuracy
        self.macro_accuracy = macro_accuracy
        self.majority_accuracy = majority_accuracy
        self.n_train = n_train
        self.n_val = n_val
        self.margin_threshold = margin_threshold

    @property
    def margin(self) -> float:
        """How far top-1 beats always-guess-the-most-common-generation."""
        return self.accuracy - self.majority_accuracy

    @property
    def signal_present(self) -> bool:
        """True if the embeddings beat the majority baseline meaningfully.

        Macro accuracy must also clear the majority baseline: a model that
        only ever predicts the dominant generation posts a fine top-1 and a
        terrible macro, and that is not signal.
        """
        return (
            self.margin >= self.margin_threshold
            and self.macro_accuracy > self.majority_accuracy
        )

    def summary(self) -> str:
        verdict = (
            "SIGNAL PRESENT -- CLIP embeddings retain camera generation; "
            "the meta head can share the embedding cache."
            if self.signal_present
            else "NO USABLE SIGNAL -- CLIP appears to discard camera "
                 "generation. The meta head needs its own CNN on raw pixels "
                 "(design decision #2)."
        )
        return "\n".join(
            [
                f"train/val        {self.n_train}/{self.n_val}",
                f"majority baseline {self.majority_accuracy:.4f}",
                f"top-1             {self.accuracy:.4f}  "
                f"(margin {self.margin:+.4f})",
                f"macro top-1       {self.macro_accuracy:.4f}",
                verdict,
            ]
        )


def diagnose_meta_signal(
    embeddings: np.ndarray,
    generation_labels: np.ndarray,
    *,
    generations: Sequence[str] = GENERATIONS,
    val_fraction: float = 0.25,
    margin_threshold: float = 0.10,
    seed: int = 0,
) -> MetaDiagnosis:
    """Settle design decision #2 with a cheap measurement.

    Fits a linear probe for camera generation and compares it against always
    predicting the most common generation. A linear probe is the right test:
    if the information is linearly accessible the architecture question is
    closed, and if a linear probe finds nothing an MLP on the same vectors is
    unlikely to rescue it.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(generation_labels, dtype=np.int64)

    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError(f"{embeddings.shape[0]} embeddings but {labels.shape[0]} labels")
    if len(np.unique(labels)) < 2:
        raise ValueError(
            "need at least two distinct generations to diagnose signal; "
            "the scrape produced only one"
        )

    train_idx, val_idx = split_indices(labels, val_fraction=val_fraction, seed=seed)
    if len(val_idx) == 0:
        raise ValueError("validation split is empty; collect more data")

    head = train_linear_probe(
        embeddings[train_idx], labels[train_idx], generations, seed=seed
    )
    predictions = head.predict_proba(embeddings[val_idx]).argmax(axis=1)
    truth = labels[val_idx]

    accuracy = float((predictions == truth).mean())

    # Majority baseline is defined by the training split -- using the
    # validation split would leak.
    counts = np.bincount(labels[train_idx], minlength=len(generations))
    majority = int(counts.argmax())
    majority_accuracy = float((truth == majority).mean())

    per_class = [
        float((predictions[truth == cls] == cls).mean())
        for cls in np.unique(truth)
    ]

    return MetaDiagnosis(
        accuracy=accuracy,
        macro_accuracy=float(np.mean(per_class)),
        majority_accuracy=majority_accuracy,
        n_train=len(train_idx),
        n_val=len(val_idx),
        margin_threshold=margin_threshold,
    )
