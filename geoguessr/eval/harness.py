"""The evaluation harness.

    evaluate(predict_fn, dataset) -> {top1, top5, ece, per_country, confusion}

This signature is the stable interface of the project. Every phase after the
zero-shot baseline is just a new ``predict_fn`` handed to this same function:
zero-shot CLIP, linear probe, MLP head, MLP + meta fusion. Nothing downstream
of here should ever need to know which one it is holding.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import (
    confusion_matrix,
    expected_calibration_error,
    macro_accuracy,
    per_class_accuracy,
    reliability_bins,
    top_k_accuracy,
)

__all__ = ["EvalResult", "evaluate"]

# A predict_fn takes a batch of images and returns (B, C) class probabilities.
# Images are whatever the dataset yields, typically PIL.Image; the harness never
# inspects them, it only passes them through.
PredictFn = Callable[[list[Any]], np.ndarray]


@dataclass
class EvalResult:
    """Everything one evaluation run produced, in one serializable object."""

    name: str
    n_samples: int
    class_names: list[str]
    top1: float
    top5: float
    macro_top1: float
    ece: float
    per_class_acc: np.ndarray = field(repr=False)
    per_class_support: np.ndarray = field(repr=False)
    confusion: np.ndarray = field(repr=False)
    reliability: dict[str, np.ndarray] = field(repr=False)

    def worst_classes(self, k: int = 10, min_support: int = 10) -> list[tuple[str, float, int]]:
        """Lowest-accuracy classes with enough support to be meaningful.

        This is the collapse detector. If these are all near zero while top1
        looks respectable, the model is riding class imbalance.
        """
        rows = [
            (self.class_names[i], float(self.per_class_acc[i]), int(self.per_class_support[i]))
            for i in range(len(self.class_names))
            if self.per_class_support[i] >= min_support
        ]
        return sorted(rows, key=lambda r: r[1])[:k]

    def top_confusions(self, k: int = 10) -> list[tuple[str, str, int]]:
        """Most frequent (true, predicted) mistakes, as country name pairs."""
        cm = self.confusion.copy()
        np.fill_diagonal(cm, 0)
        if cm.sum() == 0:
            return []
        flat = np.argsort(cm, axis=None)[::-1][:k]
        out = []
        for f in flat:
            t, p = np.unravel_index(f, cm.shape)
            count = int(cm[t, p])
            if count == 0:
                break
            out.append((self.class_names[t], self.class_names[p], count))
        return out

    def summary(self) -> str:
        lines = [
            f"=== {self.name} ===",
            f"samples      {self.n_samples}",
            f"classes      {len(self.class_names)}",
            f"top-1        {self.top1:.4f}",
            f"top-5        {self.top5:.4f}",
            f"macro top-1  {self.macro_top1:.4f}   (mean over classes, catches collapse)",
            f"ECE          {self.ece:.4f}   (0 = calibrated)",
        ]
        worst = self.worst_classes(k=5)
        if worst:
            lines.append("worst classes (support >= 10):")
            lines += [f"    {n:<24} {a:.3f}  (n={s})" for n, a, s in worst]
        confusions = self.top_confusions(k=5)
        if confusions:
            lines.append("top confusions:")
            lines += [f"    {t:<20} -> {p:<20} {c}" for t, p, c in confusions]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view. Arrays become lists so runs can be diffed later."""
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "class_names": self.class_names,
            "top1": self.top1,
            "top5": self.top5,
            "macro_top1": self.macro_top1,
            "ece": self.ece,
            "per_class_acc": _nan_to_none(self.per_class_acc),
            "per_class_support": self.per_class_support.tolist(),
            "confusion": self.confusion.tolist(),
            "reliability": {k: _nan_to_none(v) for k, v in self.reliability.items()},
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False))
        return path


def _nan_to_none(array: np.ndarray) -> list[Any]:
    """JSON has no NaN. Absent classes and empty bins serialize as null."""
    return [None if isinstance(v, float) and np.isnan(v) else v for v in array.tolist()]


def _batched(iterable: Iterable[Any], size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def evaluate(
    predict_fn: PredictFn,
    dataset: Iterable[tuple[Any, int]],
    class_names: Sequence[str],
    *,
    name: str = "unnamed",
    batch_size: int = 32,
    n_bins: int = 15,
    progress: bool = True,
) -> EvalResult:
    """Run ``predict_fn`` over ``dataset`` and compute every headline metric.

    Args:
        predict_fn: ``list[image] -> (B, C)`` array of class probabilities.
            Rows must sum to 1; pass softmax outputs, not logits.
        dataset: iterable of ``(image, label_index)`` pairs. Label indices must
            align with ``class_names``.
        class_names: ordered class list. Index i names column i of the output.
        name: label for this run, used in the summary and saved results.

    Returns:
        An :class:`EvalResult`. Read ``macro_top1`` and ``worst_classes()``, not
        just ``top1`` — a model that collapses onto high-coverage countries
        still posts a respectable aggregate.
    """
    class_names = list(class_names)
    n_classes = len(class_names)
    if n_classes < 2:
        raise ValueError(f"need at least 2 classes, got {n_classes}")

    total = len(dataset) if hasattr(dataset, "__len__") else None
    batches: Iterable[list[tuple[Any, int]]] = _batched(dataset, batch_size)
    if progress:
        try:
            from tqdm.auto import tqdm

            n_batches = None if total is None else (total + batch_size - 1) // batch_size
            batches = tqdm(batches, total=n_batches, desc=name, unit="batch")
        except ImportError:
            pass

    prob_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []

    for batch in batches:
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        probs = np.asarray(predict_fn(images), dtype=np.float64)
        if probs.shape != (len(images), n_classes):
            raise ValueError(
                f"predict_fn returned shape {probs.shape}, expected "
                f"({len(images)}, {n_classes}) for a batch of {len(images)} "
                f"images over {n_classes} classes"
            )

        prob_chunks.append(probs)
        label_chunks.append(np.asarray(labels, dtype=np.int64))

    if not prob_chunks:
        raise ValueError("dataset yielded no samples")

    probs = np.concatenate(prob_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)

    acc, support = per_class_accuracy(probs, labels, n_classes)

    return EvalResult(
        name=name,
        n_samples=int(probs.shape[0]),
        class_names=class_names,
        top1=top_k_accuracy(probs, labels, k=1),
        top5=top_k_accuracy(probs, labels, k=min(5, n_classes)),
        macro_top1=macro_accuracy(probs, labels),
        ece=expected_calibration_error(probs, labels, n_bins=n_bins),
        per_class_acc=acc,
        per_class_support=support,
        confusion=confusion_matrix(probs, labels, n_classes),
        reliability=reliability_bins(probs, labels, n_bins=n_bins),
    )
