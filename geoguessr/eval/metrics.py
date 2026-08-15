"""Metric primitives for the evaluation harness.

Pure numpy. No model, no dataset, no torch. Everything here operates on a
probability matrix and an integer label vector, which makes it cheap to test
and impossible to break by changing anything upstream.

Convention used throughout:
    probs   (N, C) float array, each row a probability distribution over classes
    labels  (N,)   int array of true class indices in [0, C)
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "top_k_accuracy",
    "expected_calibration_error",
    "reliability_bins",
    "per_class_accuracy",
    "macro_accuracy",
    "confusion_matrix",
]


def _validate(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if probs.ndim != 2:
        raise ValueError(f"probs must be 2-D (N, C), got shape {probs.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-D (N,), got shape {labels.shape}")
    if probs.shape[0] != labels.shape[0]:
        raise ValueError(
            f"probs has {probs.shape[0]} rows but labels has {labels.shape[0]} entries"
        )
    if probs.shape[0] == 0:
        raise ValueError("cannot compute metrics over an empty set")
    if labels.min() < 0 or labels.max() >= probs.shape[1]:
        raise ValueError(
            f"labels out of range for {probs.shape[1]} classes: "
            f"[{labels.min()}, {labels.max()}]"
        )

    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        worst = int(np.argmax(np.abs(row_sums - 1.0)))
        raise ValueError(
            "probs rows must be probability distributions summing to 1; "
            f"row {worst} sums to {row_sums[worst]:.6f}. "
            "Pass softmax outputs, not raw logits."
        )
    return probs, labels


def top_k_accuracy(probs: np.ndarray, labels: np.ndarray, k: int = 1) -> float:
    """Fraction of samples whose true class is among the k highest-scoring."""
    probs, labels = _validate(probs, labels)
    n_classes = probs.shape[1]
    if not 1 <= k <= n_classes:
        raise ValueError(f"k={k} out of range for {n_classes} classes")

    # argpartition is O(C) per row vs O(C log C) for a full argsort.
    topk = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    return float((topk == labels[:, None]).any(axis=1).mean())


def reliability_bins(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> dict[str, np.ndarray]:
    """Bin predictions by confidence and report accuracy within each bin.

    This is the raw material for both ECE and a reliability diagram. A
    well-calibrated model has ``accuracy == confidence`` in every bin.
    """
    probs, labels = _validate(probs, labels)
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == labels).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Bin i covers (edges[i], edges[i+1]]; clip pulls confidence==0 into bin 0.
    idx = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)

    counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
    acc_sum = np.bincount(idx, weights=correct, minlength=n_bins)
    conf_sum = np.bincount(idx, weights=confidence, minlength=n_bins)

    with np.errstate(invalid="ignore", divide="ignore"):
        bin_accuracy = np.where(counts > 0, acc_sum / counts, np.nan)
        bin_confidence = np.where(counts > 0, conf_sum / counts, np.nan)

    return {
        "edges": edges,
        "counts": counts,
        "accuracy": bin_accuracy,
        "confidence": bin_confidence,
    }


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Weighted mean gap between confidence and accuracy across bins.

    ECE = sum_b (n_b / N) * |acc(b) - conf(b)|

    0 is perfect calibration. An untreated softmax typically lands well above
    0.1, which is the entire reason temperature scaling exists in this project.
    """
    bins = reliability_bins(probs, labels, n_bins=n_bins)
    counts = bins["counts"]
    total = counts.sum()
    populated = counts > 0

    gaps = np.abs(bins["accuracy"][populated] - bins["confidence"][populated])
    return float((counts[populated] * gaps).sum() / total)


def per_class_accuracy(
    probs: np.ndarray, labels: np.ndarray, n_classes: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Top-1 recall for each class, plus that class's support.

    Returns ``(accuracy, support)``, both length ``n_classes``. Classes with no
    samples in the eval set get ``nan`` accuracy and 0 support rather than being
    silently counted as 0% correct.
    """
    probs, labels = _validate(probs, labels)
    n_classes = probs.shape[1] if n_classes is None else n_classes

    predicted = probs.argmax(axis=1)
    correct = (predicted == labels).astype(np.float64)

    support = np.bincount(labels, minlength=n_classes).astype(np.int64)
    hits = np.bincount(labels, weights=correct, minlength=n_classes)

    with np.errstate(invalid="ignore", divide="ignore"):
        accuracy = np.where(support > 0, hits / np.maximum(support, 1), np.nan)
    return accuracy, support


def macro_accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    """Unweighted mean of per-class accuracy over classes present in the set.

    This is the number that catches a model collapsing onto high-coverage
    countries. A model that predicts "United States" for everything scores well
    on plain top-1 (the US is ~1/5 of GeoGuessr-50k) and near zero here.
    """
    accuracy, support = per_class_accuracy(probs, labels)
    present = support > 0
    return float(accuracy[present].mean())


def confusion_matrix(
    probs: np.ndarray, labels: np.ndarray, n_classes: int | None = None
) -> np.ndarray:
    """Counts of (true, predicted) pairs as an ``(n_classes, n_classes)`` array.

    Row = true class, column = predicted class. Row sums equal class support.
    """
    probs, labels = _validate(probs, labels)
    n_classes = probs.shape[1] if n_classes is None else n_classes

    predicted = probs.argmax(axis=1)
    flat = labels * n_classes + predicted
    return np.bincount(flat, minlength=n_classes**2).reshape(n_classes, n_classes)
