"""Known-answer tests for the metric primitives.

Every expected value here is hand-computed, not captured from a previous run.
"""

from __future__ import annotations

import numpy as np
import pytest

from geoguessr.eval.metrics import (
    confusion_matrix,
    expected_calibration_error,
    macro_accuracy,
    per_class_accuracy,
    reliability_bins,
    top_k_accuracy,
)

# preds are [0, 1, 2]; true labels [0, 2, 1] -> only the first is right.
# Every true label is in its row's top-2, though.
PROBS = np.array(
    [
        [0.7, 0.2, 0.1],
        [0.1, 0.6, 0.3],
        [0.2, 0.3, 0.5],
    ]
)
LABELS = np.array([0, 2, 1])


class TestTopK:
    def test_top1(self):
        assert top_k_accuracy(PROBS, LABELS, k=1) == pytest.approx(1 / 3)

    def test_top2_catches_all(self):
        assert top_k_accuracy(PROBS, LABELS, k=2) == pytest.approx(1.0)

    def test_k_equal_to_n_classes_is_always_perfect(self):
        assert top_k_accuracy(PROBS, LABELS, k=3) == pytest.approx(1.0)

    def test_k_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            top_k_accuracy(PROBS, LABELS, k=4)


class TestCalibration:
    def test_perfect_confidence_and_perfect_accuracy_is_zero_ece(self):
        probs = np.array([[1.0, 0.0], [0.0, 1.0]])
        labels = np.array([0, 1])
        assert expected_calibration_error(probs, labels) == pytest.approx(0.0)

    def test_maximally_overconfident_model(self):
        # Claims 100% every time, right half the time -> gap of exactly 0.5.
        probs = np.array([[1.0, 0.0]] * 4)
        labels = np.array([0, 0, 1, 1])
        assert expected_calibration_error(probs, labels) == pytest.approx(0.5)

    def test_bins_partition_every_sample(self):
        rng = np.random.default_rng(0)
        raw = rng.random((200, 8))
        probs = raw / raw.sum(axis=1, keepdims=True)
        labels = rng.integers(0, 8, size=200)

        bins = reliability_bins(probs, labels, n_bins=15)
        assert bins["counts"].sum() == 200
        assert bins["edges"].shape == (16,)

    def test_empty_bins_are_nan_not_zero(self):
        # All confidence is ~1.0, so low-confidence bins must be empty.
        probs = np.array([[1.0, 0.0], [0.0, 1.0]])
        labels = np.array([0, 1])
        bins = reliability_bins(probs, labels, n_bins=10)
        assert np.isnan(bins["accuracy"][0])
        assert bins["counts"][0] == 0

    def test_ece_is_bounded(self):
        rng = np.random.default_rng(1)
        raw = rng.random((500, 12))
        probs = raw / raw.sum(axis=1, keepdims=True)
        labels = rng.integers(0, 12, size=500)
        assert 0.0 <= expected_calibration_error(probs, labels) <= 1.0


class TestPerClassAndCollapse:
    def test_collapse_is_invisible_to_top1_but_obvious_to_macro(self):
        # 8 samples of class 0, 2 of class 1. Model always says class 0 --
        # exactly the failure mode class imbalance produces.
        probs = np.array([[0.9, 0.1]] * 10)
        labels = np.array([0] * 8 + [1] * 2)

        assert top_k_accuracy(probs, labels, k=1) == pytest.approx(0.8)
        assert macro_accuracy(probs, labels) == pytest.approx(0.5)

        acc, support = per_class_accuracy(probs, labels)
        assert acc[0] == pytest.approx(1.0)
        assert acc[1] == pytest.approx(0.0)
        assert support.tolist() == [8, 2]

    def test_absent_class_is_nan_with_zero_support(self):
        probs = np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]])
        labels = np.array([0, 1])  # class 2 never appears
        acc, support = per_class_accuracy(probs, labels)

        assert np.isnan(acc[2])
        assert support[2] == 0

    def test_macro_ignores_absent_classes(self):
        probs = np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1]])
        labels = np.array([0, 1])
        # Both present classes correct; absent class 2 must not drag it down.
        assert macro_accuracy(probs, labels) == pytest.approx(1.0)


class TestConfusion:
    def test_row_sums_equal_support(self):
        cm = confusion_matrix(PROBS, LABELS)
        _, support = per_class_accuracy(PROBS, LABELS)
        assert cm.sum(axis=1).tolist() == support.tolist()

    def test_known_entries(self):
        cm = confusion_matrix(PROBS, LABELS)
        assert cm[0, 0] == 1  # true 0, predicted 0
        assert cm[2, 1] == 1  # true 2, predicted 1
        assert cm[1, 2] == 1  # true 1, predicted 2
        assert cm.sum() == 3

    def test_shape_follows_n_classes_override(self):
        cm = confusion_matrix(PROBS, LABELS, n_classes=5)
        assert cm.shape == (5, 5)


class TestValidation:
    def test_logits_are_rejected_with_a_useful_message(self):
        logits = np.array([[2.0, 1.0, 0.5]])
        with pytest.raises(ValueError, match="softmax"):
            top_k_accuracy(logits, np.array([0]))

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="rows but labels"):
            top_k_accuracy(PROBS, np.array([0, 1]))

    def test_label_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            top_k_accuracy(PROBS, np.array([0, 1, 7]))

    def test_empty_input(self):
        with pytest.raises(ValueError, match="empty"):
            top_k_accuracy(np.zeros((0, 3)), np.zeros(0, dtype=int))
