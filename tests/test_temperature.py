"""Tests for Phase 6 temperature scaling."""

from __future__ import annotations

import numpy as np
import pytest

from geoguessr.calibrate.temperature import (
    TemperatureScaler,
    fit_temperature,
    probs_to_logits,
)
from geoguessr.eval.harness import evaluate
from geoguessr.eval.metrics import expected_calibration_error, top_k_accuracy


def synthetic(n=600, n_classes=5, sharpness=1.0, accuracy=0.6, seed=0):
    """Predictions whose confidence can be dialled independently of accuracy."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, size=n)

    logits = rng.normal(0, 1, size=(n, n_classes))
    correct = rng.random(n) < accuracy
    # Push the chosen class to the top: the true one, or a fixed wrong one.
    target = np.where(correct, labels, (labels + 1) % n_classes)
    logits[np.arange(n), target] += 3.0

    scaled = logits * sharpness
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True), labels


class TestProbsToLogits:
    def test_round_trips_through_softmax(self):
        probs, _ = synthetic()
        logits = probs_to_logits(probs)
        recovered = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        assert np.allclose(recovered, probs, atol=1e-9)

    def test_handles_zero_probability(self):
        probs = np.array([[1.0, 0.0]])
        assert np.isfinite(probs_to_logits(probs)).all()


class TestFitTemperature:
    def test_overconfident_model_gets_softened(self):
        probs, labels = synthetic(sharpness=4.0, accuracy=0.5)
        assert fit_temperature(probs, labels) > 1.0

    def test_underconfident_model_gets_sharpened(self):
        probs, labels = synthetic(sharpness=0.25, accuracy=0.95)
        assert fit_temperature(probs, labels) < 1.0

    def test_already_calibrated_stays_near_one(self):
        rng = np.random.default_rng(0)
        logits = rng.normal(0, 1.5, size=(4000, 4))
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        # Draw labels *from* the predicted distribution -> calibrated by
        # construction, so the optimal temperature is 1.
        labels = np.array([rng.choice(4, p=row) for row in probs])
        assert fit_temperature(probs, labels) == pytest.approx(1.0, abs=0.12)

    def test_rejects_shape_mismatch(self):
        probs, labels = synthetic()
        with pytest.raises(ValueError, match="rows but"):
            fit_temperature(probs, labels[:-1])

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            fit_temperature(np.zeros((0, 3)), np.zeros(0, dtype=int))

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="2-D"):
            fit_temperature(np.array([0.5, 0.5]), np.array([0]))


class TestScaler:
    def test_improves_ece(self):
        probs, labels = synthetic(sharpness=4.0, accuracy=0.5)
        scaler = TemperatureScaler.fit(probs, labels)

        assert scaler.ece_after < scaler.ece_before
        assert expected_calibration_error(scaler.apply(probs), labels) == pytest.approx(
            scaler.ece_after
        )

    def test_never_changes_the_prediction(self):
        # Temperature scaling is monotone, so ranking is preserved exactly.
        probs, labels = synthetic(sharpness=3.0)
        scaler = TemperatureScaler.fit(probs, labels)
        calibrated = scaler.apply(probs)

        assert np.array_equal(calibrated.argmax(axis=1), probs.argmax(axis=1))
        assert top_k_accuracy(calibrated, labels) == top_k_accuracy(probs, labels)

    def test_output_is_a_valid_distribution(self):
        probs, labels = synthetic()
        calibrated = TemperatureScaler.fit(probs, labels).apply(probs)
        assert np.allclose(calibrated.sum(axis=1), 1.0)
        assert (calibrated >= 0).all()

    def test_softening_lowers_peak_confidence(self):
        probs, labels = synthetic(sharpness=4.0, accuracy=0.5)
        scaler = TemperatureScaler.fit(probs, labels)
        assert scaler.temperature > 1.0
        assert scaler.apply(probs).max(axis=1).mean() < probs.max(axis=1).mean()

    def test_records_provenance(self):
        probs, labels = synthetic()
        scaler = TemperatureScaler.fit(probs, labels, fit_on="osv5m-val")
        assert scaler.n_fit == len(labels)
        assert scaler.fit_on == "osv5m-val"

    def test_summary_states_direction(self):
        probs, labels = synthetic(sharpness=4.0, accuracy=0.5)
        assert "overconfident" in TemperatureScaler.fit(probs, labels).summary()


class TestWrap:
    def test_wrapped_predict_fn_plugs_into_harness(self):
        probs, labels = synthetic(sharpness=4.0, accuracy=0.5, n=400)
        scaler = TemperatureScaler.fit(probs, labels)

        classes = [f"C{i}" for i in range(probs.shape[1])]
        dataset = list(zip(probs, labels))
        raw_fn = lambda items: np.asarray(items)

        before = evaluate(raw_fn, dataset, classes, progress=False)
        after = evaluate(scaler.wrap(raw_fn), dataset, classes, progress=False)

        assert after.ece < before.ece
        assert after.top1 == pytest.approx(before.top1)  # accuracy untouched


class TestPersistence:
    def test_round_trip(self, tmp_path):
        probs, labels = synthetic()
        scaler = TemperatureScaler.fit(probs, labels, fit_on="osv5m-val")
        loaded = TemperatureScaler.load(scaler.save(tmp_path / "temperature.json"))

        assert loaded.temperature == pytest.approx(scaler.temperature)
        assert loaded.fit_on == "osv5m-val"
        assert np.allclose(loaded.apply(probs), scaler.apply(probs))
