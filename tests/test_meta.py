"""Tests for the meta head and the decision-#2 diagnostic.

The diagnostic is the thing worth testing hard: it decides an architecture
question, so it must not report signal where there is none, and must not miss
signal that exists.
"""

from __future__ import annotations

import numpy as np
import pytest

from geoguessr.data.streetview import GENERATIONS
from geoguessr.train.meta import diagnose_meta_signal, train_meta_head

GENS = ["gen2", "gen3", "gen4"]


def separable(n_per_class=60, dim=6, noise=0.4, seed=0):
    """Embeddings that genuinely encode generation."""
    rng = np.random.default_rng(seed)
    centers = np.eye(3, dim) * 4.0
    xs, ys = [], []
    for cls in range(3):
        xs.append(rng.normal(centers[cls], noise, size=(n_per_class, dim)))
        ys.append(np.full(n_per_class, cls))
    return np.vstack(xs).astype(np.float32), np.concatenate(ys)


def uninformative(n=200, dim=6, seed=0):
    """Embeddings with no relationship to generation at all."""
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 1, size=(n, dim)).astype(np.float32),
            rng.integers(0, 3, size=n))


class TestTrainMetaHead:
    def test_linear_head_learns_generation(self):
        x, y = separable()
        head = train_meta_head(x, y, generations=GENS)
        assert (head.predict_proba(x).argmax(axis=1) == y).mean() > 0.95

    def test_mlp_variant(self):
        x, y = separable()
        head = train_meta_head(x, y, generations=GENS, kind="mlp", epochs=30)
        assert (head.predict_proba(x).argmax(axis=1) == y).mean() > 0.9

    def test_rejects_unknown_kind(self):
        x, y = separable()
        with pytest.raises(ValueError, match="unknown head kind"):
            train_meta_head(x, y, generations=GENS, kind="transformer")

    def test_defaults_to_the_project_generation_list(self):
        x, y = separable()
        head = train_meta_head(x, y)
        assert head.class_names == list(GENERATIONS)

    def test_shares_the_harness_interface(self):
        from geoguessr.eval.harness import evaluate

        x, y = separable()
        head = train_meta_head(x, y, generations=GENS)
        result = evaluate(head.predict_fn(), list(zip(x, y)), GENS, progress=False)
        assert result.top1 > 0.95


class TestDiagnosis:
    def test_detects_real_signal(self):
        x, y = separable()
        diagnosis = diagnose_meta_signal(x, y, generations=GENS)

        assert diagnosis.signal_present
        assert diagnosis.margin > 0.1
        assert "SIGNAL PRESENT" in diagnosis.summary()

    def test_reports_no_signal_on_random_embeddings(self):
        x, y = uninformative()
        diagnosis = diagnose_meta_signal(x, y, generations=GENS)

        assert not diagnosis.signal_present
        assert "NO USABLE SIGNAL" in diagnosis.summary()

    def test_majority_collapse_is_not_counted_as_signal(self):
        # 90% gen4 with embeddings carrying nothing: a classifier can score
        # ~0.9 top-1 by always answering gen4. That must not read as signal.
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, size=(300, 6)).astype(np.float32)
        y = np.array([2] * 270 + [0] * 15 + [1] * 15)

        diagnosis = diagnose_meta_signal(x, y, generations=GENS)
        assert diagnosis.majority_accuracy > 0.8
        assert not diagnosis.signal_present

    def test_majority_baseline_comes_from_training_split(self):
        # Using the val split to pick the majority class would leak.
        x, y = separable()
        diagnosis = diagnose_meta_signal(x, y, generations=GENS)
        assert 0.0 <= diagnosis.majority_accuracy <= 1.0

    def test_margin_threshold_is_configurable(self):
        x, y = separable(noise=2.5, seed=3)  # weak but nonzero signal
        lenient = diagnose_meta_signal(x, y, generations=GENS, margin_threshold=0.01)
        strict = diagnose_meta_signal(x, y, generations=GENS, margin_threshold=0.95)

        assert lenient.signal_present
        assert not strict.signal_present

    def test_requires_two_generations(self):
        x = np.random.default_rng(0).normal(size=(50, 6)).astype(np.float32)
        y = np.zeros(50, dtype=np.int64)

        with pytest.raises(ValueError, match="at least two distinct generations"):
            diagnose_meta_signal(x, y, generations=GENS)

    def test_rejects_length_mismatch(self):
        x, y = separable()
        with pytest.raises(ValueError, match="but .* labels"):
            diagnose_meta_signal(x, y[:-1], generations=GENS)

    def test_summary_reports_the_numbers(self):
        x, y = separable()
        text = diagnose_meta_signal(x, y, generations=GENS).summary()

        assert "majority baseline" in text
        assert "macro top-1" in text
        assert "top-1" in text
