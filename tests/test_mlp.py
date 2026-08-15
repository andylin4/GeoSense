"""Tests for the MLP country head."""

from __future__ import annotations

import numpy as np
import pytest

from geoguessr.eval.harness import evaluate
from geoguessr.train.mlp import MLPHead, balanced_class_weights, train_mlp_head

CLASSES = ["FR", "JP", "BR"]


def blobs(n_per_class=60, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.eye(3, dim) * 4.0
    xs, ys = [], []
    for cls in range(3):
        xs.append(rng.normal(centers[cls], 0.5, size=(n_per_class, dim)))
        ys.append(np.full(n_per_class, cls))
    return np.vstack(xs).astype(np.float32), np.concatenate(ys)


class TestBalancedWeights:
    def test_inverse_frequency(self):
        labels = np.array([0] * 8 + [1] * 2)
        weights = balanced_class_weights(labels, 2)
        # Rarer class gets the larger weight, in proportion to the imbalance.
        assert weights[1] > weights[0]
        assert weights[1] / weights[0] == pytest.approx(4.0)

    def test_matches_sklearn_formula(self):
        labels = np.array([0] * 6 + [1] * 3 + [2] * 1)
        weights = balanced_class_weights(labels, 3)
        assert weights[0] == pytest.approx(10 / (3 * 6))
        assert weights[2] == pytest.approx(10 / (3 * 1))

    def test_absent_class_gets_zero_not_infinity(self):
        weights = balanced_class_weights(np.array([0, 0, 1]), 4)
        assert weights[2] == 0.0
        assert np.isfinite(weights).all()


class TestTraining:
    def test_learns_separable_data(self):
        x, y = blobs()
        head = train_mlp_head(x, y, CLASSES, epochs=40)
        assert (head.predict_proba(x).argmax(axis=1) == y).mean() > 0.95

    def test_output_is_a_valid_distribution(self):
        x, y = blobs()
        probs = train_mlp_head(x, y, CLASSES, epochs=5).predict_proba(x)
        assert probs.shape == (len(y), 3)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_handles_single_row(self):
        x, y = blobs()
        head = train_mlp_head(x, y, CLASSES, epochs=5)
        assert head.predict_proba(x[0]).shape == (1, 3)

    def test_early_stopping_keeps_best_val_weights(self):
        x, y = blobs()
        head = train_mlp_head(x[:120], y[:120], CLASSES,
                              val_embeddings=x[120:], val_labels=y[120:],
                              epochs=100, patience=3)
        assert head.meta["best_epoch"] <= head.meta["epochs_run"]
        assert head.meta["best_val_loss"] is not None

    def test_stops_before_max_epochs_when_stale(self):
        x, y = blobs()
        head = train_mlp_head(x[:120], y[:120], CLASSES,
                              val_embeddings=x[120:], val_labels=y[120:],
                              epochs=500, patience=2)
        assert head.meta["epochs_run"] < 500

    def test_reproducible(self):
        x, y = blobs()
        a = train_mlp_head(x, y, CLASSES, epochs=10, seed=5).predict_proba(x)
        b = train_mlp_head(x, y, CLASSES, epochs=10, seed=5).predict_proba(x)
        assert np.allclose(a, b)

    def test_records_metadata(self):
        x, y = blobs()
        head = train_mlp_head(x, y, CLASSES, hidden=(32, 16), epochs=3)
        assert head.meta["kind"] == "mlp"
        assert head.meta["hidden"] == [32, 16]
        assert head.meta["embed_dim"] == 8

    def test_rejects_length_mismatch(self):
        x, y = blobs()
        with pytest.raises(ValueError, match="but .* labels"):
            train_mlp_head(x, y[:-1], CLASSES)

    def test_class_weighting_helps_the_minority(self):
        rng = np.random.default_rng(0)
        major = rng.normal([0.0, 0.0], 1.0, size=(300, 2))
        minor = rng.normal([1.5, 0.0], 1.0, size=(20, 2))
        x = np.vstack([major, minor]).astype(np.float32)
        y = np.array([0] * 300 + [20 // 20] * 20)

        weighted = train_mlp_head(x, y, ["A", "B"], epochs=40, balanced=True)
        plain = train_mlp_head(x, y, ["A", "B"], epochs=40, balanced=False)

        recall = lambda h: (h.predict_proba(x[300:]).argmax(1) == 1).mean()
        assert recall(weighted) >= recall(plain)


class TestInterfaceCompatibility:
    def test_plugs_into_the_same_harness_as_the_linear_probe(self):
        x, y = blobs()
        head = train_mlp_head(x, y, CLASSES, epochs=30)

        result = evaluate(head.predict_fn(), list(zip(x, y)), CLASSES,
                          name="mlp", progress=False)
        assert result.top1 > 0.95

    def test_image_predict_fn_is_inherited(self):
        from PIL import Image

        x, y = blobs(dim=8)
        head = train_mlp_head(x, y, CLASSES, epochs=20)

        class StubBackbone:
            def encode_images(self, images, normalize=True):
                out = np.zeros((len(images), 8), dtype=np.float32)
                for i, img in enumerate(images):
                    out[i, img.getpixel((0, 0))[0] // 100] = 4.0
                return out

        images = [Image.new("RGB", (8, 8), (c * 100, 0, 0)) for c in range(3)]
        probs = head.image_predict_fn(StubBackbone())(images)
        assert probs.argmax(axis=1).tolist() == [0, 1, 2]

    def test_works_with_the_predictor(self):
        from PIL import Image

        from geoguessr.data.crop import FULL_FRAME
        from geoguessr.serve.predictor import Predictor

        x, y = blobs(dim=8)
        head = train_mlp_head(x, y, CLASSES, epochs=20)

        class StubBackbone:
            model_id = "stub"
            embed_dim = 8

            def encode_images(self, images, normalize=True):
                out = np.zeros((len(images), 8), dtype=np.float32)
                for i, img in enumerate(images):
                    out[i, img.getpixel((0, 0))[0] // 100] = 4.0
                return out

        predictor = Predictor(StubBackbone(), head, crop=FULL_FRAME)
        guess = predictor.predict(Image.new("RGB", (32, 32), (100, 0, 0)))[0]
        assert guess.code == "JP"


class TestPersistence:
    def test_round_trip(self, tmp_path):
        x, y = blobs()
        head = train_mlp_head(x, y, CLASSES, hidden=(24,), epochs=10)
        loaded = MLPHead.load(head.save(tmp_path / "mlp.pt"))

        assert loaded.class_names == CLASSES
        assert loaded.meta["hidden"] == [24]
        assert np.allclose(loaded.predict_proba(x), head.predict_proba(x))

    def test_repr(self):
        x, y = blobs()
        head = train_mlp_head(x, y, CLASSES, hidden=(24,), epochs=2)
        assert "MLPHead" in repr(head)
        assert "classes=3" in repr(head)
