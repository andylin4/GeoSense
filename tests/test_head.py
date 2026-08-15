"""Tests for the Phase 3 country head."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from geoguessr.eval.harness import evaluate
from geoguessr.train.head import (
    CountryHead,
    labels_from_manifest,
    split_indices,
    train_linear_probe,
)

CLASSES = ["FR", "JP", "BR"]


def separable_data(n_per_class=40, dim=6, seed=0):
    """Three well-separated Gaussian blobs, one per class."""
    rng = np.random.default_rng(seed)
    centers = np.eye(3, dim) * 5.0
    embeddings, labels = [], []
    for cls in range(3):
        embeddings.append(rng.normal(centers[cls], 0.4, size=(n_per_class, dim)))
        labels.append(np.full(n_per_class, cls))
    return np.vstack(embeddings).astype(np.float32), np.concatenate(labels)


class TestLabelsFromManifest:
    def test_maps_codes_to_indices(self):
        manifest = pl.DataFrame({"country": ["JP", "FR", "BR", "FR"]})
        assert labels_from_manifest(manifest, CLASSES).tolist() == [1, 0, 2, 0]

    def test_rejects_country_outside_class_list(self):
        manifest = pl.DataFrame({"country": ["FR", "ZZ"]})
        with pytest.raises(ValueError, match="outside the class list"):
            labels_from_manifest(manifest, CLASSES)

    def test_relabeling_is_just_a_different_class_list(self):
        # The embedding array never changes when the scheme does.
        manifest = pl.DataFrame({"country": ["FR", "JP"]})
        assert labels_from_manifest(manifest, ["JP", "FR"]).tolist() == [1, 0]


class TestSplit:
    def test_partitions_without_overlap(self):
        _, labels = separable_data()
        train, val = split_indices(labels, val_fraction=0.25, seed=0)

        assert len(set(train) & set(val)) == 0
        assert len(train) + len(val) == len(labels)

    def test_is_stratified(self):
        _, labels = separable_data(n_per_class=40)
        _, val = split_indices(labels, val_fraction=0.25, seed=0)
        counts = np.bincount(labels[val], minlength=3)
        assert counts.tolist() == [10, 10, 10]

    def test_rare_class_still_appears_in_train(self):
        # A class with a single row must not be moved entirely into val.
        labels = np.array([0] * 20 + [1])
        train, val = split_indices(labels, val_fraction=0.5, seed=0)
        assert 1 in labels[train]
        assert len(val) < len(labels)

    def test_reproducible(self):
        _, labels = separable_data()
        a = split_indices(labels, seed=3)[1]
        b = split_indices(labels, seed=3)[1]
        assert a.tolist() == b.tolist()

    def test_rejects_bad_fraction(self):
        _, labels = separable_data()
        for bad in (0.0, 1.0, -0.2):
            with pytest.raises(ValueError, match="val_fraction"):
                split_indices(labels, val_fraction=bad)


class TestLinearProbe:
    def test_learns_separable_data(self):
        embeddings, labels = separable_data()
        head = train_linear_probe(embeddings, labels, CLASSES)

        probs = head.predict_proba(embeddings)
        assert (probs.argmax(axis=1) == labels).mean() > 0.95

    def test_output_is_a_valid_distribution(self):
        embeddings, labels = separable_data()
        head = train_linear_probe(embeddings, labels, CLASSES)
        probs = head.predict_proba(embeddings)

        assert probs.shape == (len(labels), 3)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_columns_cover_full_class_space_even_if_a_class_is_absent(self):
        # Only classes 0 and 1 present; column 2 must exist and be zero.
        embeddings, labels = separable_data()
        keep = labels < 2
        head = train_linear_probe(embeddings[keep], labels[keep], CLASSES)

        probs = head.predict_proba(embeddings[keep])
        assert probs.shape[1] == 3
        assert np.all(probs[:, 2] == 0.0)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_class_weighting_helps_the_minority_class(self):
        # 200 of class 0, 12 of class 1 -- overlapping enough to be confusable.
        rng = np.random.default_rng(0)
        major = rng.normal([0.0, 0.0], 1.0, size=(200, 2))
        minor = rng.normal([1.6, 0.0], 1.0, size=(12, 2))
        embeddings = np.vstack([major, minor]).astype(np.float32)
        labels = np.array([0] * 200 + [1] * 12)

        weighted = train_linear_probe(embeddings, labels, ["A", "B"],
                                      class_weight="balanced")
        unweighted = train_linear_probe(embeddings, labels, ["A", "B"],
                                        class_weight=None)

        recall = lambda h: (h.predict_proba(embeddings[200:]).argmax(1) == 1).mean()
        assert recall(weighted) > recall(unweighted)

    def test_records_metadata(self):
        embeddings, labels = separable_data()
        head = train_linear_probe(embeddings, labels, CLASSES)

        assert head.meta["kind"] == "logistic_regression"
        assert head.meta["embed_dim"] == 6
        assert head.meta["class_weight"] == "balanced"

    def test_rejects_length_mismatch(self):
        embeddings, labels = separable_data()
        with pytest.raises(ValueError, match="but .* labels"):
            train_linear_probe(embeddings, labels[:-1], CLASSES)


class TestHarnessIntegration:
    def test_embedding_predict_fn_plugs_into_harness(self):
        embeddings, labels = separable_data()
        head = train_linear_probe(embeddings, labels, CLASSES)

        dataset = list(zip(embeddings, labels))
        result = evaluate(head.predict_fn(), dataset, CLASSES, name="probe",
                          progress=False)

        assert result.top1 > 0.95
        assert result.macro_top1 > 0.95
        assert result.n_samples == len(labels)

    def test_image_predict_fn_runs_backbone_then_head(self):
        from PIL import Image

        embeddings, labels = separable_data(dim=6)
        head = train_linear_probe(embeddings, labels, CLASSES)

        class StubBackbone:
            def encode_images(self, images, normalize=True):
                # Each image's red channel selects which blob center to return.
                out = np.zeros((len(images), 6), dtype=np.float32)
                for i, img in enumerate(images):
                    out[i, img.getpixel((0, 0))[0] // 100] = 5.0
                return out

        images = [Image.new("RGB", (8, 8), (c * 100, 0, 0)) for c in range(3)]
        probs = head.image_predict_fn(StubBackbone())(images)
        assert probs.argmax(axis=1).tolist() == [0, 1, 2]


class TestPersistence:
    def test_round_trip(self, tmp_path):
        embeddings, labels = separable_data()
        head = train_linear_probe(embeddings, labels, CLASSES)
        path = head.save(tmp_path / "head.joblib")

        loaded = CountryHead.load(path)
        assert loaded.class_names == CLASSES
        assert loaded.meta == head.meta
        assert np.allclose(loaded.predict_proba(embeddings),
                           head.predict_proba(embeddings))

    def test_repr(self):
        embeddings, labels = separable_data()
        head = train_linear_probe(embeddings, labels, CLASSES)
        assert "LogisticRegression" in repr(head)
        assert "classes=3" in repr(head)
