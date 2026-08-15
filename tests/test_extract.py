"""Tests for Phase 2 embedding extraction.

A stub backbone stands in for StreetCLIP so these run in milliseconds. What is
under test is cache bookkeeping -- resume, invalidation, failure handling --
which is where this step can silently corrupt everything downstream.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
from PIL import Image

from geoguessr.data.crop import FULL_FRAME, CropSpec
from geoguessr.embed.backbone import Fingerprint
from geoguessr.embed.extract import (
    extract_embeddings,
    load_embeddings,
    manifest_signature,
    meta_path_for,
)

DIM = 8


class StubBackbone:
    """Encodes an image to a vector determined by its top-left pixel."""

    def __init__(self, model_id="stub/model", dim=DIM):
        self.model_id = model_id
        self._dim = dim
        self.calls = 0

    @property
    def embed_dim(self):
        return self._dim

    @property
    def fingerprint(self):
        return Fingerprint(self.model_id, self._dim, 224)

    def encode_images(self, images, normalize=True):
        self.calls += 1
        out = np.zeros((len(images), self._dim), dtype=np.float32)
        for i, img in enumerate(images):
            out[i, :] = img.getpixel((0, 0))[0] / 255.0
        return out


@pytest.fixture
def dataset(tmp_path):
    """8 images with distinct pixel values, plus a matching manifest."""
    images = tmp_path / "img"
    images.mkdir()
    rows = []
    for i in range(8):
        path = images / f"{i}.jpg"
        Image.new("RGB", (64, 48), (i * 30, 0, 0)).save(path)
        rows.append((i, 0.0, 0.0, "FR", "train", str(path)))

    manifest = pl.DataFrame(
        rows,
        schema={"image_id": pl.Int64, "lat": pl.Float64, "lon": pl.Float64,
                "country": pl.String, "split": pl.String, "path": pl.String},
        orient="row",
    )
    return manifest, tmp_path / "emb.npy"


class TestExtraction:
    def test_shape_dtype_and_alignment(self, dataset):
        manifest, out = dataset
        emb, meta = extract_embeddings(manifest, StubBackbone(), out,
                                       batch_size=3, progress=False)

        assert emb.shape == (8, DIM)
        assert emb.dtype == np.float16
        assert meta.n_done == 8
        # Row i must encode image i, not some reordered version.
        for i in range(8):
            assert emb[i, 0] == pytest.approx(i * 30 / 255.0, abs=1e-3)

    def test_writes_sidecar_meta(self, dataset):
        manifest, out = dataset
        extract_embeddings(manifest, StubBackbone(), out, progress=False)

        payload = json.loads(meta_path_for(out).read_text())
        assert payload["n_rows"] == 8
        assert payload["embed_dim"] == DIM
        assert payload["dtype"] == "float16"
        assert payload["manifest_sig"] == manifest_signature(manifest)

    def test_requires_resolved_paths(self, dataset):
        manifest, out = dataset
        broken = manifest.with_columns(pl.lit(None, dtype=pl.String).alias("path"))
        with pytest.raises(ValueError, match="unresolved paths"):
            extract_embeddings(broken, StubBackbone(), out, progress=False)

    def test_crop_is_applied(self, dataset, tmp_path):
        manifest, out = dataset
        # PNG, not JPEG: lossy compression rings across the colour boundary and
        # would bleed red into the bottom half, masking what this asserts.
        path = tmp_path / "img" / "crop.png"
        img = Image.new("RGB", (64, 48), (255, 0, 0))
        img.paste(Image.new("RGB", (64, 24), (0, 0, 255)), (0, 24))
        img.save(path)

        one = manifest.head(1).with_columns(pl.lit(str(path)).alias("path"))

        full, _ = extract_embeddings(one, StubBackbone(), out, progress=False)
        assert full[0, 0] == pytest.approx(1.0, abs=1e-3)  # top-left is red

        bottom = CropSpec(name="bottom", top=0.5)
        cropped, _ = extract_embeddings(one, StubBackbone(), tmp_path / "emb2.npy",
                                        crop=bottom, progress=False)
        assert cropped[0, 0] == pytest.approx(0.0, abs=1e-3)  # now blue


class TestResume:
    def test_completed_run_does_not_re_encode(self, dataset):
        manifest, out = dataset
        backbone = StubBackbone()
        extract_embeddings(manifest, backbone, out, batch_size=4, progress=False)
        first = backbone.calls

        extract_embeddings(manifest, backbone, out, batch_size=4, progress=False)
        assert backbone.calls == first  # nothing re-encoded

    def test_resumes_from_partial(self, dataset):
        manifest, out = dataset
        extract_embeddings(manifest, StubBackbone(), out, batch_size=4,
                           progress=False)

        # Rewind the checkpoint to simulate an interrupted job.
        meta = json.loads(meta_path_for(out).read_text())
        meta["n_done"] = 4
        meta_path_for(out).write_text(json.dumps(meta))

        backbone = StubBackbone()
        emb, final = extract_embeddings(manifest, backbone, out, batch_size=4,
                                        progress=False)
        assert final.n_done == 8
        assert backbone.calls == 1  # only the remaining batch
        assert emb[7, 0] == pytest.approx(7 * 30 / 255.0, abs=1e-3)

    def test_resume_false_starts_over(self, dataset):
        manifest, out = dataset
        extract_embeddings(manifest, StubBackbone(), out, progress=False)

        backbone = StubBackbone()
        extract_embeddings(manifest, backbone, out, batch_size=8,
                           resume=False, progress=False)
        assert backbone.calls == 1


class TestCacheInvalidation:
    @pytest.mark.parametrize(
        "mutate,expected",
        [
            (lambda m: m.head(4), "row count"),
            (lambda m: m.reverse(), "manifest contents"),
        ],
    )
    def test_manifest_changes_refuse_to_resume(self, dataset, mutate, expected):
        manifest, out = dataset
        extract_embeddings(manifest, StubBackbone(), out, progress=False)

        with pytest.raises(ValueError, match=expected):
            extract_embeddings(mutate(manifest), StubBackbone(), out, progress=False)

    def test_backbone_change_refuses_to_resume(self, dataset):
        manifest, out = dataset
        extract_embeddings(manifest, StubBackbone(), out, progress=False)

        with pytest.raises(ValueError, match="backbone"):
            extract_embeddings(manifest, StubBackbone("other/model"), out,
                               progress=False)

    def test_crop_change_refuses_to_resume(self, dataset):
        manifest, out = dataset
        extract_embeddings(manifest, StubBackbone(), out, crop=FULL_FRAME,
                           progress=False)

        with pytest.raises(ValueError, match="crop"):
            extract_embeddings(manifest, StubBackbone(), out,
                               crop=CropSpec(name="other", top=0.1), progress=False)

    def test_error_message_says_how_to_recover(self, dataset):
        manifest, out = dataset
        extract_embeddings(manifest, StubBackbone(), out, progress=False)
        with pytest.raises(ValueError, match="Delete the cache"):
            extract_embeddings(manifest.head(4), StubBackbone(), out, progress=False)


class TestFailureHandling:
    def test_corrupt_image_is_recorded_not_fatal(self, dataset, tmp_path):
        manifest, out = dataset
        (tmp_path / "img" / "3.jpg").write_bytes(b"not a jpeg")

        emb, meta = extract_embeddings(manifest, StubBackbone(), out,
                                       batch_size=2, progress=False)
        assert meta.failed_ids == [3]
        assert meta.n_done == 8
        assert np.all(emb[3] == 0)
        assert emb[4, 0] == pytest.approx(4 * 30 / 255.0, abs=1e-3)


class TestLoad:
    def test_round_trip(self, dataset):
        manifest, out = dataset
        written, _ = extract_embeddings(manifest, StubBackbone(), out, progress=False)
        loaded, meta = load_embeddings(out, manifest=manifest)

        assert loaded.shape == written.shape
        assert meta.n_done == 8
        assert np.array_equal(np.asarray(loaded), np.asarray(written))

    def test_rejects_mismatched_manifest(self, dataset):
        manifest, out = dataset
        extract_embeddings(manifest, StubBackbone(), out, progress=False)

        with pytest.raises(ValueError, match="different manifest"):
            load_embeddings(out, manifest=manifest.reverse())

    def test_rejects_incomplete_cache(self, dataset):
        manifest, out = dataset
        extract_embeddings(manifest, StubBackbone(), out, progress=False)

        meta = json.loads(meta_path_for(out).read_text())
        meta["n_done"] = 5
        meta_path_for(out).write_text(json.dumps(meta))

        with pytest.raises(ValueError, match="incomplete"):
            load_embeddings(out, manifest=manifest)
