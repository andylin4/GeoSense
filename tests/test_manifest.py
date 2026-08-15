"""Tests for the Phase 1 data layer.

Built against a synthetic OSV-5M-shaped CSV so they run without the 258GB
dataset.
"""

from __future__ import annotations

import polars as pl
import pytest

from geoguessr.data.manifest import (
    MANIFEST_COLUMNS,
    attach_paths,
    build_manifest,
    manifest_stats,
    read_manifest,
    subsample,
    write_manifest,
)


def write_csv(tmp_path, rows):
    path = tmp_path / "meta.csv"
    pl.DataFrame(
        rows, schema={"id": pl.Int64, "latitude": pl.Float64, "longitude": pl.Float64,
                      "country": pl.String}, orient="row"
    ).write_csv(path)
    return path


@pytest.fixture
def csv_path(tmp_path):
    # 6 France, 3 Japan, 2 China (no Street View coverage -> dropped), 1 null.
    rows = [(i, 48.8 + i / 100, 2.3, "FR") for i in range(1, 7)]
    rows += [(10 + i, 35.6, 139.7, "JP") for i in range(3)]
    rows += [(20 + i, 39.9, 116.4, "CN") for i in range(2)]
    rows += [(99, None, 2.0, "FR")]
    return write_csv(tmp_path, rows)


class TestBuildManifest:
    def test_schema_and_split(self, csv_path):
        m = build_manifest(csv_path, split="train")
        assert m.columns == MANIFEST_COLUMNS
        assert m["split"].unique().to_list() == ["train"]

    def test_drops_countries_without_street_view_coverage(self, csv_path):
        m = build_manifest(csv_path, split="train")
        assert set(m["country"].to_list()) == {"FR", "JP"}
        assert m.height == 9  # 6 FR + 3 JP; CN dropped, null row dropped

    def test_drops_rows_with_missing_coordinates(self, csv_path):
        assert 99 not in build_manifest(csv_path, split="train")["image_id"].to_list()

    def test_path_starts_null(self, csv_path):
        assert build_manifest(csv_path, split="train")["path"].null_count() == 9

    def test_stores_country_code_not_integer_label(self, csv_path):
        # Relabeling must stay free; no class index is baked in.
        m = build_manifest(csv_path, split="train")
        assert m["country"].dtype == pl.String
        assert "label" not in m.columns

    def test_sorted_by_id(self, csv_path):
        ids = build_manifest(csv_path, split="train")["image_id"].to_list()
        assert ids == sorted(ids)

    def test_normalizes_country_spellings(self, tmp_path):
        path = write_csv(tmp_path, [(1, 50.0, 14.0, "Czech Republic"),
                                    (2, 50.0, 14.0, "Czechia")])
        assert build_manifest(path, split="train")["country"].to_list() == ["CZ", "CZ"]

    def test_deduplicates_ids(self, tmp_path):
        path = write_csv(tmp_path, [(1, 48.8, 2.3, "FR"), (1, 48.8, 2.3, "FR")])
        assert build_manifest(path, split="train").height == 1

    def test_rejects_csv_missing_columns(self, tmp_path):
        path = tmp_path / "bad.csv"
        pl.DataFrame({"id": [1], "latitude": [1.0]}).write_csv(path)
        with pytest.raises(ValueError, match="missing required columns"):
            build_manifest(path, split="train")

    def test_rejects_unknown_strategy(self, csv_path):
        with pytest.raises(ValueError, match="unknown strategy"):
            build_manifest(csv_path, split="train", strategy="magic")


class TestSampling:
    def test_limit_is_respected(self, csv_path):
        assert build_manifest(csv_path, split="train", limit=4).height == 4

    def test_limit_above_available_is_a_noop(self, csv_path):
        assert build_manifest(csv_path, split="train", limit=1000).height == 9

    def test_random_sampling_is_reproducible(self, csv_path):
        a = build_manifest(csv_path, split="train", limit=5, seed=7)
        b = build_manifest(csv_path, split="train", limit=5, seed=7)
        assert a["image_id"].to_list() == b["image_id"].to_list()

    def test_stratified_balances_across_countries(self, csv_path):
        # Random would likely skew toward France (6 vs 3).
        m = build_manifest(csv_path, split="train", limit=6, strategy="stratified")
        counts = dict(m.group_by("country").len().iter_rows())
        assert counts == {"FR": 3, "JP": 3}

    def test_stratified_redistributes_when_a_country_runs_out(self, csv_path):
        # JP only has 3; the rest of the budget must come from FR.
        m = build_manifest(csv_path, split="train", limit=8, strategy="stratified")
        counts = dict(m.group_by("country").len().iter_rows())
        assert m.height == 8
        assert counts["JP"] == 3
        assert counts["FR"] == 5

    def test_stratified_sampling_is_reproducible(self, csv_path):
        # Regression: quota allocation used to iterate a non-order-stable
        # group_by, so the same seed selected different rows per process and
        # silently invalidated the embedding cache.
        a = build_manifest(csv_path, split="train", limit=7,
                           strategy="stratified", seed=3)
        b = build_manifest(csv_path, split="train", limit=7,
                           strategy="stratified", seed=3)
        assert a["image_id"].to_list() == b["image_id"].to_list()

    def test_stratified_is_order_independent(self, csv_path):
        # The same rows in a different input order must yield the same sample.
        built = build_manifest(csv_path, split="train")
        forward = subsample(built, limit=7, strategy="stratified", seed=3)
        backward = subsample(built.reverse(), limit=7, strategy="stratified", seed=3)
        assert forward["image_id"].to_list() == backward["image_id"].to_list()

    def test_max_per_country_caps(self, csv_path):
        m = build_manifest(csv_path, split="train", max_per_country=2)
        counts = dict(m.group_by("country").len().iter_rows())
        assert counts == {"FR": 2, "JP": 2}


class TestSubsampleAfterAttachPaths:
    """The real order of operations: build -> attach_paths -> subsample.

    Sampling before knowing which images exist would spend the budget on rows
    that then get dropped.
    """

    def test_budget_is_spent_on_rows_that_survive_path_resolution(
        self, csv_path, tmp_path
    ):
        images = tmp_path / "img"
        images.mkdir()
        for i in (1, 2, 3, 4):  # only 4 of the 9 manifest rows exist
            (images / f"{i}.jpg").write_bytes(b"x")

        resolved = attach_paths(build_manifest(csv_path, split="train"), images)
        sampled = subsample(resolved, limit=3, seed=0)

        assert sampled.height == 3
        assert sampled["path"].null_count() == 0

    def test_preserves_column_contract_and_ordering(self, csv_path):
        m = subsample(build_manifest(csv_path, split="train"), limit=5, seed=1)
        assert m.columns == MANIFEST_COLUMNS
        assert m["image_id"].to_list() == sorted(m["image_id"].to_list())

    def test_noop_without_limit(self, csv_path):
        built = build_manifest(csv_path, split="train")
        assert subsample(built).equals(built)

    def test_rejects_unknown_strategy(self, csv_path):
        with pytest.raises(ValueError, match="unknown strategy"):
            subsample(build_manifest(csv_path, split="train"), strategy="nope")


class TestAttachPaths:
    def test_resolves_and_drops_missing(self, csv_path, tmp_path):
        images = tmp_path / "img"
        images.mkdir()
        for i in (1, 2, 3):
            (images / f"{i}.jpg").write_bytes(b"x")

        m = attach_paths(build_manifest(csv_path, split="train"), images)
        assert m.height == 3
        assert set(m["image_id"].to_list()) == {1, 2, 3}
        assert m["path"].null_count() == 0

    def test_finds_images_in_nested_directories(self, csv_path, tmp_path):
        images = tmp_path / "img"
        (images / "shard0").mkdir(parents=True)
        (images / "shard0" / "1.jpg").write_bytes(b"x")

        assert attach_paths(build_manifest(csv_path, split="train"), images).height == 1

    def test_require_all_raises_on_gaps(self, csv_path, tmp_path):
        images = tmp_path / "img"
        images.mkdir()
        (images / "1.jpg").write_bytes(b"x")

        with pytest.raises(ValueError, match="have no image"):
            attach_paths(build_manifest(csv_path, split="train"), images,
                         require_all=True)

    def test_missing_root_raises(self, csv_path, tmp_path):
        with pytest.raises(FileNotFoundError):
            attach_paths(build_manifest(csv_path, split="train"), tmp_path / "nope")


class TestRoundTrip:
    def test_write_then_read(self, csv_path, tmp_path):
        m = build_manifest(csv_path, split="train")
        path = write_manifest(m, tmp_path / "artifacts" / "manifest.parquet")
        assert path.exists()
        assert read_manifest(path).equals(m)

    def test_read_rejects_non_manifest(self, tmp_path):
        path = tmp_path / "junk.parquet"
        pl.DataFrame({"a": [1]}).write_parquet(path)
        with pytest.raises(ValueError, match="not a manifest"):
            read_manifest(path)


class TestStats:
    def test_reports_shape_and_imbalance(self, csv_path):
        stats = manifest_stats(build_manifest(csv_path, split="train"))
        assert stats["rows"] == 9
        assert stats["countries_present"] == 2
        assert stats["countries_missing"] == 98  # 100 classes total
        assert stats["largest_share"] == pytest.approx(6 / 9)
        assert stats["with_path"] == 0
