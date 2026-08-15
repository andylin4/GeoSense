"""Tests for Street View coverage probing and generation derivation.

No API key and no network: the HTTP layer is injected.
"""

from __future__ import annotations

import polars as pl
import pytest

from geoguessr.calibrate.fusion import CoverageTable
from geoguessr.data.streetview import (
    GENERATIONS,
    PanoramaMeta,
    StreetViewProbe,
    coverage_counts,
    derive_generation,
    sample_probe_locations,
)


class TestDeriveGeneration:
    @pytest.mark.parametrize(
        "date,expected",
        [
            ("2007-05", "gen1"),
            ("2008-05", "gen1"),
            ("2008-06", "gen2"),
            ("2011-12", "gen2"),
            ("2012-01", "gen3"),
            ("2017-05", "gen3"),
            ("2017-06", "gen4"),
            ("2024-03", "gen4"),
        ],
    )
    def test_boundaries(self, date, expected):
        assert derive_generation(date) == expected

    @pytest.mark.parametrize("bad", [None, "", "not-a-date", "20XX-01", "1850-01"])
    def test_unparseable_is_unknown_not_a_guess(self, bad):
        assert derive_generation(bad) == "unknown"

    def test_year_only_is_accepted(self):
        assert derive_generation("2020") == "gen4"

    def test_every_output_is_a_known_generation(self):
        for date in ("2007-01", "2010-01", "2015-01", "2022-01", None):
            assert derive_generation(date) in GENERATIONS


class TestPanoramaMeta:
    def test_ok_status_means_exists(self):
        assert PanoramaMeta(0, 0, "OK").exists
        assert not PanoramaMeta(0, 0, "ZERO_RESULTS").exists

    def test_distinguishes_google_from_photosphere(self):
        google = PanoramaMeta(0, 0, "OK", copyright="© Google")
        user = PanoramaMeta(0, 0, "OK", copyright="© Jane Doe")

        assert google.is_google
        assert not user.is_google

    def test_missing_copyright_is_not_google(self):
        assert not PanoramaMeta(0, 0, "OK").is_google

    def test_generation_comes_from_date(self):
        assert PanoramaMeta(0, 0, "OK", date="2019-04").generation == "gen4"


class TestSampleProbeLocations:
    @pytest.fixture
    def manifest(self):
        rows = [(i, 48.0 + i / 100, 2.0, "FR") for i in range(50)]
        rows += [(100 + i, 35.0, 139.0, "JP") for i in range(5)]
        return pl.DataFrame(
            rows,
            schema={"image_id": pl.Int64, "lat": pl.Float64, "lon": pl.Float64,
                    "country": pl.String},
            orient="row",
        )

    def test_caps_per_country(self, manifest):
        sampled = sample_probe_locations(manifest, per_country=10)
        counts = dict(sampled.group_by("country").len().iter_rows())
        assert counts["FR"] == 10
        assert counts["JP"] == 5  # only 5 available

    def test_balances_rather_than_following_natural_distribution(self, manifest):
        # FR outnumbers JP 10:1, but a balanced draw should even that out.
        sampled = sample_probe_locations(manifest, per_country=5)
        counts = dict(sampled.group_by("country").len().iter_rows())
        assert counts["FR"] == counts["JP"] == 5

    def test_reproducible(self, manifest):
        a = sample_probe_locations(manifest, per_country=7, seed=2)
        b = sample_probe_locations(manifest, per_country=7, seed=2)
        assert a["image_id"].to_list() == b["image_id"].to_list()

    def test_rejects_empty(self):
        empty = pl.DataFrame(schema={"image_id": pl.Int64, "lat": pl.Float64,
                                     "lon": pl.Float64, "country": pl.String})
        with pytest.raises(ValueError, match="empty manifest"):
            sample_probe_locations(empty)


class TestProbe:
    def test_parses_a_hit(self):
        def fetcher(url, params):
            return {"status": "OK", "pano_id": "abc", "date": "2019-07",
                    "copyright": "© Google"}

        meta = StreetViewProbe("key", fetcher=fetcher).probe(48.85, 2.29)

        assert meta.exists and meta.is_google
        assert meta.generation == "gen4"
        assert meta.pano_id == "abc"

    def test_parses_a_miss(self):
        probe = StreetViewProbe("key", fetcher=lambda u, p: {"status": "ZERO_RESULTS"})
        assert not probe.probe(0.0, 0.0).exists

    def test_sends_location_and_key(self):
        seen = {}

        def fetcher(url, params):
            seen.update(params)
            return {"status": "OK"}

        StreetViewProbe("secret", radius=25, fetcher=fetcher).probe(1.5, -2.5)

        assert seen["location"] == "1.5,-2.5"
        assert seen["key"] == "secret"
        assert seen["radius"] == 25

    def test_counts_calls_for_quota_tracking(self):
        probe = StreetViewProbe("key", fetcher=lambda u, p: {"status": "OK"})
        for _ in range(3):
            probe.probe(0, 0)
        assert probe.calls == 3

    def test_requires_a_key(self):
        with pytest.raises(ValueError, match="API key is required"):
            StreetViewProbe("")

    def test_probe_many_reports_progress(self):
        probe = StreetViewProbe("key", fetcher=lambda u, p: {"status": "OK"})
        seen = []
        results = probe.probe_many([(0, 0), (1, 1)], on_result=seen.append)

        assert len(results) == 2
        assert len(seen) == 2


class TestCoverageCounts:
    def test_tallies_by_country_and_generation(self):
        results = [
            PanoramaMeta(0, 0, "OK", date="2019-01", copyright="© Google"),
            PanoramaMeta(0, 0, "OK", date="2020-01", copyright="© Google"),
            PanoramaMeta(0, 0, "OK", date="2010-01", copyright="© Google"),
        ]
        counts = coverage_counts(results, ["FR", "FR", "JP"])

        assert counts["FR"]["gen4"] == 2
        assert counts["JP"]["gen2"] == 1

    def test_skips_misses(self):
        results = [PanoramaMeta(0, 0, "ZERO_RESULTS")]
        assert coverage_counts(results, ["FR"]) == {}

    def test_excludes_photospheres_by_default(self):
        results = [PanoramaMeta(0, 0, "OK", date="2019-01", copyright="© Jane")]
        assert coverage_counts(results, ["FR"]) == {}

    def test_can_include_photospheres(self):
        results = [PanoramaMeta(0, 0, "OK", date="2019-01", copyright="© Jane")]
        counts = coverage_counts(results, ["FR"], google_only=False)
        assert counts["FR"]["gen4"] == 1

    def test_output_feeds_the_coverage_table_directly(self):
        results = [
            PanoramaMeta(0, 0, "OK", date="2019-01", copyright="© Google"),
            PanoramaMeta(0, 0, "OK", date="2010-01", copyright="© Google"),
        ]
        counts = coverage_counts(results, ["FR", "JP"])

        table = CoverageTable.from_counts(counts, ["FR", "JP"], GENERATIONS)
        assert table.matrix[0, GENERATIONS.index("gen4")] > 0.9
        assert table.matrix[1, GENERATIONS.index("gen2")] > 0.9
