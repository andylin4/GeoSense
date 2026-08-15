"""Tests for the paid image fetcher.

The spend ceiling is the point of these: this is the only code in the project
that can cost money, so the guards get tested harder than the happy path.
No key, no network, no cost.
"""

from __future__ import annotations

import pytest

from geoguessr.data.streetview_images import (
    DOWNWARD,
    HORIZON,
    USD_PER_1000,
    BudgetExhausted,
    FetchConfig,
    StaticImageFetcher,
)


def fake_fetcher(calls=None):
    def fetch(url, params):
        if calls is not None:
            calls.append(params)
        return b"\xff\xd8\xff-fake-jpeg"

    return fetch


class TestSpendCeiling:
    def test_max_requests_is_mandatory(self):
        with pytest.raises(TypeError):
            StaticImageFetcher("key")  # no ceiling -> refuses to exist

    @pytest.mark.parametrize("bad", [0, -1, None])
    def test_rejects_nonpositive_ceiling(self, bad):
        with pytest.raises(ValueError, match="max_requests"):
            StaticImageFetcher("key", max_requests=bad)

    def test_raises_at_the_ceiling(self):
        fetcher = StaticImageFetcher("key", max_requests=2, fetcher=fake_fetcher())
        fetcher.fetch("a")
        fetcher.fetch("b")

        with pytest.raises(BudgetExhausted, match="2-request ceiling"):
            fetcher.fetch("c")

    def test_counts_and_costs(self):
        fetcher = StaticImageFetcher("key", max_requests=1000, fetcher=fake_fetcher())
        for i in range(10):
            fetcher.fetch(f"p{i}")

        assert fetcher.requests == 10
        assert fetcher.remaining == 990
        assert fetcher.estimated_cost_usd == pytest.approx(10 / 1000 * USD_PER_1000)
        assert fetcher.ceiling_cost_usd == pytest.approx(USD_PER_1000)

    def test_ceiling_cost_is_knowable_before_spending(self):
        fetcher = StaticImageFetcher("key", max_requests=5000, fetcher=fake_fetcher())
        assert fetcher.requests == 0
        assert fetcher.ceiling_cost_usd == pytest.approx(35.0)

    def test_requires_a_key(self):
        with pytest.raises(ValueError, match="API key"):
            StaticImageFetcher("", max_requests=10)


class TestFetchByPanoId:
    def test_sends_pano_not_location(self):
        calls = []
        StaticImageFetcher("k", max_requests=5,
                           fetcher=fake_fetcher(calls)).fetch("PANO123")

        assert calls[0]["pano"] == "PANO123"
        assert "location" not in calls[0]  # coordinate requests corrupt labels

    def test_empty_pano_id_rejected(self):
        fetcher = StaticImageFetcher("k", max_requests=5, fetcher=fake_fetcher())
        with pytest.raises(ValueError, match="pano_id is required"):
            fetcher.fetch("")

    def test_failed_request_does_not_consume_budget_silently(self):
        def boom(url, params):
            raise RuntimeError("503")

        fetcher = StaticImageFetcher("k", max_requests=5, fetcher=boom)
        with pytest.raises(RuntimeError):
            fetcher.fetch("a")
        # The counter increments only on success.
        assert fetcher.requests == 0


class TestFraming:
    def test_horizon_and_downward_differ_in_pitch(self):
        assert HORIZON.pitch == 0.0
        assert DOWNWARD.pitch == -90.0

    def test_config_reaches_the_request(self):
        calls = []
        config = FetchConfig(size="512x512", pitch=-30, heading=90, fov=110)
        StaticImageFetcher("k", max_requests=1, config=config,
                           fetcher=fake_fetcher(calls)).fetch("p")

        assert calls[0]["size"] == "512x512"
        assert calls[0]["pitch"] == -30
        assert calls[0]["heading"] == 90
        assert calls[0]["fov"] == 110

    def test_heading_omitted_when_unset(self):
        calls = []
        StaticImageFetcher("k", max_requests=1, fetcher=fake_fetcher(calls)).fetch("p")
        assert "heading" not in calls[0]

    def test_with_heading_returns_a_new_config(self):
        rotated = HORIZON.with_heading(180)
        assert rotated.heading == 180
        assert HORIZON.heading is None  # frozen, original untouched

    def test_both_presets_document_their_tradeoff(self):
        assert "NOT visible" in HORIZON.note
        assert "cannot use those cues" in DOWNWARD.note


class TestFetchToDir:
    def test_writes_files_named_by_pano_id(self, tmp_path):
        fetcher = StaticImageFetcher("k", max_requests=10, fetcher=fake_fetcher())
        saved = fetcher.fetch_to_dir(["a", "b"], tmp_path)

        assert set(saved) == {"a", "b"}
        assert (tmp_path / "a.jpg").read_bytes().startswith(b"\xff\xd8\xff")

    def test_skips_already_downloaded(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"existing")
        fetcher = StaticImageFetcher("k", max_requests=10, fetcher=fake_fetcher())

        fetcher.fetch_to_dir(["a", "b"], tmp_path)

        assert fetcher.requests == 1  # only "b" cost anything
        assert fetcher.skipped == 1
        assert (tmp_path / "a.jpg").read_bytes() == b"existing"

    def test_rerun_is_free(self, tmp_path):
        first = StaticImageFetcher("k", max_requests=10, fetcher=fake_fetcher())
        first.fetch_to_dir(["a", "b", "c"], tmp_path)

        second = StaticImageFetcher("k", max_requests=10, fetcher=fake_fetcher())
        second.fetch_to_dir(["a", "b", "c"], tmp_path)
        assert second.requests == 0

    def test_stops_at_ceiling_without_raising(self, tmp_path):
        events = []
        fetcher = StaticImageFetcher("k", max_requests=2, fetcher=fake_fetcher())
        saved = fetcher.fetch_to_dir(
            ["a", "b", "c", "d"], tmp_path,
            on_progress=lambda pid, status: events.append((pid, status)),
        )

        assert fetcher.requests == 2
        assert len(saved) == 2  # partial scrape still usable
        assert ("c", "budget-exhausted") in events

    def test_one_bad_pano_does_not_abort_the_run(self, tmp_path):
        def flaky(url, params):
            if params["pano"] == "b":
                raise RuntimeError("bad pano")
            return b"\xff\xd8\xff"

        fetcher = StaticImageFetcher("k", max_requests=10, fetcher=flaky)
        saved = fetcher.fetch_to_dir(["a", "b", "c"], tmp_path)

        assert set(saved) == {"a", "c"}

    def test_summary_reports_spend(self, tmp_path):
        fetcher = StaticImageFetcher("k", max_requests=100, fetcher=fake_fetcher())
        fetcher.fetch_to_dir(["a"], tmp_path)
        assert "1 requests" in fetcher.summary()
