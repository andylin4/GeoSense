"""Tests for meta fusion.

No real meta data is needed: the coverage table is an interface, and these
verify the fusion algebra behaves sanely against synthetic tables.
"""

from __future__ import annotations

import numpy as np
import pytest

from geoguessr.calibrate.fusion import CoverageTable, fuse, fuse_predict_fn
from geoguessr.eval.harness import evaluate

COUNTRIES = ["FR", "JP", "BR"]
GENERATIONS = ["gen2", "gen3", "gen4"]


@pytest.fixture
def table():
    # FR is exclusively gen4, JP exclusively gen2, BR mixed.
    return CoverageTable.from_counts(
        {
            "FR": {"gen4": 100},
            "JP": {"gen2": 100},
            "BR": {"gen2": 50, "gen4": 50},
        },
        COUNTRIES,
        GENERATIONS,
    )


class TestCoverageTable:
    def test_rows_are_distributions(self, table):
        assert np.allclose(table.matrix.sum(axis=1), 1.0)

    def test_dominant_generation_gets_most_mass(self, table):
        assert table.matrix[0, GENERATIONS.index("gen4")] > 0.9
        assert table.matrix[1, GENERATIONS.index("gen2")] > 0.9

    def test_unseen_combination_is_floored_not_zeroed(self, table):
        # FR never observed with gen2 -- must stay recoverable.
        value = table.matrix[0, GENERATIONS.index("gen2")]
        assert 0 < value < 0.01

    def test_country_without_observations_is_uniform(self):
        table = CoverageTable.from_counts({"FR": {"gen4": 10}}, COUNTRIES, GENERATIONS)
        assert np.allclose(table.matrix[1], 1 / 3)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="expected"):
            CoverageTable(COUNTRIES, GENERATIONS, np.zeros((2, 3)))

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="negative"):
            CoverageTable(COUNTRIES, GENERATIONS, -np.ones((3, 3)))

    def test_round_trip(self, table, tmp_path):
        loaded = CoverageTable.load(table.save(tmp_path / "coverage.json"))
        assert loaded.class_names == COUNTRIES
        assert np.allclose(loaded.matrix, table.matrix)


class TestLikelihood:
    def test_hard_generation_favours_matching_country(self, table):
        gen4 = np.array([0.0, 0.0, 1.0])
        likelihood = table.likelihood(gen4)
        assert likelihood[COUNTRIES.index("FR")] > likelihood[COUNTRIES.index("JP")]

    def test_rejects_wrong_generation_count(self, table):
        with pytest.raises(ValueError, match="expected 3 generation"):
            table.likelihood(np.array([0.5, 0.5]))


class TestFuse:
    def test_meta_can_overturn_a_close_call(self, table):
        # Country head slightly prefers JP; meta says gen4, which JP never uses.
        country = np.array([[0.40, 0.45, 0.15]])
        gen4 = np.array([[0.0, 0.0, 1.0]])

        fused = fuse(country, gen4, table)
        assert fused.argmax() == COUNTRIES.index("FR")

    def test_output_is_a_distribution(self, table):
        fused = fuse(np.array([[0.3, 0.4, 0.3]]), np.array([[0.2, 0.3, 0.5]]), table)
        assert fused.sum() == pytest.approx(1.0)
        assert (fused >= 0).all()

    def test_zero_confidence_is_exactly_the_identity(self, table):
        country = np.array([[0.5, 0.3, 0.2]])
        fused = fuse(country, np.array([[1.0, 0.0, 0.0]]), table, meta_confidence=0.0)
        assert np.allclose(fused, country)

    def test_partial_confidence_moves_partway(self, table):
        country = np.array([[0.40, 0.45, 0.15]])
        gen4 = np.array([[0.0, 0.0, 1.0]])

        weak = fuse(country, gen4, table, meta_confidence=0.25)
        strong = fuse(country, gen4, table, meta_confidence=1.0)
        fr = COUNTRIES.index("FR")
        assert country[0, fr] < weak[0, fr] < strong[0, fr]

    def test_uncertain_meta_barely_moves_anything(self, table):
        country = np.array([[0.5, 0.3, 0.2]])
        # A meta head with no opinion.
        fused = fuse(country, np.array([[1 / 3, 1 / 3, 1 / 3]]), table)
        assert np.allclose(fused, country, atol=0.02)

    def test_never_annihilates_to_nan(self, table):
        # Country head is certain about a country the meta evidence rules out.
        country = np.array([[0.0, 1.0, 0.0]])
        fused = fuse(country, np.array([[0.0, 0.0, 1.0]]), table)
        assert np.isfinite(fused).all()
        assert fused.sum() == pytest.approx(1.0)

    def test_batch(self, table):
        country = np.array([[0.4, 0.45, 0.15], [0.2, 0.6, 0.2]])
        gens = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        fused = fuse(country, gens, table)

        assert fused.shape == (2, 3)
        assert fused[0].argmax() == COUNTRIES.index("FR")
        assert fused[1].argmax() == COUNTRIES.index("JP")

    def test_rejects_class_count_mismatch(self, table):
        with pytest.raises(ValueError, match="coverage table has 3"):
            fuse(np.array([[0.5, 0.5]]), np.array([[0.3, 0.3, 0.4]]), table)

    def test_rejects_bad_confidence(self, table):
        with pytest.raises(ValueError, match="meta_confidence"):
            fuse(np.array([[0.3, 0.4, 0.3]]), np.array([[0.2, 0.3, 0.5]]),
                 table, meta_confidence=1.5)


class TestFusedPredictFn:
    def test_composes_into_the_same_harness(self, table):
        # Country head is wrong-ish alone; meta fixes it.
        country_fn = lambda items: np.tile([0.40, 0.45, 0.15], (len(items), 1))
        meta_fn = lambda items: np.tile([0.0, 0.0, 1.0], (len(items), 1))

        dataset = [(None, COUNTRIES.index("FR"))] * 8

        before = evaluate(country_fn, dataset, COUNTRIES, progress=False)
        after = evaluate(fuse_predict_fn(country_fn, meta_fn, table), dataset,
                         COUNTRIES, progress=False)

        assert before.top1 == 0.0
        assert after.top1 == 1.0
