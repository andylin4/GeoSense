"""Tests for baseline predictors.

The zero-shot path is tested against a stub backbone rather than real
StreetCLIP: the logic under test is prompt assembly, OTHER-bucket handling, and
softmax shaping, none of which need 1.7GB of weights to verify.
"""

from __future__ import annotations

import numpy as np
import pytest

from geoguessr.eval.baselines import (
    PROMPT_TEMPLATES,
    coverage_weights_from_labels,
    prior_predict_fn,
    uniform_predict_fn,
    zero_shot_predict_fn,
)
from geoguessr.eval.harness import evaluate


class StubBackbone:
    """Country embeddings are one-hot axes; an "image" is an axis index."""

    def __init__(self, dim: int = 2):
        self.dim = dim
        self.text_calls = 0
        self.seen_prompts: list[str] = []

    def encode_texts(self, prompts, normalize=True):
        self.text_calls += 1
        self.seen_prompts.extend(prompts)
        return np.eye(self.dim)[: len(prompts)]

    def encode_images(self, images, normalize=True):
        return np.eye(self.dim)[list(images)]

    @property
    def logit_scale(self):
        return 100.0


CODES = ["FR", "JP", "XX"]


class TestZeroShot:
    def test_predicts_the_matching_country(self):
        fn = zero_shot_predict_fn(StubBackbone(), CODES)
        probs = fn([0, 1])  # image 0 aligns with FR, image 1 with JP
        assert probs.argmax(axis=1).tolist() == [0, 1]

    def test_returns_valid_distributions(self):
        fn = zero_shot_predict_fn(StubBackbone(), CODES)
        probs = fn([0, 1, 0])
        assert probs.shape == (3, 3)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert (probs >= 0).all()

    def test_other_never_wins_but_holds_mass(self):
        fn = zero_shot_predict_fn(StubBackbone(), CODES)
        probs = fn([0, 1])
        other = probs[:, CODES.index("XX")]
        assert (other > 0).all()
        assert probs.argmax(axis=1).tolist() != [2, 2]
        assert (other < probs.max(axis=1)).all()

    def test_text_embeddings_computed_once_not_per_batch(self):
        stub = StubBackbone()
        fn = zero_shot_predict_fn(stub, CODES)
        calls_after_build = stub.text_calls

        fn([0])
        fn([1])
        fn([0, 1])
        assert stub.text_calls == calls_after_build

    def test_averages_across_all_templates(self):
        stub = StubBackbone()
        zero_shot_predict_fn(stub, CODES)
        assert stub.text_calls == len(PROMPT_TEMPLATES)

    def test_prompts_use_display_names_not_codes(self):
        stub = StubBackbone()
        zero_shot_predict_fn(stub, CODES, templates=["A photo in {country}."])
        assert "A photo in France." in stub.seen_prompts
        assert not any("FR" in p for p in stub.seen_prompts)

    def test_other_is_not_given_a_prompt(self):
        stub = StubBackbone()
        zero_shot_predict_fn(stub, CODES, templates=["A photo in {country}."])
        assert not any("Other" in p for p in stub.seen_prompts)

    def test_works_without_an_other_class(self):
        fn = zero_shot_predict_fn(StubBackbone(), ["FR", "JP"])
        probs = fn([0, 1])
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_rejects_all_other_class_list(self):
        with pytest.raises(ValueError, match="OTHER bucket"):
            zero_shot_predict_fn(StubBackbone(), ["XX"])

    def test_plugs_into_the_harness(self):
        fn = zero_shot_predict_fn(StubBackbone(), CODES)
        dataset = [(0, 0), (1, 1), (0, 0)]
        result = evaluate(fn, dataset, CODES, name="zero-shot-stub", progress=False)
        assert result.top1 == pytest.approx(1.0)


class TestPriorBaselines:
    def test_uniform_is_chance(self):
        fn = uniform_predict_fn(CODES)
        probs = fn([None] * 4)
        assert probs.shape == (4, 3)
        assert np.allclose(probs, 1 / 3)

    def test_weighted_prior_reflects_coverage(self):
        fn = prior_predict_fn(CODES, {"FR": 8, "JP": 2})
        probs = fn([None])
        assert probs[0, 0] == pytest.approx(0.8)
        assert probs[0, 1] == pytest.approx(0.2)
        assert probs[0, 2] == pytest.approx(0.0)

    def test_prior_baseline_beats_chance_on_imbalanced_data(self):
        # The reason macro accuracy exists: always guessing the majority class
        # posts a strong top-1 while being useless.
        weights = coverage_weights_from_labels(["FR"] * 80 + ["JP"] * 20)
        fn = prior_predict_fn(CODES, weights)
        dataset = [(None, 0)] * 80 + [(None, 1)] * 20

        result = evaluate(fn, dataset, CODES, progress=False)
        assert result.top1 == pytest.approx(0.8)
        assert result.macro_top1 == pytest.approx(0.5)

    def test_missing_classes_get_zero_weight(self):
        fn = prior_predict_fn(CODES, {"FR": 1.0})
        assert fn([None])[0].tolist() == [1.0, 0.0, 0.0]

    def test_zero_weights_rejected(self):
        with pytest.raises(ValueError, match="sum to zero"):
            prior_predict_fn(CODES, {"FR": 0.0})

    def test_negative_weights_are_clamped(self):
        fn = prior_predict_fn(CODES, {"FR": 1.0, "JP": -5.0})
        assert fn([None])[0].tolist() == [1.0, 0.0, 0.0]


class TestCoverageWeights:
    def test_counts_labels(self):
        assert coverage_weights_from_labels(["FR", "FR", "JP"]) == {"FR": 2.0, "JP": 1.0}

    def test_empty(self):
        assert coverage_weights_from_labels([]) == {}
