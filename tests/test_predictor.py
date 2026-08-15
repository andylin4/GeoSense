"""Tests for the live prediction path."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from PIL import Image

from geoguessr.calibrate.temperature import TemperatureScaler
from geoguessr.data.crop import FULL_FRAME, GEOGUESSR_16_9, CropSpec
from geoguessr.serve.predictor import Guess, Predictor, format_guesses
from geoguessr.train.head import train_linear_probe

CLASSES = ["FR", "JP", "BR"]


class StubBackbone:
    model_id = "stub/model"
    embed_dim = 4

    def __init__(self):
        self.seen_sizes = []

    def encode_images(self, images, normalize=True):
        self.seen_sizes.extend(img.size for img in images)
        out = np.zeros((len(images), 4), dtype=np.float32)
        for i, img in enumerate(images):
            out[i, min(img.getpixel((0, 0))[0] // 100, 2)] = 5.0
        return out


@pytest.fixture
def head():
    rng = np.random.default_rng(0)
    embeddings, labels = [], []
    for cls in range(3):
        center = np.zeros(4)
        center[cls] = 5.0
        embeddings.append(rng.normal(center, 0.3, size=(30, 4)))
        labels.append(np.full(30, cls))
    return train_linear_probe(
        np.vstack(embeddings).astype(np.float32), np.concatenate(labels), CLASSES
    )


def image_for(cls: int, size=(320, 180)) -> Image.Image:
    return Image.new("RGB", size, (cls * 100, 0, 0))


@pytest.fixture
def predictor(head):
    return Predictor(StubBackbone(), head, crop=FULL_FRAME)


class TestPrediction:
    def test_ranks_the_right_country_first(self, predictor):
        for cls, code in enumerate(CLASSES):
            assert predictor.predict(image_for(cls))[0].code == code

    def test_returns_top_k_in_descending_order(self, predictor):
        guesses = predictor.predict(image_for(0), top_k=3)
        assert len(guesses) == 3
        probs = [g.probability for g in guesses]
        assert probs == sorted(probs, reverse=True)

    def test_top_k_is_clamped_to_class_count(self, predictor):
        assert len(predictor.predict(image_for(0), top_k=99)) == 3

    def test_probabilities_form_a_distribution(self, predictor):
        probs = predictor.probabilities(image_for(1))
        assert probs.shape == (3,)
        assert probs.sum() == pytest.approx(1.0)

    def test_guess_carries_display_name(self, predictor):
        assert predictor.predict(image_for(1))[0].name == "Japan"

    def test_converts_non_rgb_input(self, predictor):
        grayscale = Image.new("L", (320, 180), 0)
        assert len(predictor.predict(grayscale)) == 3


class TestCropIsApplied:
    def test_crop_reaches_the_backbone(self, head):
        backbone = StubBackbone()
        crop = CropSpec(name="half", top=0.5)
        Predictor(backbone, head, crop=crop).predict(image_for(0, size=(320, 180)))

        assert backbone.seen_sizes == [(320, 90)]

    def test_full_frame_passes_the_whole_image(self, head):
        backbone = StubBackbone()
        Predictor(backbone, head, crop=FULL_FRAME).predict(image_for(0))
        assert backbone.seen_sizes == [(320, 180)]

    def test_unverified_crop_warns(self, head):
        with pytest.warns(UserWarning, match="not validated against a live"):
            Predictor(StubBackbone(), head, crop=GEOGUESSR_16_9)

    def test_verified_crop_does_not_warn(self, head):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Predictor(StubBackbone(), head, crop=FULL_FRAME)


class TestCalibration:
    def test_temperature_is_applied(self, head):
        rng = np.random.default_rng(1)
        probs = rng.random((200, 3))
        probs /= probs.sum(axis=1, keepdims=True)
        labels = rng.integers(0, 3, 200)

        scaler = TemperatureScaler(temperature=3.0)
        hot = Predictor(StubBackbone(), head, scaler=scaler, crop=FULL_FRAME)
        cold = Predictor(StubBackbone(), head, crop=FULL_FRAME)

        image = image_for(0)
        # A temperature above 1 must flatten the distribution.
        assert hot.probabilities(image).max() < cold.probabilities(image).max()

    def test_calibration_does_not_change_the_winner(self, head):
        scaler = TemperatureScaler(temperature=4.0)
        hot = Predictor(StubBackbone(), head, scaler=scaler, crop=FULL_FRAME)
        cold = Predictor(StubBackbone(), head, crop=FULL_FRAME)

        for cls in range(3):
            assert hot.predict(image_for(cls))[0].code == cold.predict(
                image_for(cls))[0].code

    def test_describe_flags_missing_calibration(self, predictor):
        assert "UNCALIBRATED" in predictor.describe()

    def test_describe_reports_temperature(self, head):
        predictor = Predictor(StubBackbone(), head,
                              scaler=TemperatureScaler(temperature=1.5),
                              crop=FULL_FRAME)
        assert "1.5000" in predictor.describe()


class TestArtifactLoading:
    def test_missing_head_says_how_to_make_one(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="run_pipeline"):
            Predictor.from_artifacts(tmp_path, tag="999",
                                     backbone=StubBackbone())

    def test_loads_head_and_temperature(self, tmp_path, head):
        head.save(tmp_path / "head_7.joblib")
        TemperatureScaler(temperature=2.5).save(tmp_path / "temperature_7.json")

        predictor = Predictor.from_artifacts(tmp_path, tag="7",
                                             backbone=StubBackbone(),
                                             crop=FULL_FRAME)
        assert predictor.scaler.temperature == pytest.approx(2.5)
        assert predictor.head.class_names == CLASSES

    def test_temperature_is_optional(self, tmp_path, head):
        head.save(tmp_path / "head_7.joblib")
        predictor = Predictor.from_artifacts(tmp_path, tag="7",
                                             backbone=StubBackbone(),
                                             crop=FULL_FRAME)
        assert predictor.scaler is None


class TestFormatting:
    def test_renders_name_and_percentage(self):
        text = format_guesses([Guess("PL", "Poland", 0.34),
                               Guess("CZ", "Czechia", 0.21)])
        assert "Poland" in text and "34.0%" in text
        assert text.count("\n") == 1
