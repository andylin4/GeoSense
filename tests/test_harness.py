"""Tests for the evaluate() interface.

The fake predict_fn here treats each "image" as the class index it should
predict, which makes expected metrics trivially hand-computable and keeps these
tests free of any model or dataset dependency.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from geoguessr.eval.harness import evaluate

CLASSES = ["france", "japan", "brazil"]


def make_predict_fn(confidence: float = 0.8):
    """Returns a predict_fn that confidently predicts whatever int it is given."""

    def predict_fn(images: list[int]) -> np.ndarray:
        n_classes = len(CLASSES)
        spread = (1.0 - confidence) / (n_classes - 1)
        probs = np.full((len(images), n_classes), spread)
        probs[np.arange(len(images)), images] = confidence
        return probs

    return predict_fn


# (predicted_class, true_class): two hits, one miss.
DATASET = [(0, 0), (1, 1), (0, 2)]


def test_headline_metrics():
    result = evaluate(make_predict_fn(), DATASET, CLASSES, progress=False)

    assert result.n_samples == 3
    assert result.top1 == pytest.approx(2 / 3)
    assert result.top5 == pytest.approx(1.0)  # clamped to n_classes=3
    # france 1/1, japan 1/1, brazil 0/1
    assert result.macro_top1 == pytest.approx(2 / 3)


def test_batching_does_not_change_results():
    dataset = [(i % 3, (i + 1) % 3) for i in range(50)]
    baseline = evaluate(make_predict_fn(), dataset, CLASSES, batch_size=1, progress=False)

    for batch_size in (2, 7, 64):
        other = evaluate(
            make_predict_fn(), dataset, CLASSES, batch_size=batch_size, progress=False
        )
        assert other.top1 == pytest.approx(baseline.top1)
        assert other.ece == pytest.approx(baseline.ece)
        assert other.confusion.tolist() == baseline.confusion.tolist()


def test_ece_reflects_overconfidence():
    # Always right, but only claims 0.8 -> underconfident by 0.2.
    dataset = [(0, 0), (1, 1), (2, 2)]
    result = evaluate(make_predict_fn(0.8), dataset, CLASSES, progress=False)
    assert result.ece == pytest.approx(0.2, abs=1e-9)


def test_worst_classes_surfaces_the_failing_country():
    # brazil is always predicted as france.
    dataset = [(0, 0)] * 20 + [(0, 2)] * 20
    result = evaluate(make_predict_fn(), dataset, CLASSES, progress=False)

    worst = result.worst_classes(k=1, min_support=10)
    assert worst[0][0] == "brazil"
    assert worst[0][1] == pytest.approx(0.0)
    assert worst[0][2] == 20


def test_worst_classes_respects_min_support():
    dataset = [(0, 0)] * 20 + [(0, 2)]
    result = evaluate(make_predict_fn(), dataset, CLASSES, progress=False)
    # brazil fails but has support 1, below the threshold.
    assert [row[0] for row in result.worst_classes(min_support=10)] == ["france"]


def test_top_confusions():
    dataset = [(0, 2)] * 5 + [(1, 0)] * 3
    result = evaluate(make_predict_fn(), dataset, CLASSES, progress=False)

    assert result.top_confusions(k=2) == [
        ("brazil", "france", 5),
        ("france", "japan", 3),
    ]


def test_perfect_model_has_no_confusions():
    dataset = [(0, 0), (1, 1), (2, 2)]
    result = evaluate(make_predict_fn(), dataset, CLASSES, progress=False)
    assert result.top_confusions() == []


def test_result_round_trips_through_json(tmp_path):
    result = evaluate(make_predict_fn(), DATASET, CLASSES, name="zero-shot", progress=False)
    path = result.save(tmp_path / "run.json")

    loaded = json.loads(path.read_text())
    assert loaded["name"] == "zero-shot"
    assert loaded["top1"] == pytest.approx(2 / 3)
    assert loaded["class_names"] == CLASSES
    assert len(loaded["confusion"]) == 3


def test_absent_class_serializes_as_null_not_nan():
    # japan and brazil never appear as true labels -> nan accuracy.
    result = evaluate(make_predict_fn(), [(0, 0)], CLASSES, progress=False)
    # allow_nan=False would raise if nan leaked into the JSON.
    payload = json.dumps(result.to_dict(), allow_nan=False)
    assert json.loads(payload)["per_class_acc"][1] is None


def test_summary_mentions_the_headline_numbers():
    result = evaluate(make_predict_fn(), DATASET, CLASSES, name="probe", progress=False)
    text = result.summary()
    assert "probe" in text
    assert "top-1" in text
    assert "ECE" in text


def test_bad_predict_fn_shape_is_caught():
    def wrong_shape(images):
        return np.ones((len(images), 99)) / 99

    with pytest.raises(ValueError, match="expected"):
        evaluate(wrong_shape, DATASET, CLASSES, progress=False)


def test_empty_dataset():
    with pytest.raises(ValueError, match="no samples"):
        evaluate(make_predict_fn(), [], CLASSES, progress=False)


def test_rejects_degenerate_class_list():
    with pytest.raises(ValueError, match="at least 2 classes"):
        evaluate(make_predict_fn(), DATASET, ["france"], progress=False)
