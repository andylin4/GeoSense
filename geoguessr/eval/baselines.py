"""Baseline predictors, in the order the design says to beat them.

    random weighted by coverage  ->  zero-shot StreetCLIP  ->  linear probe -> ...

Each is a ``predict_fn`` suitable for :func:`geoguessr.eval.harness.evaluate`.
The trained heads land here later as further predict_fns; the harness never
learns which one it is holding.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ..data.countries import display_name

if TYPE_CHECKING:
    from ..embed.backbone import Backbone

__all__ = [
    "PROMPT_TEMPLATES",
    "zero_shot_predict_fn",
    "prior_predict_fn",
    "uniform_predict_fn",
]

# StreetCLIP was trained with captions of roughly this shape, so the first
# template is the one its own model card uses. Averaging a few templates is
# standard CLIP practice and reliably beats any single prompt.
PROMPT_TEMPLATES: tuple[str, ...] = (
    "A Street View photo in {country}.",
    "A photo taken in {country}.",
    "A street scene in {country}.",
    "A photo I took while traveling in {country}.",
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def zero_shot_predict_fn(
    backbone: Backbone,
    class_codes: Sequence[str],
    *,
    templates: Sequence[str] = PROMPT_TEMPLATES,
    other_code: str = "XX",
) -> Callable[[list[Any]], np.ndarray]:
    """Zero-shot CLIP classification over country prompts.

    Text embeddings are computed once here, not per batch -- there are only
    ~109 classes and they never change, so this is a fixed matrix the whole
    evaluation reuses.

    The ``OTHER`` class has no sensible prompt (there is no photo of "not a
    Street View country"), so it is assigned a floor score that keeps the
    output a valid distribution without ever letting it win.
    """
    class_codes = list(class_codes)

    prompt_codes = [c for c in class_codes if c != other_code]
    if not prompt_codes:
        raise ValueError("no promptable classes: every code was the OTHER bucket")

    # Average each country's embedding across templates, then renormalize.
    per_template = []
    for template in templates:
        prompts = [template.format(country=display_name(c)) for c in prompt_codes]
        per_template.append(backbone.encode_texts(prompts))

    text_embeds = np.mean(per_template, axis=0)
    text_embeds /= np.linalg.norm(text_embeds, axis=-1, keepdims=True)

    scale = backbone.logit_scale
    other_positions = [i for i, c in enumerate(class_codes) if c == other_code]
    prompt_positions = [i for i, c in enumerate(class_codes) if c != other_code]

    def predict_fn(images: list[Any]) -> np.ndarray:
        image_embeds = backbone.encode_images(images)  # already L2-normalized
        similarity = image_embeds @ text_embeds.T  # cosine, in [-1, 1]

        logits = np.full((len(images), len(class_codes)), -np.inf, dtype=np.float64)
        logits[:, prompt_positions] = similarity * scale
        if other_positions:
            # Floor: strictly below every real class, so OTHER never wins but
            # still receives non-zero mass.
            floor = logits[:, prompt_positions].min(axis=1, keepdims=True) - 1.0
            logits[:, other_positions] = floor

        return _softmax(logits)

    return predict_fn


def prior_predict_fn(
    class_codes: Sequence[str], weights: dict[str, float] | None = None
) -> Callable[[list[Any]], np.ndarray]:
    """Ignore the image, always return a fixed prior.

    This is the "random weighted by coverage" baseline. It is not a joke
    baseline: on a set where the US is a fifth of the data, always guessing the
    prior scores far above chance, which is exactly why macro accuracy and not
    top-1 is the number that matters.
    """
    class_codes = list(class_codes)
    if weights is None:
        prior = np.full(len(class_codes), 1.0 / len(class_codes))
    else:
        prior = np.array([max(weights.get(c, 0.0), 0.0) for c in class_codes])
        if prior.sum() <= 0:
            raise ValueError("weights sum to zero; cannot form a prior")
        prior = prior / prior.sum()

    def predict_fn(images: list[Any]) -> np.ndarray:
        return np.tile(prior, (len(images), 1))

    return predict_fn


def uniform_predict_fn(class_codes: Sequence[str]) -> Callable[[list[Any]], np.ndarray]:
    """Pure chance. The floor everything else must clear."""
    return prior_predict_fn(class_codes, weights=None)


def coverage_weights_from_labels(labels: Sequence[str]) -> dict[str, float]:
    """Empirical class frequencies, for use as ``prior_predict_fn`` weights.

    Fit this on training labels, never on the eval set -- a prior fit on the
    eval set is leakage and would flatter the baseline.
    """
    counts: dict[str, float] = {}
    for label in labels:
        counts[label] = counts.get(label, 0.0) + 1.0
    return counts
