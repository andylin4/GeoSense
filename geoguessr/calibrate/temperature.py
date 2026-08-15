"""Phase 6: temperature scaling.

Raw softmax outputs are systematically overconfident. Temperature scaling fits
a single scalar `T` on held-out data by minimizing negative log likelihood:

    p_hat_i = exp(z_i / T) / sum_j exp(z_j / T)

One parameter, fits in seconds, and it cannot change which class wins -- it
only reshapes how peaked the distribution is. That is exactly what makes
"87% Poland" mean the model is right 87% of the time rather than being a
decorative number.

Per the design, this is not a phase you finish. It is a step appended to the
end of every training run, because any change to the head invalidates the
previous `T`.

Two cautions specific to this project:

* Fit on a *held-out* split. Fitting `T` on training predictions gives an
  overconfident model an artificially low temperature.
* The production `T` must eventually be fit on real game screenshots, not
  OSV-5M. Calibration is distribution-specific: a temperature fitted on
  Mapillary photos does not describe how the model behaves on screenshots.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..eval.metrics import expected_calibration_error

__all__ = ["TemperatureScaler", "fit_temperature", "probs_to_logits"]

_EPS = 1e-12


def probs_to_logits(probs: np.ndarray) -> np.ndarray:
    """Recover logits from a probability matrix.

    ``log(p)`` differs from the original logits by a per-row constant, and
    softmax is invariant to per-row constants, so this round-trips exactly at
    ``T = 1``. That lets us temperature-scale the output of models like
    sklearn's ``LogisticRegression`` that only expose ``predict_proba``.
    """
    return np.log(np.clip(np.asarray(probs, dtype=np.float64), _EPS, None))


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def _nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    probs = _softmax(logits, temperature)
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], _EPS, None)).mean())


def fit_temperature(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    bounds: tuple[float, float] = (0.05, 20.0),
) -> float:
    """Find the temperature minimizing NLL on held-out predictions.

    ``T > 1`` softens an overconfident model (the usual case); ``T < 1``
    sharpens an underconfident one.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if probs.ndim != 2:
        raise ValueError(f"probs must be 2-D (N, C), got {probs.shape}")
    if probs.shape[0] != labels.shape[0]:
        raise ValueError(f"{probs.shape[0]} rows but {labels.shape[0]} labels")
    if probs.shape[0] == 0:
        raise ValueError("cannot fit a temperature on an empty set")

    logits = probs_to_logits(probs)

    from scipy.optimize import minimize_scalar

    result = minimize_scalar(
        lambda t: _nll(logits, labels, t),
        bounds=bounds,
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(result.x)


@dataclass
class TemperatureScaler:
    """A fitted temperature plus the evidence that fitting it helped."""

    temperature: float
    n_fit: int = 0
    ece_before: float | None = None
    ece_after: float | None = None
    fit_on: str = ""

    @classmethod
    def fit(
        cls,
        probs: np.ndarray,
        labels: np.ndarray,
        *,
        fit_on: str = "",
        n_bins: int = 15,
    ) -> TemperatureScaler:
        temperature = fit_temperature(probs, labels)
        scaler = cls(
            temperature=temperature,
            n_fit=int(np.asarray(probs).shape[0]),
            ece_before=expected_calibration_error(probs, labels, n_bins=n_bins),
            fit_on=fit_on,
        )
        scaler.ece_after = expected_calibration_error(
            scaler.apply(probs), labels, n_bins=n_bins
        )
        return scaler

    def apply(self, probs: np.ndarray) -> np.ndarray:
        """Rescale a probability matrix. Argmax is provably unchanged."""
        return _softmax(probs_to_logits(probs), self.temperature)

    def wrap(self, predict_fn):
        """Wrap a predict_fn so it returns calibrated probabilities.

        This is how calibration reaches the harness and the live path without
        either of them knowing it happened.
        """

        def calibrated(items):
            return self.apply(predict_fn(items))

        return calibrated

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "temperature": self.temperature,
                    "n_fit": self.n_fit,
                    "ece_before": self.ece_before,
                    "ece_after": self.ece_after,
                    "fit_on": self.fit_on,
                },
                indent=2,
            )
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> TemperatureScaler:
        return cls(**json.loads(Path(path).read_text()))

    def summary(self) -> str:
        direction = (
            "softening (model was overconfident)"
            if self.temperature > 1
            else "sharpening (model was underconfident)"
        )
        lines = [f"T = {self.temperature:.4f}  {direction}"]
        if self.ece_before is not None and self.ece_after is not None:
            lines.append(
                f"ECE {self.ece_before:.4f} -> {self.ece_after:.4f} "
                f"on {self.n_fit} held-out samples"
                + (f" ({self.fit_on})" if self.fit_on else "")
            )
        return "\n".join(lines)
