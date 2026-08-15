"""Phase 6b: meta fusion.

Where the two tracks meet. The country head reports what the *place* looks
like; the meta head reports how the *imagery* was captured. Camera generation
is close to a hard constraint on geography -- if a country was only ever driven
with Gen 4, a confident Gen 2 reading eliminates it -- so a large fraction of
the world can drop out before vegetation is considered at all.

The design's rule, kept deliberately simple for v1:

    s'_c = s_c * P(g_hat | c)

Learned fusion is a later refinement. This is a lookup table and a multiply,
which means it is inspectable: you can always answer "why did Poland fall off"
by reading one number.

Two properties this module is careful about:

* **Uncertain meta must not destroy information.** If the meta head is
  unconfident, fusion should approach a no-op rather than confidently deleting
  countries. That is handled by mixing toward a uniform likelihood in
  proportion to meta confidence.
* **Never assign zero to an unseen combination.** A country with no observed
  panoramas of some generation gets a small floor, not 0. A single hard zero is
  unrecoverable no matter how certain the country head is, and coverage tables
  built from a few thousand scraped points are full of accidental gaps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

__all__ = ["CoverageTable", "fuse", "fuse_predict_fn"]

# Probability floor for (country, generation) pairs never observed. Small
# enough to strongly penalize, large enough to stay recoverable.
DEFAULT_FLOOR = 1e-3


@dataclass
class CoverageTable:
    """P(camera generation | country), estimated from scraped metadata.

    Rows are countries, columns are generations, and every row sums to 1.
    """

    class_names: list[str]
    generations: list[str]
    matrix: np.ndarray  # (n_countries, n_generations)

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=np.float64)
        expected = (len(self.class_names), len(self.generations))
        if self.matrix.shape != expected:
            raise ValueError(
                f"matrix is {self.matrix.shape}, expected {expected} "
                f"({len(self.class_names)} countries x {len(self.generations)} gens)"
            )
        if (self.matrix < 0).any():
            raise ValueError("coverage probabilities cannot be negative")

    @classmethod
    def from_counts(
        cls,
        counts: Mapping[str, Mapping[str, float]],
        class_names: Sequence[str],
        generations: Sequence[str],
        *,
        floor: float = DEFAULT_FLOOR,
    ) -> CoverageTable:
        """Build from observed ``{country: {generation: count}}`` tallies.

        This is what the Phase 4 scrape produces directly: every probed
        panorama contributes one (country, derived generation) pair.

        Countries with no observations at all fall back to a uniform row --
        fusion then leaves them untouched rather than guessing.
        """
        class_names = list(class_names)
        generations = list(generations)
        matrix = np.zeros((len(class_names), len(generations)), dtype=np.float64)

        for i, country in enumerate(class_names):
            row = counts.get(country, {})
            observed = np.array([float(row.get(g, 0.0)) for g in generations])

            if observed.sum() <= 0:
                matrix[i] = 1.0 / len(generations)  # no evidence -> no opinion
            else:
                probabilities = observed / observed.sum()
                # Floor unseen combinations, then renormalize.
                probabilities = np.maximum(probabilities, floor)
                matrix[i] = probabilities / probabilities.sum()

        return cls(class_names, generations, matrix)

    def likelihood(self, generation_probs: np.ndarray) -> np.ndarray:
        """P(observed generation evidence | country) for every country.

        Takes a *distribution* over generations rather than a hard label, so a
        hedging meta head contributes proportionally instead of all-or-nothing.
        """
        generation_probs = np.asarray(generation_probs, dtype=np.float64)
        if generation_probs.shape[-1] != len(self.generations):
            raise ValueError(
                f"expected {len(self.generations)} generation scores, "
                f"got {generation_probs.shape[-1]}"
            )
        return generation_probs @ self.matrix.T  # (..., n_countries)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "class_names": self.class_names,
                    "generations": self.generations,
                    "matrix": self.matrix.tolist(),
                },
                indent=2,
            )
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> CoverageTable:
        payload = json.loads(Path(path).read_text())
        return cls(payload["class_names"], payload["generations"],
                   np.asarray(payload["matrix"]))


def fuse(
    country_probs: np.ndarray,
    generation_probs: np.ndarray,
    table: CoverageTable,
    *,
    meta_confidence: float = 1.0,
) -> np.ndarray:
    """Reweight country scores by camera-generation evidence.

    Args:
        country_probs: ``(N, n_countries)`` from the country head.
        generation_probs: ``(N, n_generations)`` from the meta head.
        table: the coverage lookup.
        meta_confidence: how much to trust the meta head, in [0, 1]. At 0 this
            is exactly the identity -- the country head's output is returned
            untouched. At 1 the full likelihood is applied. Anything in between
            interpolates the likelihood toward uniform, which is the safe
            direction: an unsure meta head weakens its own vote rather than
            deleting countries.

    Returns:
        ``(N, n_countries)`` renormalized probabilities.
    """
    if not 0.0 <= meta_confidence <= 1.0:
        raise ValueError(f"meta_confidence must be in [0, 1], got {meta_confidence}")

    country_probs = np.atleast_2d(np.asarray(country_probs, dtype=np.float64))
    if country_probs.shape[-1] != len(table.class_names):
        raise ValueError(
            f"country_probs has {country_probs.shape[-1]} classes but the "
            f"coverage table has {len(table.class_names)}"
        )

    likelihood = np.atleast_2d(table.likelihood(generation_probs))

    # Interpolate toward a flat likelihood as confidence falls.
    uniform = np.full_like(likelihood, 1.0 / likelihood.shape[-1])
    effective = meta_confidence * likelihood + (1.0 - meta_confidence) * uniform

    fused = country_probs * effective
    totals = fused.sum(axis=-1, keepdims=True)

    # If meta annihilated everything, fall back to the country head rather than
    # emitting NaNs. Better to ignore the meta signal than return nothing.
    degenerate = (totals <= 0).ravel()
    if degenerate.any():
        fused[degenerate] = country_probs[degenerate]
        totals = fused.sum(axis=-1, keepdims=True)

    return fused / totals


def fuse_predict_fn(country_fn, meta_fn, table: CoverageTable, *,
                    meta_confidence: float = 1.0):
    """Compose a country predict_fn and a meta predict_fn into one.

    The result is itself a predict_fn, so the fused model evaluates through the
    same harness as every other stage with no special-casing.
    """

    def fn(items):
        return fuse(country_fn(items), meta_fn(items), table,
                    meta_confidence=meta_confidence)

    return fn
