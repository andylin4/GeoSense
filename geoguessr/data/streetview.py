"""Phase 4 groundwork: Street View coverage probing and generation labels.

Splits cleanly into the free part and the paid part:

* **Metadata is free.** ``/streetview/metadata`` returns whether a panorama
  exists, its id, its capture date, and whether Google or a user shot it. No
  charge, and it takes no ``pitch`` -- so everything in this module is
  independent of the unresolved framing question.
* **Imagery costs money.** Actually downloading pixels is the Static API, which
  bills per request. That lives elsewhere and is deliberately not implemented
  until the pitch decision is made, because the wrong framing makes the whole
  scrape unusable.

Probe locations come from the OSV-5M manifest rather than random points on
land. Those coordinates are known roadsides inside the 100 target countries, so
the hit rate is high and the country label comes free -- random land points
would burn quota discovering that deserts have no Street View.

Generation labels are **derived, not measured**. Google publishes no generation
field; the boundaries below come from public rollout timelines and are
approximate, with real overlap at the edges. That is exactly why the design
calls for hand-labelling a validation subset -- treat these as a prior to be
checked, not ground truth.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import polars as pl

__all__ = [
    "GENERATIONS",
    "PanoramaMeta",
    "derive_generation",
    "sample_probe_locations",
    "StreetViewProbe",
    "coverage_counts",
]

METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"

# Ordered oldest to newest. "unknown" absorbs panoramas with no usable date.
GENERATIONS = ["gen1", "gen2", "gen3", "gen4", "unknown"]

# Approximate global rollout boundaries, as (exclusive upper bound year-month).
# Country rollouts differ by a year or more in places; see module docstring.
_GENERATION_BOUNDARIES = [
    ("gen1", (2008, 6)),
    ("gen2", (2012, 1)),
    ("gen3", (2017, 6)),
]


@dataclass(frozen=True)
class PanoramaMeta:
    """One metadata probe result."""

    lat: float
    lon: float
    status: str
    pano_id: str | None = None
    date: str | None = None  # "YYYY-MM"
    copyright: str | None = None

    @property
    def exists(self) -> bool:
        return self.status == "OK"

    @property
    def is_google(self) -> bool:
        """True for official Google coverage, false for user photospheres.

        Photospheres have none of the meta signature the design cares about --
        no car, no consistent camera, no blur -- so they are noise for the meta
        head and must be filtered out.
        """
        return bool(self.copyright and "Google" in self.copyright)

    @property
    def generation(self) -> str:
        return derive_generation(self.date)


def derive_generation(date: str | None) -> str:
    """Map a ``YYYY-MM`` capture date to a camera generation label.

    Returns ``"unknown"`` for missing or unparseable dates rather than
    guessing, so downstream counts can exclude them explicitly.
    """
    if not date:
        return "unknown"

    parts = str(date).strip().split("-")
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
    except (ValueError, IndexError):
        return "unknown"

    if not 2000 <= year <= 2100:
        return "unknown"

    for name, (boundary_year, boundary_month) in _GENERATION_BOUNDARIES:
        if (year, month) < (boundary_year, boundary_month):
            return name
    return "gen4"


def sample_probe_locations(
    manifest: pl.DataFrame,
    *,
    per_country: int = 40,
    seed: int = 0,
) -> pl.DataFrame:
    """Pick coordinates to probe, balanced across countries.

    Balanced rather than natural: the coverage table needs a usable estimate of
    P(generation | country) for *every* country, and the natural distribution
    would spend most of the budget re-measuring the United States.
    """
    if manifest.height == 0:
        raise ValueError("cannot sample probe locations from an empty manifest")

    return (
        manifest.select("image_id", "lat", "lon", "country")
        .sort("image_id")
        .sample(fraction=1.0, shuffle=True, seed=seed)
        .group_by("country", maintain_order=True)
        .head(per_country)
        .sort(["country", "image_id"])
    )


class StreetViewProbe:
    """Client for the free metadata endpoint.

    The HTTP call is injected so this is testable without a key or a network,
    and so a caller can swap in their own retry/caching layer.
    """

    def __init__(
        self,
        api_key: str,
        *,
        radius: int = 50,
        source: str = "outdoor",
        fetcher: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        min_interval: float = 0.0,
    ):
        if not api_key:
            raise ValueError("an API key is required, even for free metadata calls")
        self.api_key = api_key
        self.radius = radius
        self.source = source
        self._fetch = fetcher or _default_fetcher
        self.min_interval = min_interval

        self.calls = 0
        self._last_call = 0.0

    def probe(self, lat: float, lon: float) -> PanoramaMeta:
        """Look up whether a panorama exists near a coordinate. Free."""
        if self.min_interval:
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)

        params = {
            "location": f"{lat},{lon}",
            "key": self.api_key,
            "radius": self.radius,
            "source": self.source,
        }
        payload = self._fetch(METADATA_URL, params)
        self.calls += 1
        self._last_call = time.monotonic()

        return PanoramaMeta(
            lat=lat,
            lon=lon,
            status=payload.get("status", "UNKNOWN"),
            pano_id=payload.get("pano_id"),
            date=payload.get("date"),
            copyright=payload.get("copyright"),
        )

    def probe_many(
        self, locations: Iterable[tuple[float, float]], *, on_result=None
    ) -> list[PanoramaMeta]:
        results = []
        for lat, lon in locations:
            meta = self.probe(lat, lon)
            results.append(meta)
            if on_result is not None:
                on_result(meta)
        return results


def _default_fetcher(url: str, params: dict[str, Any]) -> dict[str, Any]:
    import urllib.parse
    import urllib.request
    import json as _json

    query = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{url}?{query}", timeout=20) as response:
        return _json.loads(response.read().decode())


def coverage_counts(
    results: Iterable[PanoramaMeta],
    countries: Iterable[str],
    *,
    google_only: bool = True,
) -> dict[str, dict[str, float]]:
    """Tally ``{country: {generation: count}}`` for :class:`CoverageTable`.

    ``results`` and ``countries`` are zipped positionally -- the probe does not
    know what country a coordinate was in, but the manifest that produced it
    does.
    """
    counts: dict[str, dict[str, float]] = {}

    for meta, country in zip(results, countries):
        if not meta.exists:
            continue
        if google_only and not meta.is_google:
            continue
        bucket = counts.setdefault(country, {})
        bucket[meta.generation] = bucket.get(meta.generation, 0.0) + 1.0

    return counts
