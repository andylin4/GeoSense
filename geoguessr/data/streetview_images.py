"""The paid half of Phase 4: fetching panorama imagery.

This is the only module in the project that can spend money, so it is built
defensively:

* ``max_requests`` is **required**. There is no default. A runaway loop over a
  5-million-row manifest cannot happen because the object refuses to exist
  without an explicit ceiling.
* Every request is counted, and the ceiling raises :class:`BudgetExhausted`
  rather than silently continuing.
* Already-downloaded panoramas are skipped, so a re-run costs nothing.

**Fetch by ``pano_id``, never by lat/lon.** A coordinate request can return a
different (usually newer) panorama than the metadata probe described, which
would pair a 2019-derived generation label with a 2023 image. That is silent
label corruption -- invisible to every test, fatal to the meta head.

Framing (``pitch``, ``heading``, ``fov``) is deliberately left as configuration
rather than hardcoded. It is the unresolved design question: images must match
what the model sees at inference, and the wrong choice makes the whole scrape
unusable. :data:`HORIZON` matches gameplay framing; :data:`DOWNWARD` shows the
car, blur, and antenna but looks nothing like a screenshot.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

__all__ = [
    "FetchConfig",
    "HORIZON",
    "DOWNWARD",
    "BudgetExhausted",
    "StaticImageFetcher",
    "USD_PER_1000",
]

STATIC_URL = "https://maps.googleapis.com/maps/api/streetview"

# Street View Static API list price. Google's monthly allowance may absorb
# some or all of this; treat estimates here as an upper bound.
USD_PER_1000 = 7.0


class BudgetExhausted(RuntimeError):
    """Raised when the configured request ceiling is reached."""


@dataclass(frozen=True)
class FetchConfig:
    """Framing parameters. These decide whether the scrape is usable at all."""

    size: str = "640x640"
    pitch: float = 0.0
    heading: float | None = None  # None lets Google pick the default view
    fov: float = 90.0
    note: str = ""

    def params(self) -> dict[str, object]:
        params: dict[str, object] = {
            "size": self.size,
            "pitch": self.pitch,
            "fov": self.fov,
            "return_error_code": "true",
        }
        if self.heading is not None:
            params["heading"] = self.heading
        return params

    def with_heading(self, heading: float) -> FetchConfig:
        return replace(self, heading=heading)


HORIZON = FetchConfig(
    pitch=0.0,
    note=(
        "Gameplay framing. Matches what a player's screenshot looks like, so "
        "the meta head trains and infers on the same distribution. Camera "
        "generation is learnable from resolution and lens character; car blur, "
        "antenna, and snorkel are NOT visible and cannot be learned."
    ),
)

DOWNWARD = FetchConfig(
    pitch=-90.0,
    note=(
        "Looks straight down at the vehicle. Car blur, antenna, and snorkel "
        "are clearly visible -- but no player screenshot is framed this way, "
        "so a head trained here cannot use those cues at inference."
    ),
)


class StaticImageFetcher:
    """Downloads panorama images, with a hard ceiling on spend.

    The HTTP call is injected so this is fully testable without a key, a
    network, or a cent.
    """

    def __init__(
        self,
        api_key: str,
        *,
        max_requests: int,
        config: FetchConfig = HORIZON,
        fetcher: Callable[[str, dict], bytes] | None = None,
        min_interval: float = 0.0,
    ):
        if not api_key:
            raise ValueError("an API key is required")
        if max_requests is None or max_requests <= 0:
            raise ValueError(
                "max_requests must be a positive integer -- this is the spend "
                "ceiling and is deliberately mandatory"
            )

        self.api_key = api_key
        self.max_requests = int(max_requests)
        self.config = config
        self.min_interval = min_interval
        self._fetch = fetcher or _default_image_fetcher

        self.requests = 0
        self.skipped = 0
        self._last_call = 0.0

    @property
    def remaining(self) -> int:
        return max(0, self.max_requests - self.requests)

    @property
    def estimated_cost_usd(self) -> float:
        """List-price cost of what has been spent so far."""
        return self.requests / 1000.0 * USD_PER_1000

    @property
    def ceiling_cost_usd(self) -> float:
        """Worst case if the ceiling is reached."""
        return self.max_requests / 1000.0 * USD_PER_1000

    def fetch(self, pano_id: str) -> bytes:
        """Download one panorama by id. Costs one request."""
        if not pano_id:
            raise ValueError("pano_id is required; never fetch by coordinate")
        if self.requests >= self.max_requests:
            raise BudgetExhausted(
                f"reached the {self.max_requests}-request ceiling "
                f"(~${self.estimated_cost_usd:.2f} at list price)"
            )

        if self.min_interval:
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)

        params = {**self.config.params(), "pano": pano_id, "key": self.api_key}
        payload = self._fetch(STATIC_URL, params)

        self.requests += 1
        self._last_call = time.monotonic()
        return payload

    def fetch_to_dir(
        self,
        pano_ids: Iterable[str],
        out_dir: str | Path,
        *,
        on_progress: Callable[[str, str], None] | None = None,
    ) -> dict[str, Path]:
        """Download many panoramas, skipping any already on disk.

        Returns ``{pano_id: path}`` for everything available locally after the
        call, whether newly fetched or already present. Stops cleanly at the
        ceiling rather than raising, so a partial scrape is still usable.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved: dict[str, Path] = {}

        for pano_id in pano_ids:
            destination = out_dir / f"{pano_id}.jpg"

            if destination.exists():
                self.skipped += 1
                saved[pano_id] = destination
                if on_progress:
                    on_progress(pano_id, "skipped")
                continue

            if self.requests >= self.max_requests:
                if on_progress:
                    on_progress(pano_id, "budget-exhausted")
                break

            try:
                destination.write_bytes(self.fetch(pano_id))
            except BudgetExhausted:
                break
            except Exception:
                if on_progress:
                    on_progress(pano_id, "error")
                continue

            saved[pano_id] = destination
            if on_progress:
                on_progress(pano_id, "fetched")

        return saved

    def summary(self) -> str:
        return (
            f"{self.requests} requests (~${self.estimated_cost_usd:.2f}), "
            f"{self.skipped} skipped, {self.remaining} left of "
            f"{self.max_requests} (ceiling ~${self.ceiling_cost_usd:.2f})"
        )


def _default_image_fetcher(url: str, params: dict) -> bytes:
    import urllib.parse
    import urllib.request

    query = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{url}?{query}", timeout=30) as response:
        return response.read()
