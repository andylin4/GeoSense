"""Phase 1: the data layer.

Turns raw OSV-5M metadata into ``manifest.parquet`` -- the contract with Phase
2. Phase 2 never learns where the images came from, so swapping data sources
later means rewriting only this module.

Two deviations from the design doc, both discovered on 2026-08-13:

* **No Natural Earth join.** OSV-5M already ships ISO alpha-2 country codes, so
  the point-in-polygon step (and DuckDB's spatial extension) is unnecessary. We
  accept upstream's labels, which is the same posture the review settled on for
  disputed borders.
* **OTHER rows are dropped, not bucketed.** See :mod:`geoguessr.data.countries`.

The manifest stores the country *code*, never an integer class index. Label
schemes change; embeddings do not. Deriving indices at train time is what keeps
relabeling free.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .countries import OTHER, STREET_VIEW_COUNTRIES, to_code

__all__ = [
    "MANIFEST_COLUMNS",
    "build_manifest",
    "subsample",
    "attach_paths",
    "write_manifest",
    "read_manifest",
    "manifest_stats",
]

MANIFEST_COLUMNS = ["image_id", "lat", "lon", "country", "split", "path"]


def build_manifest(
    csv_path: str | Path,
    *,
    split: str,
    limit: int | None = None,
    strategy: str = "random",
    max_per_country: int | None = None,
    seed: int = 0,
) -> pl.DataFrame:
    """Read an OSV-5M CSV and emit manifest rows for in-coverage countries.

    Args:
        csv_path: OSV-5M ``train.csv`` or ``test.csv``.
        split: value written to the ``split`` column, e.g. ``"train"``.
        limit: total rows to keep. ``None`` keeps everything.
        strategy: ``"random"`` preserves OSV-5M's natural class distribution,
            which is what decision #5 assumes -- imbalance is corrected by
            loss reweighting at train time, not by resampling here.
            ``"stratified"`` instead draws as evenly across countries as the
            data allows, which is useful for a small pipeline-proving subset
            where random sampling would leave rare countries empty.
        max_per_country: hard cap per country, applied before ``limit``.
        seed: sampling seed, so a manifest is reproducible.

    Returns:
        A DataFrame with :data:`MANIFEST_COLUMNS`. ``path`` starts null and is
        filled in by :func:`attach_paths` once images are on disk.
    """
    if strategy not in ("random", "stratified"):
        raise ValueError(f"unknown strategy {strategy!r}")

    frame = pl.read_csv(csv_path, infer_schema_length=10_000)

    missing = {"id", "latitude", "longitude", "country"} - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    frame = (
        frame.select(
            pl.col("id").cast(pl.Int64).alias("image_id"),
            pl.col("latitude").cast(pl.Float64).alias("lat"),
            pl.col("longitude").cast(pl.Float64).alias("lon"),
            pl.col("country"),
        )
        .drop_nulls(["image_id", "lat", "lon", "country"])
        # maintain_order is load-bearing: without it polars may return rows in a
        # different order per run, which would make seeded sampling below
        # non-reproducible and a manifest impossible to reconstruct.
        .unique(subset=["image_id"], keep="first", maintain_order=True)
    )

    # Canonicalize, then drop everything outside Street View coverage.
    mapping = {c: to_code(c) for c in frame["country"].unique().to_list()}
    frame = (
        frame.with_columns(
            pl.col("country").replace_strict(mapping, default=OTHER).alias("country")
        )
        .filter(pl.col("country") != OTHER)
    )

    frame = frame.with_columns(
        pl.lit(split).alias("split"),
        pl.lit(None, dtype=pl.String).alias("path"),
    ).select(MANIFEST_COLUMNS)

    return subsample(
        frame,
        limit=limit,
        strategy=strategy,
        max_per_country=max_per_country,
        seed=seed,
    )


def subsample(
    manifest: pl.DataFrame,
    *,
    limit: int | None = None,
    strategy: str = "random",
    max_per_country: int | None = None,
    seed: int = 0,
) -> pl.DataFrame:
    """Reduce a manifest, preserving the column contract and id ordering.

    Split out from :func:`build_manifest` because the useful order of
    operations is usually build -> attach_paths -> subsample: sampling before
    knowing which images actually exist on disk wastes most of the budget on
    rows that get dropped.
    """
    if strategy not in ("random", "stratified"):
        raise ValueError(f"unknown strategy {strategy!r}")

    # Sample from a canonical row order so the result depends only on the set of
    # rows and the seed -- never on how the caller happened to order them. This
    # is what makes a manifest reconstructible, and the embedding cache keyed to
    # it trustworthy.
    frame = manifest.sort("image_id")
    if max_per_country is not None:
        frame = (
            frame.sample(fraction=1.0, shuffle=True, seed=seed)
            .group_by("country")
            .head(max_per_country)
        )

    if limit is not None and frame.height > limit:
        if strategy == "random":
            frame = frame.sample(n=limit, shuffle=True, seed=seed)
        else:
            frame = _stratified_sample(frame, limit=limit, seed=seed)

    return frame.select(MANIFEST_COLUMNS).sort("image_id")


def _stratified_sample(frame: pl.DataFrame, *, limit: int, seed: int) -> pl.DataFrame:
    """Draw as evenly across countries as the data allows.

    Countries with fewer rows than the even quota contribute everything they
    have, and the shortfall is redistributed to countries that still have rows
    left, so the result hits ``limit`` whenever the data can support it.
    """
    shuffled = frame.sample(fraction=1.0, shuffle=True, seed=seed)
    # sorted() is load-bearing: polars' group_by does not guarantee row order,
    # and the quota loop below is order-dependent (whoever is visited first
    # claims the remainder). Without a deterministic order the same seed picks
    # different rows per process, which silently invalidates the embedding cache.
    counts = dict(sorted(shuffled.group_by("country").len().iter_rows()))

    quota: dict[str, int] = {c: 0 for c in counts}
    remaining = min(limit, shuffled.height)

    # Repeatedly hand out an equal share to whoever still has capacity.
    while remaining > 0:
        hungry = sorted(c for c in counts if quota[c] < counts[c])
        if not hungry:
            break
        share = max(1, remaining // len(hungry))
        for country in hungry:
            if remaining <= 0:
                break
            take = min(share, counts[country] - quota[country], remaining)
            quota[country] += take
            remaining -= take

    keep = pl.concat(
        [
            group.head(quota[country])
            for (country,), group in shuffled.group_by(["country"], maintain_order=True)
            if quota.get(country, 0) > 0
        ]
    )
    return keep


def attach_paths(
    manifest: pl.DataFrame, image_root: str | Path, *, require_all: bool = False
) -> pl.DataFrame:
    """Fill the ``path`` column by scanning ``image_root`` for ``<id>.jpg``.

    Rows with no image on disk are dropped, since Phase 2 cannot embed them.
    Set ``require_all`` to raise instead, which is what you want when a shard
    was supposed to be complete.
    """
    image_root = Path(image_root)
    if not image_root.exists():
        raise FileNotFoundError(f"image root does not exist: {image_root}")

    found = {
        int(p.stem): str(p)
        for p in image_root.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.stem.isdigit()
    }

    resolved = manifest.with_columns(
        pl.col("image_id")
        .replace_strict(found, default=None, return_dtype=pl.String)
        .alias("path")
    )

    missing = resolved.filter(pl.col("path").is_null()).height
    if missing and require_all:
        raise ValueError(f"{missing} manifest rows have no image under {image_root}")

    return resolved.filter(pl.col("path").is_not_null())


def write_manifest(manifest: pl.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_parquet(path)
    return path


def read_manifest(path: str | Path) -> pl.DataFrame:
    manifest = pl.read_parquet(path)
    missing = set(MANIFEST_COLUMNS) - set(manifest.columns)
    if missing:
        raise ValueError(f"{path} is not a manifest; missing {sorted(missing)}")
    return manifest


def manifest_stats(manifest: pl.DataFrame) -> dict[str, object]:
    """Summary used to sanity-check a manifest before spending GPU time on it."""
    counts = manifest.group_by("country").len().sort("len", descending=True)
    total = manifest.height
    present = counts.height

    return {
        "rows": total,
        "countries_present": present,
        "countries_missing": len(STREET_VIEW_COUNTRIES) - present,
        "with_path": int(manifest["path"].is_not_null().sum()),
        "largest_share": (counts["len"][0] / total) if total else 0.0,
        "top": counts.head(10).rows(),
        "smallest": counts.tail(10).rows(),
    }
