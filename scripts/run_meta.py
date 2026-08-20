"""End-to-end meta track: probe -> coverage table -> fetch -> diagnose -> head.

The free steps and the paid step are deliberately separated, and the paid step
never runs unless you ask for it by name.

    # free: probe coverage, build the fusion table, report what images'd cost
    uv run python scripts/run_meta.py --api-key "$KEY" --dry-run

    # paid: also download imagery, with a hard ceiling
    uv run python scripts/run_meta.py --api-key "$KEY" --max-requests 2000

Order matters. ``--dry-run`` already produces the coverage table that Phase 6b
fusion needs, entirely for free -- so run it first and confirm the hit rate and
generation spread look sane before spending anything.

Framing (pitch/heading/fov) is the unresolved design question. ``--framing
horizon`` matches gameplay; ``--framing downward`` shows the car but does not
resemble a screenshot. Choose before spending: re-scraping costs again.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ARTIFACTS = Path("artifacts")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--api-key", required=True,
                        help="Google Maps Platform key (metadata calls are free)")
    parser.add_argument("--manifest", default=None,
                        help="manifest parquet to draw probe locations from")
    parser.add_argument("--tag", default="20000", help="artifact tag of the manifest")
    parser.add_argument("--per-country", type=int, default=40,
                        help="probe locations per country (free calls)")
    parser.add_argument("--dry-run", action="store_true",
                        help="probe and build the coverage table only; spend nothing")
    parser.add_argument("--max-requests", type=int, default=None,
                        help="HARD ceiling on paid image requests. Required to "
                             "download anything.")
    parser.add_argument("--framing", default="horizon", choices=["horizon", "downward"])
    parser.add_argument("--image-dir", default="data_raw/streetview")
    parser.add_argument("--min-interval", type=float, default=0.05,
                        help="seconds between API calls, to stay under rate limits")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import numpy as np

    from geoguessr.calibrate.fusion import CoverageTable
    from geoguessr.data.countries import class_names
    from geoguessr.data.manifest import read_manifest
    from geoguessr.data.streetview import (
        GENERATIONS,
        StreetViewProbe,
        coverage_counts,
        sample_probe_locations,
    )
    from geoguessr.data.streetview_images import (
        DOWNWARD,
        HORIZON,
        StaticImageFetcher,
    )

    ARTIFACTS.mkdir(exist_ok=True)
    manifest_path = Path(args.manifest or ARTIFACTS / f"manifest_{args.tag}.parquet")
    if not manifest_path.exists():
        print(f"error: no manifest at {manifest_path}. Run run_pipeline.py first.",
              file=sys.stderr)
        return 1

    # --- Phase 4a: free coverage probing ------------------------------------
    manifest = read_manifest(manifest_path)
    locations = sample_probe_locations(manifest, per_country=args.per_country,
                                       seed=args.seed)
    log(f"probing {locations.height} locations across "
        f"{locations['country'].n_unique()} countries (free)")

    probe = StreetViewProbe(args.api_key, min_interval=args.min_interval)
    results, countries = [], []

    for row in locations.iter_rows(named=True):
        meta = probe.probe(row["lat"], row["lon"])
        results.append(meta)
        countries.append(row["country"])
        if probe.calls % 100 == 0:
            hits = sum(1 for m in results if m.exists)
            log(f"  {probe.calls}/{locations.height} probed, {hits} hits")

    hits = [m for m in results if m.exists]
    google = [m for m in hits if m.is_google]
    log(f"probed {probe.calls}: {len(hits)} panoramas, {len(google)} official Google")

    if not google:
        print("error: no official Google coverage found. Check the key and the "
              "manifest's countries.", file=sys.stderr)
        return 1

    # --- Phase 6b input: the coverage table ---------------------------------
    counts = coverage_counts(results, countries)
    table = CoverageTable.from_counts(counts, class_names(), GENERATIONS)
    table.save(ARTIFACTS / "coverage_table.json")

    spread: dict[str, int] = {}
    for meta in google:
        spread[meta.generation] = spread.get(meta.generation, 0) + 1
    log(f"generation spread: {dict(sorted(spread.items()))}")
    log(f"wrote {ARTIFACTS / 'coverage_table.json'} "
        f"({len(counts)} countries with data)")

    # --- Phase 4b: the paid part --------------------------------------------
    framing = HORIZON if args.framing == "horizon" else DOWNWARD

    if args.dry_run or not args.max_requests:
        would_fetch = len(google)
        cost = would_fetch / 1000 * 7.0
        print()
        print("=" * 66)
        print(f"DRY RUN -- nothing was downloaded and nothing was charged.")
        print(f"  {would_fetch} panoramas are available to fetch")
        print(f"  at list price that is ~${cost:.2f} (Google's monthly")
        print(f"  allowance may absorb some or all of it)")
        print(f"  framing would be: {args.framing} (pitch {framing.pitch})")
        print()
        print(f"  to proceed:  --max-requests {would_fetch} --framing {args.framing}")
        print("=" * 66)
        print(framing.note)
        return 0

    fetcher = StaticImageFetcher(args.api_key, max_requests=args.max_requests,
                                 config=framing, min_interval=args.min_interval)
    log(f"fetching up to {args.max_requests} images "
        f"(ceiling ~${fetcher.ceiling_cost_usd:.2f}), framing={args.framing}")

    pano_ids = [m.pano_id for m in google if m.pano_id]
    by_pano = {m.pano_id: (m, c) for m, c in zip(results, countries) if m.exists}

    saved = fetcher.fetch_to_dir(pano_ids, args.image_dir)
    log(f"  {fetcher.summary()}")

    if not saved:
        print("error: nothing downloaded", file=sys.stderr)
        return 1

    # --- Phase 5: embed, diagnose, train ------------------------------------
    from PIL import Image

    from geoguessr.data.crop import FULL_FRAME
    from geoguessr.embed.backbone import Backbone
    from geoguessr.train.meta import diagnose_meta_signal, train_meta_head

    log("embedding scraped panoramas")
    backbone = Backbone()

    paths = sorted(saved.items())
    vectors, labels = [], []
    gen_index = {g: i for i, g in enumerate(GENERATIONS)}

    for start in range(0, len(paths), 32):
        chunk = paths[start:start + 32]
        images = [Image.open(p).convert("RGB") for _, p in chunk]
        vectors.append(backbone.encode_images([FULL_FRAME.apply(i) for i in images]))
        labels += [gen_index[by_pano[pid][0].generation] for pid, _ in chunk]

    embeddings = np.vstack(vectors)
    labels = np.asarray(labels)
    log(f"  {embeddings.shape[0]} x {embeddings.shape[1]}")

    # Can the meta head share the embedding space at all?
    print()
    print("=" * 66)
    print("DOES CLIP RETAIN CAMERA GENERATION?")
    print("=" * 66)
    diagnosis = diagnose_meta_signal(embeddings, labels)
    print(diagnosis.summary())
    print("=" * 66)

    if not diagnosis.signal_present:
        print("\nStopping before training a head on embeddings that do not "
              "carry the signal. Build a pixel CNN instead, or collect more "
              "data if the sample was small.")
        return 0

    head = train_meta_head(embeddings, labels)
    head.save(ARTIFACTS / "meta_head.joblib")
    log(f"wrote {ARTIFACTS / 'meta_head.joblib'}")
    log("fusion is ready: coverage_table.json + meta_head.joblib")
    return 0


if __name__ == "__main__":
    sys.exit(main())
