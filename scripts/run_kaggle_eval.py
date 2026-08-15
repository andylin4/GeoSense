"""Phase 0 real evaluation: score the already-trained OSV-5M heads against
GeoGuessr-50k, actual game screenshots with the HUD cropped out.

This is the number every other eval in this project is explicitly NOT: OSV-5M
train/val share a distribution, so that only proves the pipeline runs. This
dataset is out-of-distribution game screenshots, so it is the first real
measurement of whether the model generalizes.

    uv run python scripts/run_kaggle_eval.py

Requires the OSV-5M dataset pulled via kagglehub beforehand, and a trained
head on disk (default tag 20000: head_20000.joblib / mlp_20000.pt /
temperature*_20000.json -- override with --train-tag for a different scale).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

ARTIFACTS = Path("artifacts")
KAGGLE_ROOT = (
    Path.home()
    / ".cache/kagglehub/datasets/ubitquitin/geolocation-geoguessr-images-50k"
    / "versions/1/compressed_dataset"
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-tag", default="20000",
                        help="tag of the OSV-5M-trained artifacts to score (default: 20000)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rebuild-manifest", action="store_true")
    args = parser.parse_args()

    from geoguessr.calibrate.temperature import TemperatureScaler
    from geoguessr.data.countries import class_names
    from geoguessr.data.crop import GEOGUESSR_16_9
    from geoguessr.data.manifest import (
        build_manifest_from_country_folders,
        manifest_stats,
        read_manifest,
        write_manifest,
    )
    from geoguessr.embed.backbone import Backbone
    from geoguessr.embed.extract import extract_embeddings, load_embeddings
    from geoguessr.eval.harness import evaluate
    from geoguessr.train.head import CountryHead, labels_from_manifest
    from geoguessr.train.mlp import MLPHead

    ARTIFACTS.mkdir(exist_ok=True)

    # --- manifest --------------------------------------------------------
    manifest_path = ARTIFACTS / "manifest_kaggle50k.parquet"
    if manifest_path.exists() and not args.rebuild_manifest:
        manifest = read_manifest(manifest_path)
        log(f"reusing {manifest_path.name}")
    else:
        if not KAGGLE_ROOT.exists():
            raise FileNotFoundError(
                f"{KAGGLE_ROOT} not found; run "
                "kagglehub.dataset_download('ubitquitin/geolocation-geoguessr-images-50k') first"
            )
        log(f"building manifest from {KAGGLE_ROOT}")
        manifest = build_manifest_from_country_folders(KAGGLE_ROOT, split="eval")
        write_manifest(manifest, manifest_path)

    stats = manifest_stats(manifest)
    log(f"{stats['rows']} rows across {stats['countries_present']} countries "
        f"(largest class {stats['largest_share']*100:.1f}%)")

    # --- embeddings --------------------------------------------------------
    # GEOGUESSR_16_9 crop, not FULL_FRAME: these are real screenshots with a
    # HUD, unlike OSV-5M's training images.
    embeddings_path = ARTIFACTS / "embeddings_kaggle50k.npy"
    try:
        embeddings, meta = load_embeddings(embeddings_path, manifest=manifest)
        log(f"reusing {embeddings_path.name}: {meta.n_rows} x {meta.embed_dim} "
            f"({len(meta.failed_ids or [])} failed)")
    except (FileNotFoundError, ValueError) as exc:
        if embeddings_path.exists():
            log(f"cache unusable ({exc}); re-embedding")
        log("loading frozen backbone")
        backbone = Backbone()
        log(f"  {backbone}")
        started = time.time()
        embeddings, meta = extract_embeddings(
            manifest, backbone, embeddings_path,
            crop=GEOGUESSR_16_9, batch_size=args.batch_size,
        )
        elapsed = time.time() - started
        log(f"  embedded {meta.n_done} rows in {elapsed/60:.1f} min "
            f"({elapsed/max(meta.n_done,1)*1000:.0f} ms/img); "
            f"{len(meta.failed_ids or [])} failed")

    classes = class_names()
    labels = labels_from_manifest(manifest, classes)
    matrix = np.asarray(embeddings, dtype=np.float32)
    eval_set = list(zip(matrix, labels))

    # --- evaluate whatever trained heads exist for --train-tag -------------
    results = []

    linear_path = ARTIFACTS / f"head_{args.train_tag}.joblib"
    if linear_path.exists():
        head = CountryHead.load(linear_path)
        predict_fn = head.predict_fn()
        results.append(evaluate(predict_fn, eval_set, classes,
                                name="linear probe", progress=False))

        scaler_path = ARTIFACTS / f"temperature_{args.train_tag}.json"
        if scaler_path.exists():
            scaler = TemperatureScaler.load(scaler_path)
            results.append(evaluate(scaler.wrap(predict_fn), eval_set, classes,
                                    name="linear probe + temperature", progress=False))
    else:
        log(f"no linear probe at {linear_path}, skipping")

    mlp_path = ARTIFACTS / f"mlp_{args.train_tag}.pt"
    if mlp_path.exists():
        mlp = MLPHead.load(mlp_path)
        predict_fn = mlp.predict_fn()
        results.append(evaluate(predict_fn, eval_set, classes,
                                name="mlp", progress=False))

        scaler_path = ARTIFACTS / f"temperature_mlp_{args.train_tag}.json"
        if scaler_path.exists():
            scaler = TemperatureScaler.load(scaler_path)
            results.append(evaluate(scaler.wrap(predict_fn), eval_set, classes,
                                    name="mlp + temperature", progress=False))
    else:
        log(f"no MLP head at {mlp_path}, skipping")

    if not results:
        raise RuntimeError(
            f"no trained heads found for tag {args.train_tag!r}; "
            "run scripts/run_pipeline.py first"
        )

    for result in results:
        print()
        print(result.summary())
    results[-1].save(ARTIFACTS / "eval_kaggle50k.json")

    print()
    print("=" * 68)
    print(f"{'model':<32} {'top-1':>7} {'top-5':>7} {'macro':>7} {'ECE':>7}")
    for result in results:
        print(f"{result.name:<32} {result.top1:>7.4f} {result.top5:>7.4f} "
              f"{result.macro_top1:>7.4f} {result.ece:>7.4f}")
    print("=" * 68)
    print("GeoGuessr-50k: real game screenshots, out-of-distribution from the")
    print("OSV-5M training data. This is the number that actually matters.")


if __name__ == "__main__":
    main()
