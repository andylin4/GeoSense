"""End-to-end Phase 1 -> 3 run: manifest -> embeddings -> probe -> harness.

This is the "prove the pipeline at small scale" loop. Start small
(``--limit 2000``) so a mistake costs minutes rather than an hour, then scale.

    uv run python scripts/run_pipeline.py --limit 2000
    uv run python scripts/run_pipeline.py --limit 20000

Every intermediate artifact is keyed by ``--limit`` and cached, so re-running
is cheap and only new work executes.

IMPORTANT: the numbers this prints are an OSV-5M-internal validation split.
They measure whether the pipeline works, NOT whether the model generalizes to
game screenshots. Only GeoGuessr-50k can answer that, and it is blocked.
"""

from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path

import numpy as np

ARTIFACTS = Path("artifacts")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def ensure_shard(shard: str, dest: Path) -> Path:
    """Download and extract one OSV-5M image shard, skipping work already done."""
    from huggingface_hub import hf_hub_download

    if dest.exists() and any(dest.iterdir()):
        log(f"shard already extracted at {dest}")
        return dest

    log(f"downloading {shard} (~2.5GB, cached after first run)")
    archive = hf_hub_download("osv5m/osv5m", shard, repo_type="dataset")

    dest.mkdir(parents=True, exist_ok=True)
    log(f"extracting to {dest}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=2000,
                        help="images to use (default: 2000)")
    parser.add_argument("--shard", default="images/train/00.zip")
    parser.add_argument("--strategy", default="stratified",
                        choices=["random", "stratified"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mlp", action=argparse.BooleanOptionalAction, default=True,
                        help="also train an MLP head and compare (default: yes)")
    parser.add_argument("--hidden", type=int, nargs="+", default=[512],
                        help="MLP hidden layer widths")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--rebuild-manifest", action="store_true",
                        help="regenerate the manifest from train.csv instead of "
                             "reusing the saved one (needs the source data)")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    from geoguessr.calibrate.temperature import TemperatureScaler
    from geoguessr.data.countries import class_names, display_name
    from geoguessr.data.crop import FULL_FRAME
    from geoguessr.data.manifest import (
        attach_paths,
        build_manifest,
        manifest_stats,
        read_manifest,
        subsample,
        write_manifest,
    )
    from geoguessr.embed.backbone import Backbone
    from geoguessr.embed.extract import extract_embeddings, load_embeddings
    from geoguessr.eval.baselines import prior_predict_fn
    from geoguessr.eval.harness import evaluate
    from geoguessr.train.head import (
        labels_from_manifest,
        split_indices,
        train_linear_probe,
    )
    from geoguessr.train.mlp import train_mlp_head

    tag = f"{args.limit}"
    ARTIFACTS.mkdir(exist_ok=True)

    # --- Phase 1: data layer -------------------------------------------------
    # A saved manifest is reused verbatim. Rebuilding it would require
    # train.csv (2.9GB) and the extracted shard (2.4GB) purely to reconstruct a
    # file already on disk -- which is what would otherwise make this project
    # un-portable without dragging 5GB of source data along.
    manifest_path = ARTIFACTS / f"manifest_{tag}.parquet"

    if manifest_path.exists() and not args.rebuild_manifest:
        manifest = read_manifest(manifest_path)
        stats = manifest_stats(manifest)
        log(f"reusing {manifest_path.name}: {stats['rows']} rows across "
            f"{stats['countries_present']} countries "
            f"(--rebuild-manifest to regenerate)")
    else:
        images_dir = ensure_shard(args.shard, Path("data_raw") / Path(args.shard).stem)

        log("building manifest from train.csv")
        csv = hf_hub_download("osv5m/osv5m", "train.csv", repo_type="dataset")
        manifest = build_manifest(csv, split="train")
        log(f"  {manifest.height} rows in Street View coverage")

        manifest = attach_paths(manifest, images_dir)
        log(f"  {manifest.height} rows with images present in this shard")

        manifest = subsample(manifest, limit=args.limit, strategy=args.strategy,
                             seed=args.seed)
        stats = manifest_stats(manifest)
        log(f"  sampled {stats['rows']} rows across {stats['countries_present']} "
            f"countries (largest class {stats['largest_share']*100:.1f}%)")

        write_manifest(manifest, manifest_path)

    # --- Phase 2: embeddings -------------------------------------------------
    # Try the cache before loading the backbone: a complete cache means no
    # encoder, no images, and no GPU are needed at all.
    embeddings_path = ARTIFACTS / f"embeddings_{tag}.npy"
    embeddings = meta = None

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
            crop=FULL_FRAME, batch_size=args.batch_size,
        )
        elapsed = time.time() - started
        log(f"  embedded {meta.n_done} rows in {elapsed/60:.1f} min "
            f"({elapsed/max(meta.n_done,1)*1000:.0f} ms/img); "
            f"{len(meta.failed_ids or [])} failed")

    # --- Phase 3: country head ----------------------------------------------
    classes = class_names()
    labels = labels_from_manifest(manifest, classes)
    matrix = np.asarray(embeddings, dtype=np.float32)

    train_idx, val_idx = split_indices(labels, val_fraction=args.val_fraction,
                                       seed=args.seed)
    log(f"split: {len(train_idx)} train / {len(val_idx)} val")

    # Three disjoint held-out roles, because sharing any two of them corrupts a
    # number we care about:
    #   earlystop -- carved out of TRAIN, selects MLP weights
    #   calib     -- fits the temperature
    #   test      -- reports every metric
    # Early-stopping on calib and then fitting T on calib measurably degrades
    # calibration: the weights are already tuned to those rows, so T is
    # estimated from optimistic predictions and mis-transfers to test.
    # All three come out of the validation set, never out of train: training
    # data is the scarce resource, and one scalar temperature needs very few
    # rows to fit.
    rng = np.random.default_rng(args.seed)
    earlystop_idx, calib_idx, test_idx = np.array_split(rng.permutation(val_idx), 3)
    fit_idx = train_idx

    log("fitting linear probe (class_weight=balanced)")
    started = time.time()
    heads = {"linear probe": train_linear_probe(matrix[train_idx],
                                                labels[train_idx], classes)}
    log(f"  fit in {time.time()-started:.1f}s")
    heads["linear probe"].save(ARTIFACTS / f"head_{tag}.joblib")

    if args.mlp:
        log(f"training MLP head (hidden={args.hidden}, "
            f"{len(fit_idx)} fit / {len(earlystop_idx)} early-stop)")
        started = time.time()
        heads["mlp"] = train_mlp_head(
            matrix[fit_idx], labels[fit_idx], classes,
            val_embeddings=matrix[earlystop_idx], val_labels=labels[earlystop_idx],
            hidden=args.hidden, epochs=args.epochs, seed=args.seed,
        )
        log(f"  trained in {time.time()-started:.1f}s "
            f"(best epoch {heads['mlp'].meta['best_epoch']}"
            f"/{heads['mlp'].meta['epochs_run']})")
        heads["mlp"].save(ARTIFACTS / f"mlp_{tag}.pt")

    # --- Phase 6: calibration + evaluation -----------------------------------
    eval_set = list(zip(matrix[test_idx], labels[test_idx]))

    counts: dict[str, float] = {}
    for code in manifest["country"][train_idx].to_list():
        counts[code] = counts.get(code, 0.0) + 1.0

    results = [evaluate(prior_predict_fn(classes, counts), eval_set, classes,
                        name="coverage prior", progress=False)]

    for name, head in heads.items():
        scaler = TemperatureScaler.fit(
            head.predict_proba(matrix[calib_idx]), labels[calib_idx],
            fit_on="osv5m-val",
        )
        log(f"{name}: {scaler.summary().splitlines()[0]}")
        suffix = "" if name == "linear probe" else "_mlp"
        scaler.save(ARTIFACTS / f"temperature{suffix}_{tag}.json")

        results.append(evaluate(head.predict_fn(), eval_set, classes,
                                name=name, progress=False))
        results.append(evaluate(scaler.wrap(head.predict_fn()), eval_set, classes,
                                name=f"{name} + temperature", progress=False))

    for result in results:
        print()
        print(result.summary())
    results[-1].save(ARTIFACTS / f"eval_{tag}.json")

    print()
    print("=" * 68)
    print(f"{'model':<32} {'top-1':>7} {'top-5':>7} {'macro':>7} {'ECE':>7}")
    for result in results:
        print(f"{result.name:<32} {result.top1:>7.4f} {result.top5:>7.4f} "
              f"{result.macro_top1:>7.4f} {result.ece:>7.4f}")
    print("=" * 68)
    print("OSV-5M-internal split: proves the pipeline runs, NOT that the")
    print("model generalizes to game screenshots. That needs GeoGuessr-50k.")


if __name__ == "__main__":
    main()
