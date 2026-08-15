"""Phase 2: embedding extraction.

Runs every manifest image through the frozen encoder exactly once and writes a
memmap whose row *i* corresponds to manifest row *i*. This is the decoupling
layer: everything downstream reads a 200MB array instead of decoding JPEGs, and
that is what keeps head-training iteration measured in minutes.

Because this is the one expensive step, correctness here is mostly about not
silently reusing a stale cache. A sidecar ``.meta.json`` records the backbone
fingerprint, the crop, the normalization flag, and a hash of the manifest's
image ids. Any mismatch refuses to resume rather than appending rows that mean
something different from the ones already written.

Interrupted runs resume from the last checkpoint, so a laptop lid closing costs
minutes, not hours.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from PIL import Image

from ..data.crop import FULL_FRAME, CropSpec

if TYPE_CHECKING:
    from .backbone import Backbone

__all__ = ["EmbeddingMeta", "extract_embeddings", "load_embeddings", "meta_path_for"]

CHECKPOINT_EVERY = 20  # batches


def meta_path_for(embeddings_path: str | Path) -> Path:
    return Path(embeddings_path).with_suffix(".meta.json")


def manifest_signature(manifest: pl.DataFrame) -> str:
    """Hash of the manifest's image ids, in order.

    Row order matters: the embedding array is positional, so a reordered
    manifest invalidates the cache just as surely as a different backbone.
    """
    ids = np.asarray(manifest["image_id"].to_list(), dtype=np.int64)
    return hashlib.sha256(ids.tobytes()).hexdigest()[:16]


@dataclass
class EmbeddingMeta:
    """Everything needed to decide whether a cache is still valid."""

    backbone: dict
    crop: str
    normalized: bool
    n_rows: int
    embed_dim: int
    dtype: str
    manifest_sig: str
    n_done: int = 0
    failed_ids: list[int] | None = None

    def to_dict(self) -> dict:
        payload = {
            "backbone": self.backbone,
            "crop": self.crop,
            "normalized": self.normalized,
            "n_rows": self.n_rows,
            "embed_dim": self.embed_dim,
            "dtype": self.dtype,
            "manifest_sig": self.manifest_sig,
            "n_done": self.n_done,
            "failed_ids": self.failed_ids or [],
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> EmbeddingMeta:
        return cls(**payload)

    def incompatible_with(self, other: EmbeddingMeta) -> str | None:
        """Return a human-readable reason these caches differ, or None."""
        checks = [
            ("backbone", self.backbone.get("key"), other.backbone.get("key")),
            ("crop", self.crop, other.crop),
            ("normalized", self.normalized, other.normalized),
            ("row count", self.n_rows, other.n_rows),
            ("embedding dim", self.embed_dim, other.embed_dim),
            ("dtype", self.dtype, other.dtype),
            ("manifest contents", self.manifest_sig, other.manifest_sig),
        ]
        for label, mine, theirs in checks:
            if mine != theirs:
                return f"{label} changed ({theirs!r} -> {mine!r})"
        return None


def extract_embeddings(
    manifest: pl.DataFrame,
    backbone: Backbone,
    out_path: str | Path,
    *,
    crop: CropSpec = FULL_FRAME,
    batch_size: int = 32,
    normalize: bool = True,
    resume: bool = True,
    progress: bool = True,
) -> tuple[np.memmap, EmbeddingMeta]:
    """Embed every image in ``manifest`` into a memmap at ``out_path``.

    The manifest must already have paths attached. Row ordering is preserved
    exactly, so ``embeddings[i]`` describes ``manifest[i]``.

    Images that fail to load are recorded in ``meta.failed_ids`` and left as
    zero rows -- a corrupt JPEG should not kill an eight-hour job.
    """
    if "path" not in manifest.columns or manifest["path"].null_count():
        raise ValueError(
            "manifest has unresolved paths; call attach_paths() before embedding"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = meta_path_for(out_path)

    desired = EmbeddingMeta(
        backbone=backbone.fingerprint.as_dict(),
        crop=crop.fingerprint(),
        normalized=normalize,
        n_rows=manifest.height,
        embed_dim=backbone.embed_dim,
        dtype="float16",
        manifest_sig=manifest_signature(manifest),
    )

    start = 0
    failed: list[int] = []

    if resume and meta_path.exists() and out_path.exists():
        existing = EmbeddingMeta.from_dict(json.loads(meta_path.read_text()))
        reason = desired.incompatible_with(existing)
        if reason:
            raise ValueError(
                f"refusing to resume {out_path.name}: {reason}. "
                "Delete the cache to re-embed from scratch."
            )
        start = existing.n_done
        failed = list(existing.failed_ids or [])
        if start >= manifest.height:
            embeddings = np.memmap(
                out_path, dtype=np.float16, mode="r",
                shape=(manifest.height, backbone.embed_dim),
            )
            return embeddings, existing

    mode = "r+" if (out_path.exists() and start > 0) else "w+"
    embeddings = np.memmap(
        out_path, dtype=np.float16, mode=mode,
        shape=(manifest.height, backbone.embed_dim),
    )

    paths = manifest["path"].to_list()
    ids = manifest["image_id"].to_list()
    desired.n_done = start
    desired.failed_ids = failed

    batch_starts = range(start, manifest.height, batch_size)
    if progress:
        try:
            from tqdm.auto import tqdm

            batch_starts = tqdm(
                batch_starts, desc=f"embed {out_path.name}", unit="batch",
                initial=0, total=len(range(start, manifest.height, batch_size)),
            )
        except ImportError:
            pass

    for n, offset in enumerate(batch_starts, start=1):
        stop = min(offset + batch_size, manifest.height)

        images, slots = [], []
        for row in range(offset, stop):
            try:
                with Image.open(paths[row]) as handle:
                    image = crop.apply(handle.convert("RGB"))
                    image.load()
                images.append(image)
                slots.append(row)
            except Exception:  # corrupt/truncated file: record and move on
                failed.append(ids[row])
                embeddings[row] = 0

        if images:
            vectors = backbone.encode_images(images, normalize=normalize)
            embeddings[slots] = vectors.astype(np.float16)

        desired.n_done = stop
        desired.failed_ids = failed
        if n % CHECKPOINT_EVERY == 0:
            embeddings.flush()
            meta_path.write_text(json.dumps(desired.to_dict(), indent=2))

    embeddings.flush()
    meta_path.write_text(json.dumps(desired.to_dict(), indent=2))
    return embeddings, desired


def load_embeddings(
    path: str | Path, *, manifest: pl.DataFrame | None = None
) -> tuple[np.memmap, EmbeddingMeta]:
    """Open an existing embedding cache read-only, verifying it if possible.

    Passing the manifest is strongly recommended: it is the only way to catch
    an array that was built from different rows than the ones about to be
    trained on.
    """
    path = Path(path)
    meta = EmbeddingMeta.from_dict(json.loads(meta_path_for(path).read_text()))

    if meta.n_done < meta.n_rows:
        raise ValueError(
            f"{path.name} is incomplete: {meta.n_done}/{meta.n_rows} rows embedded. "
            "Re-run extract_embeddings() to finish it."
        )

    if manifest is not None:
        signature = manifest_signature(manifest)
        if signature != meta.manifest_sig:
            raise ValueError(
                f"{path.name} was built from a different manifest "
                f"({meta.manifest_sig} != {signature}); embeddings are positional, "
                "so this would silently mislabel every row."
            )

    embeddings = np.memmap(
        path, dtype=np.float16, mode="r", shape=(meta.n_rows, meta.embed_dim)
    )
    return embeddings, meta
