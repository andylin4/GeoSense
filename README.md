# GeoGuessr country prediction

Frozen CLIP backbone + trained country/meta heads. See
[`geoguessr-model-design.md`](geoguessr-model-design.md) for the design and the
numbered decisions referenced throughout the code.

## Setup

```bash
uv sync --extra data --extra dev
uv run pytest -q
```

## Layout

| Path | Role |
|---|---|
| `geoguessr/eval/metrics.py` | Pure-numpy metric primitives (top-k, ECE, per-class, confusion) |
| `geoguessr/eval/harness.py` | `evaluate(predict_fn, dataset, class_names)` — the project's stable interface |
| `geoguessr/eval/baselines.py` | Zero-shot CLIP, coverage-prior, and uniform predict_fns |
| `geoguessr/embed/backbone.py` | Frozen StreetCLIP loader + fingerprint (guards the embedding cache) |
| `geoguessr/data/countries.py` | Canonical ~109-class list, ISO codes, cross-dataset name aliases |
| `geoguessr/data/crop.py` | Fractional crop specs (inference-side only, per decision #6) |
| `artifacts/` | `manifest.parquet`, `embeddings.npy`, `*.pt`, `temperature.json` |

## Measured facts (differ from the design doc)

Verified on this machine, 2026-08-13:

- StreetCLIP embeddings are **768-dim, not 512** — it is ViT-L/14 based.
- Its input is **336px, not 224** (`geolocal/StreetCLIP` = ViT-L/14-336).
- Image encoding costs **~164 ms/img** on MPS (fp16, M-series Air). That is
  ~55 min for 20k images locally, so bulk embedding belongs on Colab as the
  design assumed — but the margin is larger than the doc's estimate implies.
- `logit_scale` is 100.0.

## Status

Phase 0 is built and verified end-to-end on labeled sample images (San
Francisco → United States 90.2%, Nagasaki → Japan 94.9%).

**TODO:** the real Phase 0 baseline number needs GeoGuessr-50k from Kaggle,
which requires credentials at `~/.kaggle/kaggle.json`.
