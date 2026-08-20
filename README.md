


https://github.com/user-attachments/assets/0b0faf7c-2a5a-4d59-bf09-10ac2c809d75



# GeoGuessr country prediction

Frozen CLIP backbone + trained country/meta heads. See
[`geoguessr-model-design.md`](geoguessr-model-design.md) for the design and the
numbered decisions referenced throughout the code.

## Setup

```bash
uv sync --extra data --extra dev
uv run pytest -q
```

## Usage: the live overlay

```bash
uv sync --extra serve
uv run python scripts/overlay.py --tag 20000
```

macOS requires two permissions for the terminal you launch this from, and
**neither takes effect until you restart that terminal** after granting them:

- `System Settings -> Privacy & Security -> Screen Recording`
- `System Settings -> Privacy & Security -> Accessibility` (needed for the
  global hotkey)

### Flow

1. **Start the overlay.** A small always-on-top window appears in the top-left
   corner, showing `waiting for input` and the hotkey (default `` ` ``).
   Drag it anywhere; it stays on top of the game.
2. **Load a GeoGuessr round** in a browser window, same as normal play.
3. **Press the hotkey** (`` ` `` by default) once the scene has loaded. The
   window shows `calculating...` while it grabs the screen, runs it through
   the encoder (roughly 1.2s on MPS without CoreML), and ranks countries.
4. **Read the ranked list.** It updates in place with the top-k countries and
   their percentages, most likely first.

`--watch` replaces the hotkey with continuous, change-triggered monitoring: a
cheap screen signature is checked every `--poll-interval` seconds, and a real
prediction only runs once the scene changes and settles for
`--settle-time` seconds, with a `--min-gap` floor between predictions so a
static screen doesn't re-encode on every poll. Useful for hands-free play;
the hotkey still works alongside it.

Press `Escape` to close the overlay.

### Reading the output

```
Poland          34.2%
Czechia         21.0%
Slovakia        13.9%
Lithuania        9.1%
Germany          7.3%
`   (uncalibrated)
```

- **The percentages are the model's calibrated confidence**, not a similarity
  score — if calibration is working, "34%" should be right about a third of
  the time it's said, not just the highest number in an arbitrary softmax.
  They're produced by temperature scaling fit on held-out data (see
  `geoguessr-model-design.md` §3, "Calibration").
- **`(uncalibrated)` next to the hotkey means no temperature file was found**
  for the loaded `--tag`, so the percentages are raw softmax output and will
  read as overconfident or underconfident — treat them as a ranking only, not
  as honest probabilities, until a `temperature_<tag>.json` exists.
- **Only the classes the head was trained on can appear.** With `--tag 20000`
  that's the ~100-country list in `geoguessr/data/countries.py`, not every
  country GeoGuessr can serve a round in. A correct answer outside that list
  can never show up in the ranking, however low.
- **Treat this as a training aid, not an oracle.** Per-country accuracy is
  uneven — Central European neighbors (Czechia, Austria, Slovakia, Hungary)
  are the model's weakest region, the same region human players find hardest.
  Confusions tend to be geographically sensible (neighboring countries), so a
  wrong top guess is often still useful information.
- **In `--watch` mode**, the status line also shows a running count of
  predictions made (`N reads`), so you can tell it's alive versus idle.

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
