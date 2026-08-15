# GeoGuessr Country Prediction Model

Design document. Status: **implemented** (semantic track complete, meta track code-complete
and awaiting data). Design review 2026-08-13; implementation findings folded in 2026-08-14.

See Section 14 for what was measured and where reality diverged from this document.

---

## 1. Goal

A tool that reads a GeoGuessr round from a screenshot and returns a ranked list of countries, each with a calibrated confidence percentage.

The model should use both categories of clue that human players use:

- **Semantic clues.** Vegetation, architecture, road markings, signage, script and language, soil color, vehicle types, utility pole design.
- **Meta clues.** Google Street View camera generation, camera mount height, car blur signature, antenna visibility, snorkel, rally car. These are artifacts of how the imagery was captured rather than properties of the place, and they narrow the candidate set aggressively.

Target: beat zero-shot StreetCLIP on real game screenshots, with confidence percentages that are actually honest rather than decorative.

---

## 2. Core idea

We are not teaching a model to see. CLIP was trained on hundreds of millions of image and text pairs and already has an internal representation of eucalyptus trees, Cyrillic script, and European road sign geometry. What it lacks is the mapping from visual features to country.

That mapping is the only thing being built.

CLIP converts any image into a fixed length vector -- 768 numbers for StreetCLIP -- which acts as a fingerprint. Similar scenes produce similar fingerprints. The project collects a few hundred thousand fingerprints with known country labels, then fits a small classifier on top of them.

Consequences of this framing:

- The expensive model is downloaded, not trained.
- The trained component is small enough to fit in minutes on a CPU.
- The one heavy compute step, converting images to fingerprints, happens exactly once.

### Prior work worth knowing

- **StreetCLIP** (`geolocal/StreetCLIP` on Hugging Face). CLIP fine-tuned on 1.1M geotagged street level images. Works zero-shot for country classification. This is the starting backbone and the baseline to beat.
- **PIGEON / PIGEOTTO** (Stanford, 2023). Fine-tuned StreetCLIP with semantic geocells, reported ~92% country accuracy. Represents a realistic ceiling.
- **GeoCLIP.** Alternative approach using GPS coordinate embeddings rather than discrete classes. Not the chosen approach here but relevant if country-level granularity turns out to be too coarse.

---

## 3. Architecture

```
Screen capture (crop out HUD and minimap)
        |
        v
CLIP vision encoder (frozen)  ->  768-dim embedding
        |
        +----------------------+
        |                      |
        v                      v
  Country head            Meta head
  (softmax over           (camera generation,
   ~100 classes)           cam height)
        |                      |
        +----------------------+
                   |
                   v
        Fusion and calibration
        (reweight by meta, then temperature scale)
                   |
                   v
        Ranked countries with calibrated %
```

### Crop specification

Training images (OSV-5M) have no game UI to crop — they are full-frame street scenes, used as-is aside from the standard resize to CLIP's input size. The crop rule applies only on the inference side, and to GeoGuessr-50k, which is re-cropped identically before evaluation:

- The capture surface is constrained to a fullscreen browser window at a fixed, documented resolution (e.g. 1920x1080). Under that constraint the minimap, compass, and score bar sit at deterministic pixel offsets, so the crop rectangle is a fixed constant rather than something detected at runtime.
- No synthetic UI overlay is added to training data, and no dynamic HUD detection is built. Both would solve a problem — variable window sizes — that's out of scope by construction.

### Why two heads instead of one

Meta cues and semantic cues live at different levels of the image. Semantic cues are content. Meta cues are low level image statistics: resolution, chromatic aberration, blur signature, horizon position, lens distortion.

More importantly, they cannot share training data. The bulk semantic dataset comes from Mapillary, which contains no Google car, no Google blur, and no camera generations. A single head trained on that data can never learn meta, no matter how large it gets.

So the meta track is a fully separate data pipeline with its own images, its own labels, and its own small model. The two tracks meet only at fusion.

### Fusion

Start simple. Build a lookup table $P(\text{gen} \mid \text{country})$ directly from the scrape metadata collected for the meta head (Section 4) — the same (country, capture-date → derived-gen) pairs, aggregated per country. No separate coverage-table sourcing effort. Then reweight:

$$s'_c = s_c \cdot P(\hat{g} \mid c)$$

where $s_c$ is the country head score and $\hat{g}$ is the predicted camera generation. Learned fusion is a later refinement, not a starting point.

### Calibration

Raw softmax outputs are usually miscalibrated. Fit a single scalar temperature $T$ on a held-out validation set by minimizing negative log likelihood:

$$\hat{p}_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}, \qquad T^* = \arg\min_T \; -\sum_{n} \log \hat{p}^{(n)}_{y_n}$$

One scalar, fits in seconds. This is what makes "87% Poland" mean the model is right about 87% of the time rather than being a number it made up.

**Measured, the correction ran the other way.** This document assumed overconfidence ($T > 1$,
softening). The trained probe was severely *under*confident -- claiming ~5% while being ~57%
accurate -- and the fitted temperature was $T = 0.186$, i.e. sharpening. Cause:
`class_weight="balanced"` spread across 100 classes with few examples each. ECE fell from
0.5253 to 0.0682. Do not carry the overconfidence assumption into the meta head or the
production refit.

Calibration is not a phase. It is a step appended to the end of every training run, since any change to the head invalidates the previous $T$.

---

## 4. Datasets

| Dataset | Source | Size | Role | Notes |
|---|---|---|---|---|
| **OSV-5M** | Hugging Face `osv5m/osv5m` | 5.1M images, 225 countries | Primary training data for the country head | Sourced from Mapillary. Free. Has an enforced train/test split. Contains **no** Google meta cues. |
| **GeoGuessr-50k** *(BLOCKED)* | Kaggle `ubitquitin/geolocation-geoguessr-images-50k` | ~50k images, 124 countries | Held-out evaluation only | Actual game screenshots with UI overlay. This is the only data matching production distribution. Heavily imbalanced -- the US is ~12k of 50k (24%), and the smallest classes hold a single image. **Never train on this.** `kaggle.com` was blocked by the work machine's Cloudflare Zero Trust policy as of 2026-08-14; on this (personal) machine it's reachable, so it just needs credentials at `~/.kaggle/kaggle.json` to pull the real evaluation number. Fallback workarounds if needed: a HuggingFace mirror, or self-captured screenshots via `scripts/label_screenshots.py`. |
| **Google Street View (self-scraped)** | Street View Static API | ~2k to 5k images | Training data for the meta head | Costs roughly $7 per 1000 images. The metadata endpoint is **free**, so probe coordinates for coverage first and only pay for confirmed panoramas. |
| ~~Country boundaries~~ | ~~Natural Earth admin-0~~ | n/a | **Not needed** | OSV-5M already ships ISO 3166-1 alpha-2 codes, so the point-in-polygon join was deleted along with DuckDB's spatial extension. Upstream labels are accepted as-is -- the same posture agreed for disputed borders. |

### Why training and test data come from different sources

This is deliberate. Training on Mapillary and testing on game screenshots is the check that the model learned geography rather than learning to recognize Mapillary's characteristic look. If the two came from the same place, a high score would prove nothing.

### Disputed and ambiguous borders

Boundaries are contested or ambiguous for a handful of regions with Street View coverage (Crimea, Western Sahara, Kosovo, parts of the India/Pakistan/China border). Upstream labels are accepted as-is, with no special-casing. *(As built this means OSV-5M's own country codes rather than Natural Earth's, since the geo join proved unnecessary -- but the posture is identical: accept the upstream call.)* These regions are a small fraction of both total coverage and the ~100-country target list; revisit only if per-country error analysis flags one of them as a disproportionate confusion source relative to its coverage share.

### Getting meta labels without hand labeling everything

The Street View metadata API returns the panorama capture date for free. Camera generation correlates strongly with capture year and country, so most gen labels can be derived programmatically.

Cam height has no metadata signal, but Google's documented deployment history — which countries/regions were shot with trekker backpacks, snowmobiles, or older low-mount cars versus current tall roof-rack rigs — gives a rough first-pass label the same way capture date gives one for camera generation. Hand-label only the residual where that derivation is ambiguous (transition periods, mixed-coverage countries). That's what the ~500-image manual budget is actually for, not a from-scratch labeling pass.

---

## 5. Tech stack

| Layer | Choice | Rationale |
|---|---|---|
| Environment | `uv`, Python 3.11 | Fast, handles torch on Apple Silicon cleanly |
| Framework | PyTorch 2.x with MPS | Native Apple Silicon acceleration |
| Encoder | `open_clip` or HF `transformers` | StreetCLIP is on the HF hub |
| Data download | `huggingface_hub.snapshot_download` | OSV-5M is one call |
| Tabular | Polars | Reads the 2.9GB `train.csv` and filters 5.1M rows in seconds. DuckDB and its spatial extension were dropped once OSV-5M's own country codes made the geo join unnecessary. |
| Embedding storage | numpy `memmap` | 768-dim, so 200k rows at fp16 is ~307MB (not 200MB); all 5.1M would be ~7.8GB. No vector DB needed -- classification, not retrieval. A sidecar `.meta.json` fingerprints backbone, crop, and manifest so a stale cache is refused rather than silently reused. |
| Head training | scikit-learn first, then PyTorch | `LogisticRegression` is a three line baseline that validates the pipeline |
| Experiment tracking | Weights and Biases | Free tier, and there will be many runs |
| Screen capture | `mss` plus Pillow | Sub-10ms grabs on macOS, much faster than `ImageGrab` |
| Global hotkey | `pynput` | Requires Accessibility permission in System Settings |
| Inference runtime | `coremltools` | Apple Neural Engine, roughly 300ms to 40ms improvement over MPS |
| UI | Gradio for dev, small Tkinter overlay for live use | Gradio gives a demo in 20 lines |

### Hardware plan

Target machine is a fanless MacBook Air (Apple Silicon).

- **Head training runs locally.** Fitting a classifier on cached embeddings is minutes of CPU. The laptop stays cool.
- **Embedding extraction runs on Colab's free T4.** It is embarrassingly parallel, needs to happen once, and the output file is small enough to download in a couple of minutes.
- **Optional backbone fine-tuning runs on Modal or Colab.** Only attempt after the frozen-backbone version is working and measured.

Sustained MPS load on a fanless Air will thermally throttle. That is the machine protecting itself, not damaging itself, but it makes local embedding extraction slow enough to be annoying rather than dangerous.

---

## 6. Example workflow, round start to guess

**T = 0s.** Round loads. Player looks at the scene.

**T = 3s.** Player presses the global hotkey.

**T + 10ms.** `mss` grabs the primary display. The image is cropped to the game viewport, cutting the minimap, compass, and score bar. This crop must exactly match the crop used during training, since a different crop produces a different fingerprint.

**T + 50ms.** The cropped image is preprocessed to 336x336 and passed through the CoreML CLIP encoder. Output: a 768-dim embedding.

> **Measured:** the PyTorch/MPS path takes **~1259ms**, not 50ms -- a 25x miss. Encoding is the entire
> cost; the heads are a matrix multiply. CoreML conversion exists (`scripts/convert_coreml.py`) but has
> not been run, so the 50ms target is still unverified.

**T + 55ms.** The embedding is passed to both heads in parallel.
- Country head returns raw logits over ~100 country classes.
- Meta head returns a predicted camera generation and cam height, each with its own confidence.

**T + 56ms.** Fusion. Country scores are reweighted by $P(\hat{g} \mid c)$. If the meta head is confident about Gen 2 with a specific blur, a large fraction of the world drops out before vegetation is even considered.

**T + 57ms.** Temperature scaling converts fused scores into calibrated probabilities.

**T + 60ms.** Overlay window updates with the ranked list:

```
Poland          34%
Czechia         21%
Slovakia        14%
Lithuania        9%
Germany          7%
...
meta: Gen 3, high cam (conf 0.81)
```

**T = 4s.** Player makes the actual guess.

Note that no training data is touched at inference time and nothing is looked up in a database. The live path loads two small saved files and does a little matrix math. It makes **no network calls of any kind** -- running the tool is free forever, whatever the meta scrape cost.

### Watch mode (added during implementation)

The hotkey flow above is one of two. `--watch` monitors the screen continuously without a keypress, but
is **change-triggered rather than interval-triggered**: a 32x32 grayscale signature is compared every
250ms (~10ms cost) and the encoder only runs when the scene changes *and* settles. Three gates -- change
threshold, settle time, and a minimum gap -- keep a static screen at roughly one encode total rather than
one per tick. That last gate is the thermal guard this document asks for in Section 5.

The change signature is computed on the *cropped* region, so GeoGuessr's ticking round timer cannot
trigger an endless encode loop.

---

## 7. Build phases

```
                 Phase 0: baseline
              zero-shot + eval harness
                    /          \
                   /            \
      Phase 1: data layer    Phase 4: meta data
      manifest of path,      Street View scrape
      lat, lon, country      + date-derived labels
            |                       |
      Phase 2: embeddings    Phase 5: meta head
      frozen encoder ->      gen and cam height
      memmap                       |
            |                      |
      Phase 3: country head        |
      probe, then MLP              |
                   \              /
                    \            /
                 Phase 6: fusion + calibration
                 calibrate on real screenshots
                            |
                     Phase 7: serving
                  capture, CoreML, overlay
```

### Phase 0. Baseline and eval harness

Produces two things: a baseline number (zero-shot StreetCLIP on GeoGuessr-50k, expect somewhere in the 55 to 70% top-1 range) and a function:

```python
evaluate(predict_fn, dataset) -> {top1, top5, ece, per_country, confusion}
```

That signature is the most important interface in the project. Every later phase is just a new `predict_fn` passed into the same harness. Write it before anything else exists.

### Phase 1. Data layer

Download an OSV-5M subset and read its country codes directly -- no geo join needed. Filter to the 100 covered countries, dropping the rest. Output `manifest.parquet` with columns `image_id, lat, lon, country, split, path`.

The manifest stores the country **code**, never an integer class index, which is what keeps relabeling
free: change the class scheme and only this column changes, never the embedding file. A saved manifest
is reused verbatim on later runs, so continuing work needs only `artifacts/` -- not the 5GB of source
data.

This is the contract with Phase 2. Phase 2 does not know where the images came from, so swapping data sources later means rewriting only this phase.

### Phase 2. Embedding extraction

Run every image through the frozen encoder once. Write to `embeddings.npy` (memmap) with a row index aligned to the manifest. Include a resume checkpoint so the job can be stopped and restarted.

This is the decoupling layer and the reason iteration stays fast.

### Phase 3. Country head

Consumes only the embedding array, never images. Fit `LogisticRegression` first as a pipeline sanity check. If the linear probe hits reasonable accuracy, the embeddings are good and an MLP will add a few points. If it is near chance, something upstream is broken and no architecture will fix it.

Handle class imbalance here. Street view coverage is dominated by the US, Russia, and Brazil, so an uncorrected model can score respectably while being useless. Either resample or reweight the loss, and always read per-country accuracy rather than the aggregate.

### Phase 4. Meta data collection

Independent of phases 1 to 3. Scrape Street View, derive gen labels from capture dates, hand label a validation subset.

### Phase 5. Meta head

Small classifier on the same embedding space, or on raw image statistics if CLIP embeddings turn out to discard the relevant low level information. **This is an open question, see section 10.**

### Phase 6. Fusion and calibration

The two tracks meet. Calibrate against real game screenshots, not training data.

### Phase 7. Serving

No learning. Consumes a frozen backbone plus two heads. CoreML conversion is the only fiddly part.

### Suggested sequencing

Build Phase 0 and Phase 7 back to back at the very start, using zero-shot StreetCLIP as the model. This yields a working screen-reading tool in a weekend. Every phase after that becomes a measurable upgrade to something already in use, rather than a step toward something not yet seen working.

Then run phases 1 through 3 as a single loop at small scale, roughly 20k images, purely to prove the pipeline end to end. Scale up only after it works.

---

## 8. What runs once versus what runs constantly

| Step | Cadence | Invalidated by |
|---|---|---|
| Data prep | Once | New data source, changed label scheme |
| Embedding extraction | Once | Changed backbone, changed crop region |
| Head training | Constantly | Any experiment |
| Calibration | Every training run | Any change to the head |
| Live inference | Every round | Nothing, it just loads saved files |

### Invalidation rules

- **Changing the backbone is the expensive mistake.** Every cached embedding becomes meaningless. Same for changing the crop. Decide both early, compare candidates on a 20k subset, then stop revisiting.
- **Adding images is cheap.** Embed only the new ones and append. Existing embeddings stay valid.
- **Relabeling is free.** Embeddings depend on pixels, not labels. Switching from country-level to region-level, or merging classes, means rewriting the manifest label column and retraining the head. The embedding file is never touched. The most expensive artifact is also the most reusable.

---

## 9. Repo layout

```
geoguessr/
  data/          duckdb prep, country label join, subset sampling
  embed/         batch encode to memmap, resume logic
  train/         linear probe, mlp head, meta head
  calibrate/     temperature fitting, ECE
  eval/          the harness, confusion matrices, per-country breakdown
  serve/         capture, crop, coreml inference, overlay
  notebooks/     exploration
  artifacts/     manifest.parquet, embeddings.npy, *.pt, temperature.json
```

---

## 10. Design decisions

Resolved during design review on 2026-08-13. Each entry keeps the original question for traceability.

1. **Backbone choice.** StreetCLIP first pass, embedded on the 20k pipeline-proving subset — domain-matched to street-level imagery (Section 2's working premise: frozen backbone, small trained head, no fine-tuning). ViT-L/14 is a fallback comparison, pulled in only if StreetCLIP's linear probe looks weak or turns out to discard the meta signal (see #2). At 20k images the embedding-time gap between backbones is minutes — not worth trading accuracy for speed by defaulting to B/32.

2. **Does the meta head belong on CLIP embeddings?** Test cheaply before building Phase 5: once the ~2k-5k scraped images exist, extract their CLIP embeddings (reusing Phase 2's pipeline) and fit a small camera-gen classifier directly on them. If it clears a reasonable bar above the gen-frequency baseline, CLIP embeddings suffice and Phase 5 is just another head. If it's near chance, build a separate small CNN on raw pixel statistics instead.

3. **Country classes versus geocells.** Direct country classification, not geocells — a consequence of committing to a fixed ~100-country list in #4.

4. **How many classes.** ~100 classes: countries with real Street View coverage, a superset of GeoGuessr-50k's 124 trimmed to ones with enough training images to learn anything. Inputs joined to an excluded country route to a single `other` bucket rather than being silently dropped.

5. **Handling class imbalance.** Reweight the loss (inverse-frequency or effective-number weighting), not resampling. Keeps every training image in play — unlike undersampling, which discards usable majority-class data, or oversampling, which risks overfitting repeated minority-class images — and leaves one consistent transformation for temperature scaling to correct afterward, rather than a data-dependent one that varies per training run.

6. **Crop specification.** Inference-side only — see Section 3, "Crop specification." OSV-5M trains on full-frame images with no cropping needed; GeoGuessr-50k is re-cropped identically to the inference rule before evaluation. The capture surface is constrained to fullscreen browser at a fixed, documented resolution, so the crop rectangle is a fixed constant rather than something detected at runtime.

7. **Domain gap severity.** Measured empirically as a byproduct of #9's step-scaling — GeoGuessr-50k accuracy at each dataset-size step is exactly the domain-gap-cost signal, no separate experiment needed.

8. **Fusion weighting.** Multiplicative $P(\hat g \mid c)$ reweighting for v1, built from the project's own scrape metadata (same data as the meta head, aggregated per country — see Section 3, "Fusion"). Learned fusion deferred until the simple version is shown to leave accuracy on the table.

9. **Target subset size.** Step scale 20k → 200k → full, evaluating GeoGuessr-50k accuracy at each step via the Phase 0 harness. Stop scaling once the gain between steps drops to 1-2 points.

10. **Success bar and stopping rule.** Ship when top-1 accuracy on GeoGuessr-50k clears zero-shot StreetCLIP by at least 10 points AND expected calibration error is below a set threshold (~0.05). Meta fusion refinements and the interpretability angle (Section 13) are optional post-v1 upgrades, not blockers — this bounds the head/fusion/calibration loop, the one part of the project with no natural end since every phase after Phase 2 is cheap to re-run.

---

## 11. Risks

**Domain mismatch.** Training on Mapillary, testing on game screenshots. A model can quietly learn to recognize the source rather than the country. Mitigation is testing on GeoGuessr-50k from day one.

**Class imbalance.** The US, Russia, and Brazil dominate coverage. Mitigation is per-country metrics rather than a single aggregate number.

**Overconfidence.** Untreated softmax will report 99% on images it is guessing at. Mitigation is temperature scaling and tracking expected calibration error as a first-class metric alongside accuracy.

**Meta data cost.** Street View scraping costs real money. Mitigation is the free metadata endpoint for coverage probing, so nothing is paid for unless a panorama is confirmed to exist.

**Scope creep on the backbone.** Re-embedding is the only genuinely expensive operation. Mitigation is deciding the backbone once, early, on a small subset.

---

## 12. Evaluation

Primary metrics:

- Top-1 and top-5 country accuracy on GeoGuessr-50k
- Expected calibration error, since the confidence percentage is a stated product requirement
- Per-country accuracy, to catch the model collapsing onto high-coverage countries
- Median geodesic error, if geocells are used

Baselines to beat, in order: random weighted by coverage, then zero-shot StreetCLIP, then the linear probe, then the MLP head, then MLP plus meta fusion.

---

## 13. Constraints and notes

Using this during ranked or competitive play violates GeoGuessr's terms of service. The intended use is as a training tool where the player guesses first and compares afterward, or as a pure ML project. The interpretability angle, showing what the model keyed on, is more interesting than raw accuracy anyway and is worth building toward.

---

## 14. Implementation status and measured findings

Written 2026-08-14, after building the project. Everything above is the design; this is what
actually happened.

### Status by phase

| Phase | State | Notes |
|---|---|---|
| 0 Baseline + harness | Built | `evaluate()` verified on real photos. **Baseline number blocked** on eval data. |
| 1 Data layer | Built | Simpler than designed -- no geo join. |
| 2 Embeddings | Built | Resume + cache fingerprinting. |
| 3 Country head | Built | Linear probe **and** MLP. |
| 4 Meta data | Code complete | Free half runnable; paid half needs a key and the framing decision. |
| 5 Meta head | Code complete | Includes the decision-#2 diagnostic. Needs data. |
| 6 Fusion + calibration | Built | Both halves. Fusion untested against real meta data. |
| 7 Serving | Built | CoreML written but **never executed**. |

341 tests, none touching the network.

### Measured results

20,000 OSV-5M images, stratified across 99 countries, 1,335 evaluation samples
(validation split three ways: early-stop / calibration / test):

| Model | top-1 | top-5 | macro | ECE |
|---|---|---|---|---|
| coverage prior | 0.0277 | 0.1169 | 0.0110 | 0.0035 |
| linear probe | 0.6544 | 0.9115 | 0.5970 | 0.3282 |
| linear probe + temperature | 0.6544 | 0.9115 | 0.5970 | **0.0213** |
| MLP (768->512->100) | 0.6732 | 0.9243 | 0.5985 | 0.0301 |
| MLP + temperature | **0.6732** | **0.9243** | 0.5985 | **0.0280** |

Scaling behaved as decision #9 hoped -- 2k to 20k bought **+8.2 points** of top-1
(0.5722 -> 0.6544 for the linear probe), so the curve had not flattened and there is more
to gain from 200k.

The MLP earns roughly the "few points" this document predicted: +1.9 top-1, +1.3 top-5.
Macro is flat (0.5970 vs 0.5985), so the extra capacity is sharpening common classes
rather than rescuing rare ones.

**The two heads miscalibrate in opposite directions.** The linear probe is underconfident
($T = 0.47$, sharpening); the MLP is mildly *over*confident ($T = 1.045$, softening) --
the direction originally assumed here. Calibration direction is a property of the head, not
of the project, which is exactly why $T$ is refitted after every training run.

Embedding 20k images took 39.2 minutes at 118 ms/img on MPS, with 0 failures. Thermal
throttling was visible: batches drifted from 3.75s to 4.25s over the run.

**These are OSV-5M-internal.** Train and validation share a distribution, so they measure that the
pipeline works -- not that the model generalizes to screenshots. That remains unmeasured.

Three things worth carrying forward:

1. **The frozen-backbone premise holds.** 67% top-1 and 92% top-5 across 99 countries, from a
   classifier that fits in under a second on cached vectors. Section 2's core bet was correct.
2. **No class collapse.** Macro (0.5824) tracks top-1 (0.5722), so the model is not riding imbalance.
3. **The confusions are geographically sensible** -- Poland→Lithuania, Austria→Switzerland,
   US→Canada, Sweden→Finland, Latvia→Estonia. Every one is a neighbour, and every one is a mistake
   skilled human players make. Good evidence the model learned geography rather than dataset
   artifacts.
4. **Central Europe is the weak spot.** The worst classes with real support are Czechia (0.06),
   Austria (0.27), Lithuania (0.27), Slovakia (0.28), Hungary (0.31) -- precisely the region human
   players find hardest. This is where meta cues should help most, and it is a concrete argument
   for finishing the meta track rather than only scaling data.

Counterweight: on an out-of-distribution photo the 2k probe ranked the correct country **2nd at
2.1%**, where zero-shot StreetCLIP said **90.2%**. One image is not a measurement, but it is
consistent with the probe being badly under-trained, and it is a reminder that zero-shot StreetCLIP
is a strong baseline that scale -- not architecture -- has to beat.

### Where reality diverged from this document

| Assumption | Reality |
|---|---|
| 512-dim embeddings | **768** (StreetCLIP is ViT-L/14 based) |
| 224px input | **336px** -- 2.25x the pixels, so embedding is far more expensive than assumed |
| Natural Earth join required | Unnecessary; OSV-5M ships ISO codes |
| Softmax is overconfident | Measured **under**confident; $T = 0.186$ |
| ~50ms inference | **~1259ms** on MPS without CoreML |
| ViT-B/32 is "4x faster" | Understated -- B/32 at 224px vs L/14 at 336px is a far larger gap |
| US is ~1/5 of GeoGuessr-50k | ~24%, with single-image classes in the tail |

### Class scheme, as built

Exactly **100 classes**. OSV-5M covers 219 countries, but 119 of them (22.6% of rows -- China, Iran,
Ethiopia, Belarus...) have no Google Street View coverage and were dropped rather than bucketed:
those locations essentially cannot be the answer in a round, so a softmax class for them spends
capacity on an impossible answer. Eight micro-territories with zero rows (Andorra, Monaco, San
Marino, Gibraltar, Macau, Guam, American Samoa, Northern Marianas) were dropped as dead classes.
`OTHER` survives as a *filter sentinel*, never a trainable class.

### Additions not in the original design

- **Watch mode** -- change-triggered continuous monitoring (Section 6).
- **Cache fingerprinting** -- embeddings carry backbone, crop, and manifest hashes; a mismatch
  refuses to resume. This caught two real reproducibility bugs where the same seed selected
  different rows across processes.
- **Fusion safety** -- the bare $s_c \cdot P(\hat g \mid c)$ formula gained a confidence mixer
  (uncertain meta degrades toward a no-op) and a probability floor (no unrecoverable hard zeros
  from an under-sampled coverage table).
- **Spend ceiling** -- the image fetcher cannot be constructed without an explicit `max_requests`.
- **Decision-#2 diagnostic** -- `diagnose_meta_signal()` compares against the majority-class
  baseline, not chance, so a head that always answers "gen4" cannot masquerade as signal.

### Open, in priority order

1. **Real evaluation.** GeoGuessr-50k is now downloaded (6.8GB, 124 country folders) and the crop
   is verified against it (item 2). Still needed: a manifest builder for its
   `compressed_dataset/<Country>/canvas_*.jpg` layout (unlike OSV-5M, filenames aren't numeric
   ids, so `attach_paths` doesn't apply as-is) and a run through the Phase 0 harness.
2. **Crop validation.** `GEOGUESSR_16_9` was verified 2026-08-15 against two real GeoGuessr-50k
   screenshots (France, Japan; both a fixed 1536x662 render) -- corrected to `top=0.12,
   bottom=0.55`, confirmed clean of all HUD elements. Still open: this dataset's images are a
   scripted, fixed-size capture, not a real 1920x1080 fullscreen browser at 16:9 as originally
   assumed -- the same fractional crop is **not** yet validated against a live `mss` screen
   capture, which may run at a different resolution/aspect ratio. That check still gates serving.
3. **Framing decision** (pitch/heading/fov) before any paid scraping -- re-scraping costs again.
4. **Run the CoreML conversion.** coremltools 9.0 warns it is untested against torch 2.13.
5. Scale past 20k, per the step-and-measure rule in decision #9.
