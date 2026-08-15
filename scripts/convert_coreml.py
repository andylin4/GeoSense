"""Convert the StreetCLIP vision encoder to CoreML and verify it.

    uv run python scripts/convert_coreml.py

Conversion takes several minutes and writes ~1.5GB. It is a build step, run
once per backbone -- never on the live path.

The verification at the end is the part that matters: it measures how far the
converted encoder's embeddings drift from the PyTorch ones the head was
actually trained on, and times both.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="artifacts/streetclip_vision.mlpackage")
    parser.add_argument("--compute-units", default="ALL",
                        choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"])
    parser.add_argument("--skip-convert", action="store_true",
                        help="only verify an existing package")
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download
    from PIL import Image

    from geoguessr.embed.backbone import Backbone
    from geoguessr.serve.coreml import (
        CoreMLBackbone,
        compare_backends,
        convert_vision_encoder,
    )

    out = Path(args.out)

    print("loading PyTorch backbone (CPU, fp32 -- required for tracing)")
    torch_backbone = Backbone(device="cpu", dtype=None)
    print(f"  {torch_backbone}")

    if not args.skip_convert:
        print(f"\nconverting to {out} (several minutes)")
        started = time.time()
        convert_vision_encoder(torch_backbone, out, compute_units=args.compute_units)
        print(f"  converted in {(time.time() - started) / 60:.1f} min")

    if not out.exists():
        print(f"error: {out} does not exist", file=sys.stderr)
        return 1

    print("\nloading CoreML package")
    coreml_backbone = CoreMLBackbone(out, processor=torch_backbone.processor)
    print(f"  {coreml_backbone}")

    # --- fidelity ----------------------------------------------------------
    samples = [
        Image.open(hf_hub_download("geolocal/StreetCLIP", name)).convert("RGB")
        for name in ("sanfrancisco.jpeg", "nagasaki.jpg")
    ]

    print("\nfidelity vs PyTorch (cosine should exceed ~0.99)")
    stats = compare_backends(torch_backbone, coreml_backbone, samples)
    for key, value in stats.items():
        print(f"  {key:<14} {value:.6f}")

    if stats["min_cosine"] < 0.99:
        print("\nWARNING: embeddings drifted. The head was trained on PyTorch")
        print("vectors, so predictions may differ. Investigate before serving.")

    # --- speed -------------------------------------------------------------
    print(f"\nlatency over {args.trials} trials (single image)")
    for label, backbone in (("pytorch", torch_backbone), ("coreml", coreml_backbone)):
        backbone.encode_images(samples[:1])  # warm up
        started = time.perf_counter()
        for _ in range(args.trials):
            backbone.encode_images(samples[:1])
        each = (time.perf_counter() - started) / args.trials * 1000
        print(f"  {label:<8} {each:7.0f} ms/img")

    return 0


if __name__ == "__main__":
    sys.exit(main())
