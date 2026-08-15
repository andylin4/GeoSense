"""Live prediction: capture the screen (or read an image) and rank countries.

    # from a saved image
    uv run python scripts/guess.py --tag 2000 --image shot.png

    # from the screen (needs macOS Screen Recording permission)
    # waits --delay seconds (default 3) before capturing so you can switch
    # to the game window first
    uv run python scripts/guess.py --tag 2000

    # just save a screenshot, to validate the crop or start an eval set
    uv run python scripts/guess.py --save-only shots/round1.png

    # draw the crop rectangle on an image so you can check it by eye
    uv run python scripts/guess.py --preview-crop --image shot.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="2000",
                        help="artifact tag, i.e. the --limit used to train")
    parser.add_argument("--image", help="read this file instead of capturing")
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--backend", default="auto", choices=["auto", "pytorch", "coreml"],
                        help="vision encoder to use (default: auto -- CoreML if "
                             "artifacts/streetclip_vision.mlpackage exists, else PyTorch)")
    parser.add_argument("--coreml-package", default="artifacts/streetclip_vision.mlpackage")
    parser.add_argument("--crop", default="geoguessr_16_9",
                        help="crop preset; use full_frame for uncropped images")
    parser.add_argument("--save-only", metavar="PATH",
                        help="capture and save, then exit without predicting")
    parser.add_argument("--delay", type=float, default=3.0,
                        help="seconds to wait before capturing, so you can switch "
                             "to the game first (default: 3; ignored with --image)")
    parser.add_argument("--preview-crop", action="store_true",
                        help="write <image>.crop.png with the crop drawn on it")
    parser.add_argument("--list-monitors", action="store_true")
    args = parser.parse_args()

    from PIL import Image

    from geoguessr.data.crop import get_preset

    if args.list_monitors:
        from geoguessr.serve.capture import list_monitors

        for i, monitor in enumerate(list_monitors()):
            label = "all displays" if i == 0 else f"display {i}"
            print(f"  {i}: {label:<14} {monitor['width']}x{monitor['height']}")
        return 0

    if args.save_only:
        from geoguessr.serve.capture import save_screenshot

        if args.delay:
            print(f"capturing in {args.delay:.0f}s -- switch to the game now...")
            time.sleep(args.delay)
        print(f"saved {save_screenshot(args.save_only, monitor=args.monitor)}")
        return 0

    crop = get_preset(args.crop)

    # --- get an image -------------------------------------------------------
    if args.image:
        image = Image.open(args.image).convert("RGB")
        source = args.image
    else:
        from geoguessr.serve.capture import grab_screen

        if args.delay:
            print(f"capturing in {args.delay:.0f}s -- switch to the game now...")
            time.sleep(args.delay)
        started = time.perf_counter()
        image = grab_screen(monitor=args.monitor)
        source = f"screen ({(time.perf_counter() - started) * 1000:.0f}ms grab)"

    if args.preview_crop:
        out = Path(args.image or "screen").with_suffix(".crop.png")
        crop.preview(image).save(out)
        print(f"wrote {out}  ({image.width}x{image.height}, "
              f"crop keeps {crop.kept_area * 100:.0f}%)")
        return 0

    # --- predict ------------------------------------------------------------
    from geoguessr.serve.predictor import Predictor, format_guesses

    backbone = None
    coreml_path = Path(args.coreml_package)
    use_coreml = args.backend == "coreml" or (args.backend == "auto" and coreml_path.exists())

    if args.backend == "coreml" and not coreml_path.exists():
        print(f"error: no CoreML package at {coreml_path}. "
              "Run scripts/convert_coreml.py first.", file=sys.stderr)
        return 1

    if use_coreml:
        from geoguessr.serve.coreml import CoreMLBackbone

        backbone = CoreMLBackbone(coreml_path)
    elif args.backend == "auto":
        print(f"note: no CoreML package at {coreml_path}, falling back to PyTorch "
              "(slower -- run scripts/convert_coreml.py to speed this up)")

    predictor = Predictor.from_artifacts(tag=args.tag, backbone=backbone, crop=crop)
    print(predictor.describe())
    print(f"source       {source}  ({image.width}x{image.height})")
    print()

    started = time.perf_counter()
    guesses = predictor.predict(image, top_k=args.top_k)
    elapsed = (time.perf_counter() - started) * 1000

    print(format_guesses(guesses))
    print()
    print(f"{elapsed:.0f}ms")

    if predictor.scaler is None:
        print("NOTE: no temperature fitted -- percentages are uncalibrated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
