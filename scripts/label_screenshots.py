"""Capture and label real game screenshots, building an evaluation set.

This is the route around the blocked Kaggle download. Every labelled
screenshot is one row of production-distribution evaluation data -- the only
kind that can answer whether the model generalizes past Mapillary.

    # capture the screen, then type the true country
    uv run python scripts/label_screenshots.py

    # label images already on disk
    uv run python scripts/label_screenshots.py --from-dir shots/

Labels are appended to ``eval_set/labels.csv`` after every single entry, so
quitting at any point keeps everything done so far. Country input accepts codes
or names in any spelling the project's alias table knows ("czechia", "Czech
Republic", "CZ" all work).

Note the workflow the design intends: guess first, then label. Capturing the
answer before you have committed to a guess turns a training tool into a
cheat sheet.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path


def load_done(labels_path: Path) -> set[str]:
    if not labels_path.exists():
        return set()
    with labels_path.open() as handle:
        return {row["image"] for row in csv.DictReader(handle)}


def append_label(labels_path: Path, image: str, code: str, name: str) -> None:
    exists = labels_path.exists()
    with labels_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(["image", "country", "country_name", "labelled_at"])
        writer.writerow([image, code, name, time.strftime("%Y-%m-%dT%H:%M:%S")])


def prompt_country() -> tuple[str, str] | None:
    """Ask until a resolvable country is given. Returns None to quit."""
    from geoguessr.data.countries import OTHER, display_name, to_code

    while True:
        raw = input("  country (or 'skip' / 'quit'): ").strip()
        if not raw:
            continue
        if raw.lower() in {"quit", "q"}:
            return None
        if raw.lower() in {"skip", "s"}:
            return ("", "")

        code = to_code(raw)
        if code == OTHER:
            print(f"    '{raw}' is not in the 100-class list -- try again "
                  "(or 'skip')")
            continue
        return code, display_name(code)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default="eval_set", help="output directory")
    parser.add_argument("--from-dir", help="label existing images instead of capturing")
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--predict", action="store_true",
                        help="show the model's guess after you label (never before)")
    parser.add_argument("--tag", default="20000")
    args = parser.parse_args()

    out_dir = Path(args.out)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "labels.csv"

    done = load_done(labels_path)
    if done:
        print(f"{len(done)} already labelled in {labels_path}")

    predictor = None
    if args.predict:
        from geoguessr.data.crop import get_preset
        from geoguessr.serve.predictor import Predictor

        predictor = Predictor.from_artifacts(tag=args.tag,
                                             crop=get_preset("geoguessr_16_9"))
        print(predictor.describe())

    # --- decide the source of images ----------------------------------------
    if args.from_dir:
        pending = [
            p for p in sorted(Path(args.from_dir).iterdir())
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and p.name not in done
        ]
        if not pending:
            print("nothing left to label")
            return 0
        print(f"{len(pending)} images to label\n")
        source = iter(pending)
    else:
        print("Press Enter to capture the screen; 'quit' to stop.\n")
        source = None

    from PIL import Image

    labelled = 0
    while True:
        # --- get the next image ---------------------------------------------
        if source is not None:
            path = next(source, None)
            if path is None:
                break
            print(f"[{path.name}]")
        else:
            command = input("Enter to capture ('quit' to stop): ").strip().lower()
            if command in {"quit", "q"}:
                break

            from geoguessr.serve.capture import grab_screen

            try:
                frame = grab_screen(monitor=args.monitor)
            except Exception as exc:
                print(f"  capture failed: {exc}")
                continue

            path = images_dir / f"shot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            frame.save(path)
            print(f"  saved {path.name} ({frame.width}x{frame.height})")

        # --- label it --------------------------------------------------------
        answer = prompt_country()
        if answer is None:
            break
        code, name = answer
        if not code:
            print("  skipped")
            continue

        append_label(labels_path, path.name, code, name)
        labelled += 1
        print(f"  -> {name} ({code})")

        # Deliberately after labelling: seeing the guess first would make this
        # a cheat sheet rather than a training tool.
        if predictor is not None:
            guesses = predictor.predict(Image.open(path).convert("RGB"), top_k=3)
            hit = "correct" if guesses[0].code == code else "wrong"
            print(f"     model said {guesses[0].name} "
                  f"({guesses[0].probability*100:.0f}%) -- {hit}")
        print()

    print(f"\n{labelled} labelled this session, "
          f"{len(done) + labelled} total in {labels_path}")
    if len(done) + labelled >= 100:
        print("That is enough to fit a production temperature on real "
              "screenshots (calibration needs far less data than training).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
