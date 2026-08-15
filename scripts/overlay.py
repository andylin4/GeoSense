"""Run the always-on-top overlay.

    uv run python scripts/overlay.py --tag 20000

Press the hotkey (default `) to read the screen and rank countries.
Escape closes it; drag the window to move it.

macOS needs two permissions for the host terminal, and neither takes effect
until the terminal is restarted:

    System Settings -> Privacy & Security -> Screen Recording
    System Settings -> Privacy & Security -> Accessibility
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="20000", help="artifact tag to load")
    parser.add_argument("--hotkey", default="`")
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--crop", default="geoguessr_16_9")
    parser.add_argument("--watch", action="store_true",
                        help="continuously watch the screen; predict when the "
                             "scene changes and settles (no hotkey needed)")
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--settle-time", type=float, default=0.4)
    parser.add_argument("--min-gap", type=float, default=1.5,
                        help="minimum seconds between encodes (thermal guard)")
    parser.add_argument("--coreml", metavar="PACKAGE", nargs="?",
                        const="artifacts/streetclip_vision.mlpackage",
                        help="use a converted CoreML encoder for faster inference")
    args = parser.parse_args()

    from geoguessr.data.crop import get_preset
    from geoguessr.serve.overlay import Overlay
    from geoguessr.serve.predictor import Predictor
    from geoguessr.serve.watch import WatchConfig

    backbone = None
    if args.coreml:
        from geoguessr.serve.coreml import CoreMLBackbone

        backbone = CoreMLBackbone(args.coreml)
        print(f"using {backbone}")

    predictor = Predictor.from_artifacts(
        tag=args.tag, crop=get_preset(args.crop), backbone=backbone
    )
    print(predictor.describe())
    if args.watch:
        print(f"\nwatching screen: poll {args.poll_interval}s, "
              f"settle {args.settle_time}s, min gap {args.min_gap}s")
    print(f"hotkey: {args.hotkey}   (Escape to quit)")

    Overlay(
        predictor,
        hotkey=args.hotkey,
        monitor=args.monitor,
        top_k=args.top_k,
        watch=args.watch,
        watch_config=WatchConfig(
            poll_interval=args.poll_interval,
            settle_time=args.settle_time,
            min_gap=args.min_gap,
        ),
    ).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
