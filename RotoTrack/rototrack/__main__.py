"""Command line entry point: ``python -m rototrack``.

With ``--json-progress`` every event is printed as one JSON object per line,
which is what the Nuke plugin reads to drive its progress bar.
"""

import argparse
import sys

from aitools_engine.events import run_cli

from . import pipeline
from .model import SIZES


def _keys(value):
    try:
        return [int(part) for part in value.replace(",", " ").split()]
    except ValueError:
        raise argparse.ArgumentTypeError("key frames are frame numbers, e.g. 1001,1040")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rototrack", description=__doc__)
    parser.add_argument("--input", required=True, help="plate pattern, e.g. plate.####.exr")
    parser.add_argument("--output", required=True, help="cache pattern, e.g. cache/matte.####.exr")
    parser.add_argument("--prompt", required=True, help="pattern of the prompt's alpha on the key frames")
    parser.add_argument("--keys", type=_keys, required=True, help="key frames, e.g. 1001,1040")
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    parser.add_argument("--mode", choices=pipeline.MODES, default="shape",
                        help="shape: a rough shape SAM 2 fits to the object; matte: an exact matte to follow")
    parser.add_argument("--size", choices=sorted(SIZES), default="large")
    parser.add_argument("--transfer", choices=["auto", "linear", "display"], default="auto")
    parser.add_argument("--layer", default=pipeline.DEFAULT_LAYER,
                        help="channel the matte is written to, e.g. rototrack1.alpha "
                             "(letters, digits, '_' and '.' only)")
    parser.add_argument("--force", action="store_true", help="rewrite frames already cached")
    parser.add_argument("--json-progress", action="store_true")
    args = parser.parse_args(argv)

    def work(report):
        return pipeline.run(
            args.input, args.first, args.last, args.output, args.prompt, args.keys,
            mode=args.mode, size=args.size, transfer=args.transfer, layer=args.layer,
            force=args.force, progress=report,
        )

    return run_cli(args.json_progress, work,
                   "Close other GPU apps (ComfyUI, other Nuke sessions) or use the base model.")


if __name__ == "__main__":
    sys.exit(main())
