"""CLI entry point used by the Nuke EngineJob."""

import argparse
import json
import math
import sys

from aitools_engine.events import run_cli
from aitools_engine.i18n import tr

from . import pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(prog="alphagen", description=__doc__)
    parser.add_argument("--check", action="store_true", help="check ComfyUI/model/workflow without queuing GPU work")
    parser.add_argument("--backend", choices=("start", "stop", "status"), help="control the local ComfyUI backend")
    parser.add_argument("--input", help="source EXR pattern, e.g. plate.####.exr")
    parser.add_argument("--output", help="alpha EXR pattern, e.g. cache/alpha.####.exr")
    parser.add_argument("--first", type=int)
    parser.add_argument("--last", type=int)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--resolution", type=int, choices=pipeline.RESOLUTIONS, default=544)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--layer", default="alphagen.alpha")
    parser.add_argument("--force", action="store_true", help="recompute all chunks")
    parser.add_argument(
        "--low-vram", action="store_true",
        help="force small VAE decode tiles on constrained GPUs; automatic on physical devices up to 8 GB",
    )
    parser.add_argument(
        "--subject", dest="subject_args", action="append", nargs=3,
        metavar=("FRAME", "X", "Y"),
        help=tr("Seed SAM 2.1 with a subject point in plate pixels; repeat for more subjects."),
    )
    parser.add_argument(
        "--subjects-json",
        help=tr("SAM prompts as JSON [[frame, x, y, subject, include], ...]; points of one subject are clicks on it."),
    )
    parser.add_argument(
        "--preview-subjects", action="store_true",
        help=tr("only track the SAM selection into --output (subject.alpha), without Alpha Gen"),
    )
    parser.add_argument(
        "--analyze-subjects", type=int, metavar="FRAME",
        help=tr("track every person found on FRAME into --output (subjects.id map), without Alpha Gen"),
    )
    parser.add_argument("--id-map", help=tr("person map EXR pattern written by --analyze-subjects"))
    parser.add_argument("--ids", help=tr("comma-separated people of --id-map to keep, e.g. 1,3"))
    parser.add_argument("--json-progress", action="store_true")
    args = parser.parse_args(argv)

    subjects = []
    for values in args.subject_args or []:
        try:
            frame = int(values[0])
            x, y = float(values[1]), float(values[2])
        except ValueError:
            parser.error(tr("--subject expects an integer FRAME and numeric X Y coordinates"))
        if not math.isfinite(x) or not math.isfinite(y):
            parser.error(tr("--subject X and Y must be finite numbers"))
        subjects.append([frame, x, y])
    if args.subjects_json:
        try:
            subjects.extend(json.loads(args.subjects_json))
        except (ValueError, TypeError):
            parser.error(tr("--subjects-json must be a JSON list"))

    if args.backend:
        from . import backend
        return run_cli(args.json_progress, lambda report: backend.control(args.backend, progress=report), "")

    if args.check:
        try:
            pipeline.check_runtime()
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(tr("Alpha Gen: ComfyUI, LTX-2.5 workflow, and adapter are ready."))
        return 0

    missing = [name for name in ("input", "output", "first", "last") if getattr(args, name) is None]
    if missing:
        parser.error("required unless --check: " + ", ".join("--" + name.replace("_", "-") for name in missing))

    id_map = None
    if args.id_map or args.ids:
        try:
            id_map = {"pattern": args.id_map, "ids": [int(value) for value in (args.ids or "").split(",") if value]}
        except ValueError:
            parser.error(tr("--ids expects comma-separated integers"))

    def work(report):
        if args.analyze_subjects is not None:
            return pipeline.analyze_subjects(
                args.input, args.first, args.last, args.output, args.analyze_subjects, progress=report,
            )
        if args.preview_subjects:
            return pipeline.preview_subjects(
                args.input, args.first, args.last, args.output, subjects, progress=report,
            )
        return pipeline.run(
            args.input, args.first, args.last, args.output,
            fps=args.fps, resolution=args.resolution, seed=args.seed,
            layer=args.layer, force=args.force, progress=report,
            subjects=subjects or None, low_vram=args.low_vram, id_map=id_map,
        )

    return run_cli(
        args.json_progress, work,
        tr("Use 544p, close other ComfyUI generations, or free GPU memory before retrying."),
    )


if __name__ == "__main__":
    sys.exit(main())
