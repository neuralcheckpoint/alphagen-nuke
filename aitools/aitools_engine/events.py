"""Progress events, cache manifests and the command line wrapper every engine uses.

Engines report through ``emit`` as dictionaries; with ``--json-progress``
the command line prints each one as a JSON line, which is what the Nuke
plugins read to drive their progress bar and status.
"""

import json
import time
import traceback

from .i18n import tr


def emit(progress, event, **fields):
    if progress:
        fields["event"] = event
        progress(fields)


def read_manifest(path):
    """The parameters a cache was made with, or {} when there is none to trust."""
    try:
        with open(path, "r") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def run_cli(json_progress, work, out_of_memory_hint):
    """Run ``work(report)`` and report its end as a "finished" or "error" event.

    ``work`` returns the number of frames written. Returns the process exit
    status.
    """
    started = time.time()

    def report(event):
        if json_progress:
            print(json.dumps(event), flush=True)
        elif event["event"] == "progress":
            print("frame {frame} ({done}/{total})".format(**event), flush=True)
        else:
            print(event, flush=True)

    try:
        written = work(report)
    except Exception as exc:
        message = str(exc)
        if type(exc).__name__ == "OutOfMemoryError":
            message = tr("GPU out of memory. ") + out_of_memory_hint
        report({"event": "error", "message": message, "traceback": traceback.format_exc()})
        return 1
    report({"event": "finished", "written": written, "seconds": round(time.time() - started, 2)})
    return 0
