"""Control the configured local ComfyUI with its installed CLI."""

import os
import shutil
import subprocess
import time
from urllib.error import HTTPError

from aitools_engine.events import emit
from aitools_engine.i18n import tr
from . import pipeline


def online(config):
    try:
        stats = pipeline._request(config, "/system_stats", timeout=2)
    except HTTPError:
        raise RuntimeError(tr("The backend port is occupied by an unexpected HTTP service"))
    except OSError:
        return False
    if not isinstance(stats, dict) or "system" not in stats or "devices" not in stats:
        raise RuntimeError(tr("The backend port is occupied by an unexpected HTTP service"))
    return True


def control(action, *, config=None, progress=None):
    config = config or pipeline._runtime()
    active = online(config)
    if action not in ("start", "stop", "status"):
        raise ValueError("Unknown backend action")
    if action == "status" or active == (action == "start"):
        emit(progress, "backend", message=tr("backend online" if active else "backend offline"))
        return 0
    cli = shutil.which("comfy")
    if cli is None:
        raise RuntimeError(tr("Install comfy-cli to control the backend from Nuke"))
    if action == "stop":
        pipeline._require_comfy_idle(config)
        command = [cli, "--workspace", config["root"], "stop", "--port", str(config["port"])]
    else:
        command = [cli, "--workspace", config["root"], "launch", "--background", "--",
                   "--listen", config["host"], "--port", str(config["port"]), "--vram-headroom", "1"]
    environment = os.environ.copy()
    environment["PATH"] = os.path.dirname(cli) + os.pathsep + environment.get("PATH", "")
    emit(progress, "status", message=tr("starting backend..." if action == "start" else "stopping backend..."), percent=0)
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip()[-3000:])
    deadline = time.monotonic() + 120
    while online(config) != (action == "start"):
        if time.monotonic() >= deadline:
            raise RuntimeError(tr("Backend did not reach the requested state; check comfy logs"))
        time.sleep(0.5)
    emit(progress, "backend", message=tr("backend online" if action == "start" else "backend offline"))
    return 0
