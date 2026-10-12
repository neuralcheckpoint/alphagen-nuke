"""Shared plumbing of the Rafa Tools AI nodes for Nuke.

Each tool (SmartVectors, TemporalDepth, RotoTrack) is a Group whose Compute
runs the tool's engine, a GPU Python of its own that Nuke never imports, as
a subprocess printing one JSON event per line, and whose internal Read shows
the engine's EXR cache. This module holds what they share: finding and
starting the engine, the background job and its progress task, rendering
the Group's inputs for the engine, and the cache's paths and Read.
"""

import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

import nuke

_AITOOLS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AITOOLS not in sys.path:
    sys.path.insert(0, _AITOOLS)
from aitools_engine.i18n import tr  # noqa: E402  (standard library only)

# Computations in flight, keyed by the node's full name, so a second press
# of Compute doesn't start a second engine on the same node.
running = {}

MOVIE_EXTENSIONS = (".mov", ".mp4", ".mxf", ".avi", ".mkv")


def engine_python(project_root, env_var):
    """The tool's engine interpreter: ``env_var`` if set, else its .venv."""
    override = os.environ.get(env_var)
    if override:
        return override
    if sys.platform.startswith("win"):
        return os.path.join(project_root, ".venv", "Scripts", "python.exe")
    return os.path.join(project_root, ".venv", "bin", "python")


def engine_env():
    """Environment for an engine without Nuke's Python and library paths."""
    env = dict(os.environ)
    for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "LD_LIBRARY_PATH", "LD_PRELOAD"):
        env.pop(key, None)
    env["PYTHONUNBUFFERED"] = "1"
    return env


# ------------------------------------------------------------------- cache


def default_cache_pattern(node, folder):
    """``<script dir>/<folder>/<script>/<node>/<node>.####.exr``, or under ~/.cache when unsaved."""
    script = nuke.root().name()
    if script and script != "Root":
        base = os.path.join(os.path.dirname(script), folder, os.path.splitext(os.path.basename(script))[0])
    else:
        cache_home = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
        base = os.path.join(cache_home, folder, "untitled")
    name = node.fullName().replace(".", "_")
    return os.path.join(base, name, "{}.####.exr".format(name)).replace("\\", "/")


def normalize_cache_pattern(pattern, node_name):
    """Turn whatever was typed in a cache knob into an EXR frame pattern.

    A folder (picked in the file browser, often with a trailing slash) gets
    ``<node>.####.exr`` inside it; a file name without ``#`` or ``%d`` gets
    ``.####`` before an ``.exr`` extension.
    """
    pattern = pattern.strip().replace("\\", "/")
    name = node_name.replace(".", "_")
    if pattern.endswith("/") or os.path.isdir(pattern):
        return pattern.rstrip("/") + "/{}.####.exr".format(name)
    if "#" in pattern or re.search(r"%0?\d*d", pattern):
        return pattern
    root, extension = os.path.splitext(pattern)
    if extension.lower() != ".exr":
        root = pattern
    return root + ".####.exr"


def cache_pattern(node, folder):
    """The node's cache pattern, filled in from its cache knob or the default, and stored back."""
    typed = node["cache_pattern"].value()
    pattern = normalize_cache_pattern(typed, node.fullName()) if typed else default_cache_pattern(node, folder)
    node["cache_pattern"].setValue(pattern)
    return pattern


def add_cache_read(name="Cache"):
    """The Read that shows the cache, inside the current Group.

    It follows the Group's cache_pattern and frame range knobs and goes
    black outside the range or on a missing frame.
    """
    cache = nuke.nodes.Read(name=name)
    # [knob] keeps the #### pattern; [value] would hand the Read one frame's
    # file name, and outside the range it would then fail on a missing file
    # instead of going black.
    cache["file"].setValue("[knob parent.cache_pattern]")
    # Set explicitly; otherwise the Read guesses it during validate.
    cache["file_type"].setValue("exr")
    for knob in ("first", "origfirst"):
        cache[knob].setExpression("parent.range_first")
    for knob in ("last", "origlast"):
        cache[knob].setExpression("parent.range_last")
    cache["before"].setValue("black")
    cache["after"].setValue("black")
    cache["on_error"].setValue("black")
    cache["raw"].setValue(True)
    return cache


def reload_cache(node, name="Cache"):
    with node:
        cache = nuke.toNode(name)
        if cache is not None:
            cache["reload"].execute()


# ------------------------------------------------------------------ inputs


def direct_read(node, input_index=0):
    """The file pattern of a Read feeding the input straight, if the engine can read it as is.

    Not with the node's source_mode on "Render input", a Read with a frame
    offset or a script in proxy mode (Nuke shows other pixels than the
    files), or a movie file.
    """
    if node["source_mode"].value() != "Auto":
        return None
    source = node.input(input_index)
    if source is None or source.Class() != "Read":
        return None
    if source["frame"].value():
        return None
    if nuke.root()["proxy"].value():
        return None
    pattern = nuke.filename(source)
    if not pattern or pattern.lower().endswith(MOVIE_EXTENSIONS):
        return None
    return pattern


class Cancelled(Exception):
    """The user pressed Cancel on a progress task."""


def render_input(node, input_name, channels, frames, folder, name, canvas=None, title=None):
    """Render one of the Group's Input nodes to a half-float EXR sequence; return its pattern.

    ``frames`` is a list of (first, last). Always at full resolution: in
    proxy mode the Write would want a proxy file name, and engines need the
    full-size frames. ``canvas`` (width, height) puts the image on a canvas
    of that size without scaling it, pixel coordinates kept. The folder is
    emptied first.

    With ``title`` (a tool name) the render runs one frame at a time under a
    progress bar that can be cancelled; it raises Cancelled when the user
    presses it. Without it the whole range is handed to Nuke in one call, as
    before.
    """
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    pattern = os.path.join(folder, name + ".####.exr").replace("\\", "/")
    with node:
        upstream = nuke.toNode(input_name)
        conform = None
        if canvas is not None:
            conform = nuke.nodes.Reformat(name="AIToolsCanvas", inputs=[upstream])
            conform["type"].setValue("to box")
            conform["box_fixed"].setValue(True)
            conform["box_width"].setValue(canvas[0])
            conform["box_height"].setValue(canvas[1])
            conform["resize"].setValue("none")
            conform["center"].setValue(False)
            upstream = conform
        write = nuke.nodes.Write(name="AIToolsRender", inputs=[upstream])
    try:
        write["file"].setValue(pattern)
        write["file_type"].setValue("exr")
        write["channels"].setValue(channels)
        write["datatype"].setValue("16 bit half")
        write["compression"].setValue("Zip (1 scanline)")
        proxy = nuke.root()["proxy"].value()
        nuke.root()["proxy"].setValue(False)
        task = nuke.ProgressTask(tr("{}: rendering input", title)) if title and nuke.GUI else None
        total = sum(last - first + 1 for first, last in frames)
        rendered = 0
        try:
            for first, last in frames:
                # One frame per call only in the GUI: that is where the
                # progress bar and its Cancel live. In a terminal session a
                # whole range in one call is faster and just as good.
                if task is None:
                    nuke.execute(write, first, last)
                    rendered += last - first + 1
                    continue
                for frame in range(first, last + 1):
                    if task.isCancelled():
                        raise Cancelled()
                    task.setProgress(int(100 * rendered / max(total, 1)))
                    task.setMessage(tr("rendering frame {} ({}/{})", frame, rendered, total))
                    nuke.execute(write, frame, frame)
                    rendered += 1
        finally:
            if task is not None:
                del task
            nuke.root()["proxy"].setValue(proxy)
    finally:
        nuke.delete(write)
        if conform is not None:
            nuke.delete(conform)
    return pattern


def render_for_compute(node, title, input_name, channels, frames, folder, name, canvas=None,
                       status="rendering source..."):
    """``render_input`` for a Compute: status text, progress and a clean failure.

    Returns the pattern, or None after saying what went wrong, so a failed or
    cancelled render never reaches the engine and never leaves the node stuck
    on its rendering status.
    """
    node["status"].setValue(status)
    try:
        pattern = render_input(node, input_name, channels, frames, folder, name, canvas, title=title)
    except Cancelled:
        node["status"].setValue(tr("cancelled"))
        return None
    except Exception as exc:
        node["status"].setValue(tr("failed: {}", exc))
        nuke.tprint("{}: rendering the input failed:\n{}".format(title, traceback.format_exc()))
        if nuke.GUI:
            nuke.message(tr("{}: rendering the input failed:\n{}", title, exc))
        return None
    return pattern


def range_from_input(node, title):
    source = node.input(0)
    if source is None:
        nuke.message(tr("{}: connect a source first.", title))
        return
    node["range_first"].setValue(int(source.firstFrame()))
    node["range_last"].setValue(int(source.lastFrame()))


def check_can_compute(node, title, project_root, env_var):
    """The engine interpreter, or None after telling the user why not."""
    if node.fullName() in running:
        nuke.message(tr("{}: {} is already computing.", title, node.fullName()))
        return None
    if node.input(0) is None:
        nuke.message(tr("{}: connect a source first.", title))
        return None
    python = engine_python(project_root, env_var)
    if not os.path.isfile(python):
        nuke.message(tr("{} engine not installed in:\n{}\nRun the tool's installer first.", title, project_root))
        return None
    # Single-frame nodes (HDRI) have no range.
    if "range_first" in node.knobs() and int(node["range_last"].value()) < int(node["range_first"].value()):
        nuke.message(tr("{}: last frame is before first frame.", title))
        return None
    return python


# --------------------------------------------------------------------- job


def terminate_tree(process, timeout=5.0):
    """Stop a process and everything it started, then reap it.

    Engines run in their own session (``start_new_session``), so the group is
    signalled at once: a hung engine can't hold the node on "computing"
    while the workers it spawned stay alive. SIGTERM gets ``timeout``
    seconds, then the group gets SIGKILL.
    """
    if process is None or process.poll() is not None:
        return
    if os.name != "nt":
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except OSError:
            process.terminate()
    else:
        process.terminate()
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name != "nt":
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except OSError:
            process.kill()
    else:
        process.kill()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass


class EngineJob(threading.Thread):
    """Runs an engine command in the background, feeding Nuke's progress task.

    Hooks, all optional: ``describe_setup(event)`` turns the engine's setup
    event into text for the task and the status ("<verb> <text>");
    ``on_event(job, event)`` sees the events the job doesn't know;
    ``finished(job, node)`` returns the status text once the cache is
    reloaded. ``rendered`` folders are deleted when the job ends, failed or
    not: they are redone on every Compute.
    """

    def __init__(self, node, command, cwd, title, verb=tr("computing"), rendered=(),
                 describe_setup=None, on_event=None, finished=None):
        threading.Thread.__init__(self, name="{} {}".format(title, node.fullName()))
        self.daemon = True
        self.node_name = node.fullName()
        self.command = command
        self.cwd = cwd
        self.title = title
        self.verb = verb
        self.rendered = [folder for folder in rendered if folder]
        self.describe_setup = describe_setup
        self.on_event = on_event
        self.finished_text = finished
        self.result = None
        self.error = None
        self.log = []

    def start_on(self, node, wait=False):
        running[self.node_name] = self
        node["status"].setValue(tr("starting engine..."))
        self.start()
        if wait:
            self.join()
        return self

    def run(self):
        task = nuke.ProgressTask("{}: {}".format(self.title, self.node_name))
        started = time.time()
        process = None
        lines = queue.Queue()
        try:
            process = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                env=engine_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                # Its own session, so a stuck engine and whatever it spawned
                # (CUDA workers, helpers) can be stopped as one group.
                start_new_session=(os.name != "nt"),
            )
            # The pipe is drained on a thread: reading it inline would only
            # notice Cancel when the engine happened to print a line, and a
            # silent engine (loading weights, hung on CUDA) never did.
            reader = threading.Thread(
                target=self._drain, args=(process, lines),
                name="{} output".format(self.node_name), daemon=True,
            )
            reader.start()
            self._consume(process, lines, reader, task, started)
            if self.error is None:
                self._wait_after_output(process)
            if self.error is None and process.returncode != 0:
                self.error = "engine exited with code {}".format(process.returncode)
        except Exception as exc:
            self.error = str(exc)
        finally:
            terminate_tree(process)
            del task
            on_main_thread(self._finish)

    @staticmethod
    def _drain(process, lines):
        """Read the engine's output to a queue that the job polls with a timeout."""
        try:
            for text in process.stdout:
                lines.put(text)
        except (OSError, ValueError):
            pass
        finally:
            lines.put(None)

    def _consume(self, process, lines, reader, task, started):
        """Handle engine lines, checking Cancel every quarter second."""
        while True:
            if task.isCancelled():
                self.error = "cancelled"
                return
            try:
                text = lines.get(timeout=0.25)
            except queue.Empty:
                # The reader is gone without its end marker (killed VM, dead
                # pipe): stop instead of polling a queue nobody feeds.
                if not reader.is_alive():
                    return
                continue
            if text is None:
                return
            self._handle(text.strip(), task, started)

    def _wait_after_output(self, process):
        """Reap an engine whose output ended, with a deadline."""
        deadline = time.time() + 30
        while process.poll() is None and time.time() < deadline:
            time.sleep(0.1)
        if process.poll() is None:
            self.error = "engine still running after its output ended"

    def _handle(self, line, task, started):
        self.log.append(line)
        try:
            event = json.loads(line)
        except ValueError:
            return
        kind = event.get("event")
        if kind == "setup":
            if self.describe_setup:
                message = self.describe_setup(event)
                task.setMessage(message)
                self.status("{} {}".format(self.verb, message))
        elif kind == "status":
            task.setProgress(event["percent"])
            task.setMessage(event["message"])
            self.status("{} {}".format(self.verb, event["message"]))
        elif kind == "progress":
            done, total = event["done"], event["total"]
            task.setProgress(int(100 * done / max(total, 1)))
            remaining = (time.time() - started) / max(done, 1) * (total - done)
            task.setMessage(tr("frame {} ({}/{}), ~{:.0f}s left", event["frame"], done, total, remaining))
        elif kind == "error":
            self.error = event.get("message", tr("unknown error"))
        elif kind == "finished":
            self.result = event
        elif self.on_event:
            self.on_event(self, event)

    def status(self, text):
        on_main_thread(set_status, self.node_name, text)

    def _finish(self):
        running.pop(self.node_name, None)
        for folder in self.rendered:
            shutil.rmtree(folder, ignore_errors=True)
        node = nuke.toNode(self.node_name)
        if node is None:
            return
        if self.error:
            node["status"].setValue(tr("cancelled") if self.error == "cancelled" else tr("failed: {}", self.error))
            if self.error != "cancelled":
                nuke.tprint("{} engine log:\n{}".format(self.title, "\n".join(self.log[-15:])))
                if nuke.GUI:
                    nuke.message(tr("{} failed:\n{}", self.title, self.error))
            return
        reload_cache(node)
        if self.finished_text:
            node["status"].setValue(self.finished_text(self, node))
        else:
            node["status"].setValue(tr("cached {} new frames in {:.1f}s", self.result["written"], self.result["seconds"]))


def on_main_thread(function, *args):
    # Terminal sessions have no event loop to deliver executeInMainThread.
    if nuke.GUI:
        nuke.executeInMainThread(function, args=args)
    else:
        function(*args)


def set_status(node_name, text):
    node = nuke.toNode(node_name)
    if node is not None:
        node["status"].setValue(text)
