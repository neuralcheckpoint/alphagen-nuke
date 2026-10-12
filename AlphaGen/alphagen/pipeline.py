"""LTX-2.5 Alpha Gen orchestration and matte-cache output."""

import glob
import json
import math
import os
import re
import shutil
import tempfile
import time
import uuid
from fractions import Fraction
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import av
import numpy as np
import torch
from PIL import Image, ImageFilter
import websocket

from aitools_engine import imageio
from aitools_engine.events import emit, read_manifest
from aitools_engine.i18n import tr
from . import person, subject

CACHE_VERSION = 3
MODEL = "ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors"
MAX_WIDTH = 1920
MAX_HEIGHT = 1088
CHUNK_FRAMES = 121
OVERLAP_FRAMES = 16
RESOLUTIONS = (544, 768, 1088)
SOFTWARE = "Alpha Gen (LTX-2.5)"
SUBJECT_DILATE_PX = 4
SUBJECT_BLUR_PX = 1.5
_LAYER = re.compile(r"\A[A-Za-z0-9_.]+\Z")
# Nodes of workflow_api.json whose websocket events drive the progress bar.
SAMPLER = "5516:4829"
DECODER = "5518:5538"


def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _settings():
    """``medratools.json`` written by the Medratools installer beside the tool folder."""
    path = os.path.join(os.path.dirname(project_root()), "medratools.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return {}


def comfy_root():
    path = os.environ.get("COMFYUI_ROOT") or _settings().get("comfyui_root")
    if not path:
        path = os.path.join(project_root(), "../../../../ComfyUI")
    return os.path.abspath(os.path.expanduser(path))


def workflow_path():
    override = os.environ.get("ALPHAGEN_WORKFLOW")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflow_api.json")


# API-format loaders and the ComfyUI model folder each one reads from.
_LOADERS = {
    "UNETLoader": ("unet_name", "diffusion_models"),
    "CLIPLoader": ("clip_name", "text_encoders"),
    "VAELoader": ("vae_name", "vae"),
    "LTXICLoRALoaderModelOnly": ("lora_name", "loras"),
}


def _model_paths(root, workflow):
    paths = set()
    for node in workflow.values():
        loader = _LOADERS.get(node.get("class_type"))
        if loader:
            paths.add(os.path.join(root, "models", loader[1], node["inputs"][loader[0]]))
    return sorted(paths)


def _file_stamp(path):
    try:
        info = os.stat(path)
    except OSError:
        return None
    return [info.st_size, info.st_mtime_ns]


def _input(workflow, node_id, class_type, name):
    node = workflow.get(node_id)
    if not node or node.get("class_type") != class_type or name not in node.get("inputs", {}):
        raise ValueError("The Alpha Gen workflow changed: expected {} {} with input {}".format(
            class_type, node_id, name
        ))
    return node["inputs"]


def prepare_workflow(workflow, source, output_prefix, resolution, seed, low_vram=False):
    """Fill the bundled API graph for one clip; the adapter and empty prompts are fixed in it."""
    _input(workflow, "5001", "LoadVideo", "file")["file"] = source
    _input(workflow, "4852", "SaveVideo", "filename_prefix")["filename_prefix"] = output_prefix
    _input(workflow, "5548:5026", "ResizeImageMaskNode", "resize_type.shorter_size")[
        "resize_type.shorter_size"] = int(resolution)
    _input(workflow, "5516:4832", "RandomNoise", "noise_seed")["noise_seed"] = int(seed)
    if low_vram:
        decoder = _input(workflow, DECODER, "VAEDecodeTiled", "tile_size")
        decoder.update(tile_size=256, temporal_size=32, temporal_overlap=8)
    lora = _input(workflow, "5004:5606", "LTXICLoRALoaderModelOnly", "lora_name")
    if lora["lora_name"] != MODEL or lora["strength_model"] != 1.0:
        raise ValueError("The Alpha Gen workflow must load {} at strength 1.0".format(MODEL))
    return workflow


def layout(width, height, requested_resolution):
    """Fit the plate without stretching; pad to 32-pixel dimensions."""
    width, height = int(width), int(height)
    requested_resolution = int(requested_resolution)
    if width < 1 or height < 1:
        raise ValueError("Source dimensions must be positive")
    if requested_resolution not in RESOLUTIONS:
        raise ValueError("Resolution must be one of {}".format(", ".join(map(str, RESOLUTIONS))))

    if width >= height:
        max_short = min(MAX_HEIGHT, MAX_WIDTH * height / width)
    else:
        max_short = min(MAX_WIDTH, MAX_HEIGHT * width / height)
    short = min(requested_resolution, min(width, height), max_short)
    short = max(32, int(short // 32) * 32)
    scale = min(1.0, short / min(width, height))
    content_width = max(1, int(round(width * scale)))
    content_height = max(1, int(round(height * scale)))
    canvas_width = int(math.ceil(content_width / 32.0) * 32)
    canvas_height = int(math.ceil(content_height / 32.0) * 32)
    if canvas_width > MAX_WIDTH or canvas_height > MAX_HEIGHT:
        raise ValueError("The source aspect ratio exceeds Alpha Gen's 1920x1088 limit")
    return {
        "width": canvas_width,
        "height": canvas_height,
        "content_width": content_width,
        "content_height": content_height,
        "left": (canvas_width - content_width) // 2,
        "top": (canvas_height - content_height) // 2,
        "shorter_size": min(canvas_width, canvas_height),
    }


def plan_chunks(first, last):
    """Keep full temporal context at both ends; overlap by at least 16 frames."""
    first, last = int(first), int(last)
    if last < first:
        raise ValueError("Last frame is before first frame")
    if last - first + 1 <= CHUNK_FRAMES:
        ranges = [(first, last)]
    else:
        ranges = [(first, first + CHUNK_FRAMES - 1)]
        while ranges[-1][1] < last:
            start = min(ranges[-1][0] + CHUNK_FRAMES - OVERLAP_FRAMES,
                        last - CHUNK_FRAMES + 1)
            ranges.append((start, start + CHUNK_FRAMES - 1))

    result = []
    for start, end in ranges:
        count = end - start + 1
        padded = max(9, int(math.ceil((count - 1) / 8.0) * 8 + 1))
        if padded > CHUNK_FRAMES:
            raise AssertionError("chunk padding exceeded the Alpha Gen limit")
        result.append({"first": start, "last": end, "count": count, "padded": padded})
    return result


def _blend_weight(frame, first, last):
    """Return the later clip's linear weight over one inclusive overlap."""
    if first == last:
        return 0.5
    return (frame - first) / float(last - first)


def _fit_rgb(frame, target):
    rgb = imageio.to_display(torch.from_numpy(frame.rgb), frame.is_float).clamp(0.0, 1.0).numpy()
    pixels = np.asarray(np.rint(rgb * 255.0), dtype=np.uint8)
    content = Image.fromarray(pixels, "RGB")
    if content.size != (target["content_width"], target["content_height"]):
        content = content.resize(
            (target["content_width"], target["content_height"]), Image.Resampling.LANCZOS
        )
    canvas = Image.new("RGB", (target["width"], target["height"]), (0, 0, 0))
    canvas.paste(content, (target["left"], target["top"]))
    return np.asarray(canvas, dtype=np.uint8)


def _write_input_video(input_pattern, chunk, target, path, fps):
    rate = Fraction(str(fps)).limit_denominator(1001)
    time_base = Fraction(rate.denominator, rate.numerator)
    output = av.open(path, mode="w")
    video = output.add_stream("libx264", rate=rate)
    video.width = target["width"]
    video.height = target["height"]
    video.pix_fmt = "yuv420p"
    video.options = {"crf": "12", "preset": "veryfast"}
    audio = output.add_stream("aac", rate=48000)
    audio.layout = "stereo"
    audio_pts = 0
    target_samples = int(math.ceil(chunk["padded"] * 48000 / float(rate)))
    silence = np.zeros((2, 1024), dtype=np.float32)
    try:
        for offset in range(chunk["padded"]):
            frame_number = min(chunk["first"] + offset, chunk["last"])
            plate = imageio.read_frame(imageio.frame_path(input_pattern, frame_number))
            if (plate.width, plate.height) != (target["source_width"], target["source_height"]):
                raise ValueError("Source display dimensions changed at frame {}".format(frame_number))
            pixels = _fit_rgb(plate, target)
            picture = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            picture.pts = offset
            picture.time_base = time_base
            for packet in video.encode(picture):
                output.mux(packet)
            while audio_pts < target_samples and audio_pts * time_base.denominator <= (offset + 1) * 48000 * time_base.numerator:
                sound = av.AudioFrame.from_ndarray(silence, format="fltp", layout="stereo")
                sound.sample_rate = 48000
                sound.pts = audio_pts
                sound.time_base = Fraction(1, 48000)
                for packet in audio.encode(sound):
                    output.mux(packet)
                audio_pts += 1024
        for packet in video.encode():
            output.mux(packet)
        for packet in audio.encode():
            output.mux(packet)
    finally:
        output.close()


def _read_matte(path, expected_frames):
    """Yield decoded grayscale frames without retaining a whole clip in RAM."""
    container = av.open(path)
    try:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise ValueError("ComfyUI returned a file with no video stream")
        count = 0
        for frame in container.decode(stream):
            if count >= expected_frames:
                raise ValueError("Alpha Gen returned more than {} frames".format(expected_frames))
            count += 1
            yield frame.to_ndarray(format="gray")
    finally:
        container.close()
    if count != expected_frames:
        raise ValueError(
            "Alpha Gen returned {} frames; expected {}".format(count, expected_frames)
        )


def _restore_matte(pixels, target, width, height):
    matte = Image.fromarray(np.asarray(pixels, dtype=np.uint8), "L")
    if matte.size != (target["width"], target["height"]):
        matte = matte.resize((target["width"], target["height"]), Image.Resampling.LANCZOS)
    x, y = target["left"], target["top"]
    matte = matte.crop((x, y, x + target["content_width"], y + target["content_height"]))
    if matte.size != (width, height):
        matte = matte.resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(matte, dtype=np.float32) / 255.0


def _runtime():
    root = comfy_root()
    host = os.environ.get("COMFY_HOST", "127.0.0.1")
    port = int(os.environ.get("COMFY_PORT") or _settings().get("port") or 8188)
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise RuntimeError(tr("Alpha Gen requires a local ComfyUI server because its clips use ComfyUI's input folder"))
    if not 1 <= port <= 65535:
        raise RuntimeError(tr("COMFY_PORT must be between 1 and 65535"))
    return {
        "root": root,
        "host": host,
        "port": port,
        "workflow": workflow_path(),
    }


def _address(config):
    host = config["host"]
    return "{}:{}".format("[{}]".format(host) if ":" in host else host, config["port"])


def _request(config, path, payload=None, timeout=30):
    url = "http://{}{}".format(_address(config), path)
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _comfy_queue_idle(config):
    queue = _request(config, "/queue", timeout=5)
    running = queue.get("queue_running")
    pending = queue.get("queue_pending")
    if not isinstance(running, list) or not isinstance(pending, list):
        raise RuntimeError(tr("ComfyUI returned an invalid queue status; refusing to start SAM 2.1"))
    return not running and not pending


def _require_comfy_idle(config):
    if not _comfy_queue_idle(config):
        raise RuntimeError(tr("ComfyUI became busy; Alpha Gen stopped before starting another GPU stage"))


def _free_comfy_models(config):
    """Unload idle ComfyUI models before the isolated SAM process uses the GPU."""
    _require_comfy_idle(config)
    url = "http://{}{}".format(_address(config), "/free")
    payload = json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8")
    request = Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=30) as response:
        response.read()
    _require_comfy_idle(config)


def _cut_to_subject(matte, mask):
    """Apply a conservative soft dilation; this can remove, not recover, subjects."""
    mask = np.clip(mask, 0.0, 1.0)
    if mask.shape != matte.shape:
        raise ValueError("SAM subject mask dimensions do not match the Alpha Gen matte")
    pixels = np.asarray(np.rint(mask * 255.0), dtype=np.uint8)
    expanded = Image.fromarray(pixels, "L").filter(
        ImageFilter.MaxFilter(SUBJECT_DILATE_PX * 2 + 1)
    ).filter(ImageFilter.GaussianBlur(SUBJECT_BLUR_PX))
    soft_mask = np.asarray(expanded, dtype=np.float32) / 255.0
    return matte * soft_mask


def _check_server(config):
    if not os.path.isfile(config["workflow"]):
        raise RuntimeError(tr("Alpha Gen workflow not found: {}", config["workflow"]))
    with open(config["workflow"], "r", encoding="utf-8") as handle:
        workflow = json.load(handle)
    missing = [path for path in _model_paths(config["root"], workflow) if not os.path.isfile(path)]
    if missing:
        raise RuntimeError(tr("Model files are missing (run the Medratools model download):\n{}", "\n".join(missing)))
    try:
        stats = _request(config, "/system_stats", timeout=5)
    except Exception as exc:
        raise RuntimeError(tr("ComfyUI is not reachable on port {} (start the Medratools backend): {}", config["port"], exc))
    absent = sorted(
        class_type for class_type in {node["class_type"] for node in workflow.values()}
        if not _request(config, "/object_info/" + class_type)
    )
    if absent:
        raise RuntimeError(tr("ComfyUI is missing nodes (ComfyUI-LTXVideo or an older ComfyUI?): {}", ", ".join(absent)))
    return stats


def _small_decode(stats, forced=False):
    if forced:
        return True
    try:
        total = int(stats["devices"][0]["vram_total"])
        if total <= 0:
            raise ValueError("non-positive VRAM")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("ComfyUI did not report valid device memory") from exc
    # Free VRAM excludes reusable loaded models; it is not the device budget.
    return total <= 8 * 1024 ** 3


def check_runtime():
    """Validate the workflow, model files, server and node types; never queues GPU work."""
    config = _runtime()
    _check_server(config)
    with open(config["workflow"], "r", encoding="utf-8") as handle:
        prepare_workflow(json.load(handle), "alphagen/preflight/clip.mp4", "alphagen/preflight/test", 544, 1)
    return config


def _rejection(body):
    try:
        payload = json.loads(body)
    except ValueError:
        return body[-2000:]
    error = payload.get("error") or {}
    lines = [" ".join(filter(None, [error.get("message"), error.get("details")]))]
    for node_id, node in (payload.get("node_errors") or {}).items():
        for item in node.get("errors", []):
            lines.append("{} {}: {} {}".format(
                node.get("class_type", ""), node_id, item.get("message", ""), item.get("details", "")
            ).strip())
    return "\n".join(line for line in lines if line)


def _failure(status):
    for kind, data in status.get("messages", []):
        if kind == "execution_error":
            return "{} {}: {}".format(data.get("node_type"), data.get("node_id"), data.get("exception_message", "").strip())
        if kind == "execution_interrupted":
            return "interrupted in ComfyUI"
    return status.get("status_str") or "workflow did not complete"


def _follow(message, prompt_id, graph, status):
    """Turn one ComfyUI websocket message about this prompt into ``status(text, fraction)``."""
    data = message.get("data") or {}
    if data.get("prompt_id") != prompt_id:
        return
    node = data.get("node")
    if message.get("type") == "executing" and node in graph:
        if "Loader" in graph[node]["class_type"]:
            status(tr("loading models (the first run reads ~40 GB)"), 0.0)
        elif node == DECODER:
            status(tr("decoding the matte"), 0.9)
    elif message.get("type") == "progress" and node == SAMPLER:
        status(tr("sampling step {}/{}", data["value"], data["max"]), 0.9 * data["value"] / max(data["max"], 1))


def _run_workflow(config, input_relative, output_prefix, resolution, seed, status, low_vram=False):
    with open(config["workflow"], "r", encoding="utf-8") as handle:
        graph = prepare_workflow(json.load(handle), input_relative, output_prefix, resolution, seed, low_vram)
    client_id = uuid.uuid4().hex
    # Step progress only exists on the websocket; completion is still read from /history.
    socket = websocket.create_connection("ws://{}/ws?clientId={}".format(_address(config), client_id), timeout=1)
    try:
        try:
            prompt_id = _request(config, "/prompt", {"prompt": graph, "client_id": client_id})["prompt_id"]
        except HTTPError as exc:
            raise RuntimeError(tr("ComfyUI rejected the Alpha Gen workflow:\n{}", _rejection(exc.read().decode("utf-8", "replace"))))
        status(tr("waiting for ComfyUI"), 0.0)

        # No overall timeout: a cold start loads ~40 GB of weights before the first step.
        next_check = 0.0
        while True:
            try:
                message = socket.recv()
            except websocket.WebSocketTimeoutException:
                message = None
            except websocket.WebSocketConnectionClosedException:
                raise RuntimeError(tr("ComfyUI dropped the Alpha Gen job; was the backend restarted?"))
            if isinstance(message, str):
                _follow(json.loads(message), prompt_id, graph, status)
            if time.monotonic() < next_check:
                continue
            next_check = time.monotonic() + 1.0
            entry = _request(config, "/history/" + prompt_id).get(prompt_id)
            if entry is None:
                queue = _request(config, "/queue")
                queued = {item[1] for item in queue.get("queue_running", []) + queue.get("queue_pending", [])}
                if prompt_id not in queued and _request(config, "/history/" + prompt_id).get(prompt_id) is None:
                    raise RuntimeError(tr("ComfyUI dropped the Alpha Gen job; was the backend restarted?"))
                continue
            outcome = entry.get("status", {})
            if outcome.get("status_str") != "success" or not outcome.get("completed"):
                raise RuntimeError(tr("ComfyUI Alpha Gen failed: {}", _failure(outcome)))
            break
    finally:
        socket.close()

    output_dir = os.path.join(config["root"], "output", *output_prefix.split("/")[:-1])
    candidates = sorted(glob.glob(os.path.join(output_dir, output_prefix.rsplit("/", 1)[-1] + "*.mp4")))
    if not candidates:
        raise RuntimeError(tr("ComfyUI completed without writing the Alpha Gen matte in {}", output_dir))
    return candidates[-1]


def _write_manifest(path, params):
    temporary = path + ".partial"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(params, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.remove(temporary)
        except OSError:
            pass
        raise


def _cache_matches(cached, params, config):
    if not isinstance(cached, dict) or any(cached.get(key) != value for key, value in params.items()):
        return False
    try:
        with open(config["workflow"], "r", encoding="utf-8") as handle:
            workflow = json.load(handle)
    except (OSError, ValueError):
        return True
    model_paths = _model_paths(config["root"], workflow)
    if any(not os.path.isfile(path) for path in model_paths):
        return True
    return (
        cached.get("workflow") == config["workflow"]
        and cached.get("workflow_stamp") == _file_stamp(config["workflow"])
        and cached.get("model_stamps") == [[path, _file_stamp(path)] for path in model_paths]
    )


def _check_subject_frames(input_pattern, reference, subjects):
    for frame_number in sorted({point[0] for point in subjects}):
        point_frame = imageio.read_frame(imageio.frame_path(input_pattern, frame_number))
        if (point_frame.width, point_frame.height) != (reference.width, reference.height):
            raise ValueError("Source display dimensions changed at subject frame {}".format(frame_number))


def preview_subjects(input_pattern, first, last, output_pattern, subjects, *, progress=None, config=None):
    """Track the SAM selection alone (no Alpha Gen) so the user can check it before Compute."""
    first, last = int(first), int(last)
    if last < first:
        raise ValueError("Last frame is before first frame")
    if not imageio.has_frame_token(output_pattern):
        raise ValueError("Preview output must be an EXR frame pattern such as preview/subject.####.exr")
    reference = imageio.read_frame(imageio.frame_path(input_pattern, first))
    selected = subject.normalize_subjects(subjects, first, last, reference.width, reference.height)
    if not selected:
        raise ValueError("Add a subject point before previewing the selection")
    _check_subject_frames(input_pattern, reference, selected)
    subject.runtime()
    try:
        _free_comfy_models(config or _runtime())
    except OSError:
        pass  # ComfyUI is not running, so it holds no GPU memory; a busy queue still raises.
    os.makedirs(os.path.dirname(os.path.abspath(output_pattern)), exist_ok=True)
    emit(progress, "setup", width=reference.width, height=reference.height,
         size=0, chunks=1, first=first, last=last)
    emit(progress, "status", message=tr("tracking selected subjects with SAM 2.1"), percent=0)
    subject.run_tracker(input_pattern, first, last, output_pattern, selected,
                        detector=person.paths(comfy_root()))
    emit(progress, "progress", done=1, total=1, frame=last)
    return last - first + 1


def analyze_subjects(input_pattern, first, last, output_pattern, key, *, progress=None, config=None):
    """Track every person found on frame ``key`` into an ID map (subjects.id) the Nuke node picks from."""
    first, last, key = int(first), int(last), int(key)
    if last < first:
        raise ValueError("Last frame is before first frame")
    if not first <= key <= last:
        raise ValueError("Analysis frame {} is outside {}-{}".format(key, first, last))
    if not imageio.has_frame_token(output_pattern):
        raise ValueError("Analysis output must be an EXR frame pattern such as subjects/id.####.exr")
    detector = person.paths(comfy_root())
    if detector is None:
        raise RuntimeError(tr("Person detection needs ComfyUI's YOLOv10 and ViTPose models in models/detection"))
    subject.runtime()
    try:
        _free_comfy_models(config or _runtime())
    except OSError:
        pass  # ComfyUI is not running, so it holds no GPU memory; a busy queue still raises.
    directory = os.path.dirname(os.path.abspath(output_pattern))
    os.makedirs(directory, exist_ok=True)
    try:
        os.remove(os.path.join(directory, "analysis.json"))  # only a finished analysis may be picked from
    except FileNotFoundError:
        pass
    reference = imageio.read_frame(imageio.frame_path(input_pattern, first))
    emit(progress, "setup", width=reference.width, height=reference.height,
         size=0, chunks=1, first=first, last=last)
    emit(progress, "status", message=tr("finding and tracking people with SAM 2.1"), percent=0)
    subject.run_tracker(input_pattern, first, last, output_pattern, [], detector=detector, analyze_key=key)
    emit(progress, "progress", done=1, total=1, frame=last)
    return last - first + 1


def _check_id_map(id_map, first, last):
    """Validate {"pattern", "ids"} from the node; returns its cache entry."""
    pattern, ids = id_map.get("pattern"), id_map.get("ids")
    if not pattern or not imageio.has_frame_token(pattern):
        raise ValueError("The person map must be an EXR frame pattern")
    if not ids or any(isinstance(i, bool) or not isinstance(i, int) or i < 1 for i in ids):
        raise ValueError("Picked people must be positive integers")
    manifest = os.path.join(os.path.dirname(os.path.abspath(pattern)), "analysis.json")
    analysis = read_manifest(manifest)
    if not isinstance(analysis, dict) or analysis.get("first", first + 1) > first or analysis.get("last", last - 1) < last:
        raise ValueError(tr("The person map does not cover {}-{}; press Analyze people again", first, last))
    return {"ids": sorted(set(ids)), "analysis": _file_stamp(manifest)}


def run(input_pattern, first, last, output_pattern, *, fps=24.0, resolution=544,
        seed=1234, layer="alphagen.alpha", force=False, progress=None,
        config=None, subjects=None, low_vram=False, id_map=None):
    """id_map: {"pattern": person map EXRs, "ids": [picked people]} from analyze_subjects;
    its people are kept together with whatever the SAM ``subjects`` select."""
    first, last = int(first), int(last)
    fps = float(fps)
    if last < first:
        raise ValueError("Last frame is before first frame")
    if not 1.0 <= fps <= 240.0:
        raise ValueError("Frame rate must be between 1 and 240")
    if not _LAYER.match(layer):
        raise ValueError("The output layer must contain only letters, digits, '_' and '.'")
    if not imageio.has_frame_token(output_pattern):
        raise ValueError("Output must be an EXR frame pattern such as cache/alpha.####.exr")
    config = config or _runtime()
    source_frames = list(range(first, last + 1))
    reference = imageio.read_frame(imageio.frame_path(input_pattern, first))
    selected_subjects = subject.normalize_subjects(
        subjects, first, last, reference.width, reference.height
    )
    id_entry = _check_id_map(id_map, first, last) if id_map else None
    sam_dependencies = None
    if selected_subjects:
        _check_subject_frames(input_pattern, reference, selected_subjects)
        # Fail before contacting ComfyUI or loading SAM if its dedicated runtime is absent.
        sam_dependencies = subject.cache_fingerprint()
    target = layout(reference.width, reference.height, resolution)
    target["source_width"], target["source_height"] = reference.width, reference.height
    chunks = plan_chunks(first, last)
    output_dir = os.path.dirname(os.path.abspath(output_pattern))
    os.makedirs(output_dir, exist_ok=True)

    params = {
        "version": CACHE_VERSION,
        "model": MODEL,
        "input": input_pattern,
        "input_stamp": imageio.sequence_stamp(input_pattern, source_frames),
        "first": first,
        "last": last,
        "fps": fps,
        "resolution": int(resolution),
        "processed_size": [target["width"], target["height"]],
        "seed": int(seed),
        "layer": layer,
        "subjects": selected_subjects,
        "low_vram": bool(low_vram),
        "chunk_frames": CHUNK_FRAMES,
        "overlap_frames": OVERLAP_FRAMES,
        "subject_mask": {
            "dilate_px": SUBJECT_DILATE_PX,
            "blur_px": SUBJECT_BLUR_PX,
        } if selected_subjects or id_entry else None,
        "sam_dependencies": sam_dependencies,
    }
    if id_entry:
        params["id_map"] = id_entry  # absent otherwise, so caches from before the map stay valid
    manifest = os.path.join(output_dir, "alphagen.json")
    cached = read_manifest(manifest)
    cache_valid = not force and _cache_matches(cached, params, config)
    cache_complete = cache_valid and all(
        os.path.isfile(imageio.frame_path(output_pattern, frame)) for frame in source_frames
    )
    if cache_complete:
        emit(progress, "setup", width=target["width"], height=target["height"],
             size=target["shorter_size"], chunks=len(chunks), first=first, last=last)
        return 0

    # A shot cache is valid only as a complete set of blended chunks. A partial
    # cache is rebuilt in full so no seam mixes old and new model estimates.
    try:
        os.remove(manifest)
    except FileNotFoundError:
        pass

    small_decode = _small_decode(_check_server(config), low_vram)

    subject_temp = None
    if selected_subjects:
        _free_comfy_models(config)
        subject_temp = tempfile.mkdtemp(prefix=".alphagen-subject-", dir=output_dir)
        mask_pattern = os.path.join(subject_temp, "subject.####.exr")
        emit(progress, "status", message=tr("tracking selected subjects with SAM 2.1"), percent=0)
        try:
            subject.run_tracker(
                input_pattern,
                first,
                last,
                mask_pattern,
                selected_subjects,
                on_poll=lambda: _require_comfy_idle(config),
                detector=person.paths(comfy_root()),
            )
        except BaseException:
            shutil.rmtree(subject_temp, ignore_errors=True)
            raise
    input_dir = output_job_dir = pending_dir = None
    try:
        # A user can start another ComfyUI job while SAM is running. Never
        # queue LTX on top of it; the SAM child has already exited at this point.
        if selected_subjects:
            _require_comfy_idle(config)

        with open(config["workflow"], "r", encoding="utf-8") as handle:
            workflow = json.load(handle)
        model_paths = _model_paths(config["root"], workflow)
        params.update({
            "model_stamps": [[path, _file_stamp(path)] for path in model_paths],
            "workflow": config["workflow"],
            "workflow_stamp": _file_stamp(config["workflow"]),
            "decode_tiles": [256, 64, 32, 8] if small_decode else [512, 64, 128, 32],
        })
        job_id = uuid.uuid4().hex
        input_dir = os.path.join(config["root"], "input", "alphagen", job_id)
        output_job_dir = os.path.join(config["root"], "output", "alphagen", job_id)
        os.makedirs(input_dir)
        pending_dir = tempfile.mkdtemp(prefix=".alphagen-pending-", dir=output_dir)
    except BaseException:
        if input_dir is not None:
            shutil.rmtree(input_dir, ignore_errors=True)
        if subject_temp is not None:
            shutil.rmtree(subject_temp, ignore_errors=True)
        raise
    selected = chunks
    emit(progress, "setup", width=target["width"], height=target["height"],
         size=target["shorter_size"], chunks=len(selected), first=first, last=last)
    written = 0
    try:
        for index, chunk in enumerate(selected, 1):
            clip_name = "clip_{:03d}.mp4".format(index)
            clip_path = os.path.join(input_dir, clip_name)
            _write_input_video(input_pattern, chunk, target, clip_path, fps)
            input_relative = "alphagen/{}/{}".format(job_id, clip_name)
            prefix = "alphagen/{}/clip_{:03d}".format(job_id, index)
            def status(message, fraction, index=index):
                emit(progress, "status", message=tr("chunk {}/{}: {}", index, len(chunks), message),
                     percent=int(100 * (index - 1 + fraction) / len(chunks)))

            if selected_subjects:
                _require_comfy_idle(config)
            matte_path = _run_workflow(
                config, input_relative, prefix, target["shorter_size"], seed, status, small_decode
            )
            previous = selected[index - 2] if index > 1 else None
            following = selected[index] if index < len(selected) else None
            overlap_first = chunk["first"] if previous is None else max(chunk["first"], previous["first"])
            overlap_last = chunk["last"] if previous is None else min(chunk["last"], previous["last"])
            for offset, matte_pixels in enumerate(_read_matte(matte_path, chunk["padded"])):
                if offset >= chunk["count"]:
                    continue
                frame_number = chunk["first"] + offset
                plate = imageio.read_frame(imageio.frame_path(input_pattern, frame_number))
                matte = _restore_matte(matte_pixels, target, plate.width, plate.height)
                mask = None
                if subject_temp is not None:
                    mask = imageio.read_alpha(imageio.frame_path(mask_pattern, frame_number))
                if id_entry:
                    ids = imageio.read_alpha(imageio.frame_path(id_map["pattern"], frame_number), subject.ID_LAYER)
                    picked = np.isin(np.rint(ids), id_entry["ids"]).astype(np.float32)
                    mask = picked if mask is None else np.maximum(mask, picked)
                if mask is not None:
                    if mask.shape != matte.shape:
                        raise ValueError("Subject mask dimensions do not match the Alpha Gen matte")
                    matte = _cut_to_subject(matte, mask)

                if previous is not None and previous["first"] <= frame_number <= previous["last"]:
                    overlap_path = os.path.join(pending_dir, "frame-{}.npy".format(frame_number))
                    with open(overlap_path, "rb") as handle:
                        old_matte = np.load(handle).astype(np.float32)
                    weight = _blend_weight(frame_number, overlap_first, overlap_last)
                    matte = old_matte * (1.0 - weight) + matte * weight
                    os.remove(overlap_path)

                if following is not None and following["first"] <= frame_number <= chunk["last"]:
                    overlap_path = os.path.join(pending_dir, "frame-{}.npy".format(frame_number))
                    with open(overlap_path, "wb") as handle:
                        np.save(handle, matte.astype(np.float16))
                else:
                    imageio.write_channels(
                        imageio.frame_path(output_pattern, frame_number), plate, {layer: matte}, SOFTWARE
                    )
                    written += 1
            try:
                os.remove(matte_path)
            except OSError:
                pass
            emit(progress, "progress", done=index, total=len(chunks), frame=chunk["last"])
        if os.listdir(pending_dir):
            raise RuntimeError("Alpha Gen finished with unblended overlap frames")
        _write_manifest(manifest, params)
        return written
    finally:
        if input_dir is not None:
            shutil.rmtree(input_dir, ignore_errors=True)
        if output_job_dir is not None:
            shutil.rmtree(output_job_dir, ignore_errors=True)
        if pending_dir is not None:
            shutil.rmtree(pending_dir, ignore_errors=True)
        if subject_temp is not None:
            shutil.rmtree(subject_temp, ignore_errors=True)
