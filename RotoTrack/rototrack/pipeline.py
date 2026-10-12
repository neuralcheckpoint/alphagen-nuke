"""Track a matte through a frame range with SAM 2.1 and cache it as EXRs."""

import collections
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from aitools_engine import imageio
from aitools_engine.events import emit as _emit, read_manifest as _read_manifest

from . import prompts

CACHE_VERSION = 2
MANIFEST_NAME = "rototrack.json"
SOFTWARE = "RotoTrack (SAM 2.1)"
MODES = ("shape", "matte")
WRITERS = 2
DEFAULT_LAYER = "rototrack.alpha"
_LAYER_NAME = re.compile(r"\A[A-Za-z0-9_.]+\Z")


def _digest(keys, alphas):
    """Fingerprint of the prompt, so redrawing it makes the cache stale."""
    digest = hashlib.sha1()
    for key in keys:
        digest.update(str(key).encode())
        digest.update(np.packbits(alphas[key] > prompts.THRESHOLD).tobytes())
    return digest.hexdigest()


def run(
    input_pattern,
    first,
    last,
    output_pattern,
    prompt_pattern,
    keys,
    mode="shape",
    size="large",
    transfer="auto",
    layer=DEFAULT_LAYER,
    force=False,
    progress=None,
    tracker=None,
):
    """Write one EXR per frame with the tracked matte in the ``layer`` channel.

    ``keys`` are the frames where ``prompt_pattern`` holds the prompt: a
    rough shape (``mode`` "shape") or an exact matte ("matte"). The tracker
    runs forward from the first key to the end, then back from it to the
    start; later keys correct the track as it passes them. With the same
    parameters and prompt as the cache on disk, only missing frames are
    written, and nothing runs when none is missing.
    """
    if last < first:
        raise ValueError("Last frame {} is before first frame {}".format(last, first))
    if mode not in MODES:
        raise ValueError("Unknown prompt mode {!r}; use one of {}".format(mode, MODES))
    if not _LAYER_NAME.match(layer):
        raise ValueError(
            "The layer {!r} is not a channel name: use letters, digits, '_' and '.' only".format(layer))
    keys = sorted(set(int(key) for key in keys))
    if not keys:
        raise ValueError("No key frames: the prompt has to be read on at least one frame")
    outside = [key for key in keys if not first <= key <= last]
    if outside:
        raise ValueError("Key frames {} are outside the range {}-{}".format(outside, first, last))
    for pattern, what in ((output_pattern, "Output"), (prompt_pattern, "Prompt")):
        if not imageio.has_frame_token(pattern):
            raise ValueError("{} {!r} has no frame number; use a pattern such as cache/matte.####.exr".format(what, pattern))
    output_dir = os.path.dirname(os.path.abspath(output_pattern))
    os.makedirs(output_dir, exist_ok=True)

    alphas = {key: imageio.read_alpha(imageio.frame_path(prompt_pattern, key)) for key in keys}
    for key, alpha in alphas.items():
        if not (alpha > prompts.THRESHOLD).any():
            raise ValueError("The prompt is empty on key frame {}".format(key))

    params = {
        "version": CACHE_VERSION,
        "input": input_pattern,
        "input_stamp": imageio.sequence_stamp(input_pattern, range(first, last + 1)),
        "first": first,
        "last": last,
        "keys": keys,
        "prompt": _digest(keys, alphas),
        "mode": mode,
        "size": size,
        "transfer": transfer,
        "layer": layer,
    }
    manifest = os.path.join(output_dir, MANIFEST_NAME)
    resume = not force and _read_manifest(manifest) == params
    if not resume and os.path.exists(manifest):
        os.remove(manifest)

    frames = list(range(first, last + 1))
    done = {
        frame: resume and os.path.isfile(imageio.frame_path(output_pattern, frame))
        for frame in frames
    }
    todo = [frame for frame in frames if not done[frame]]
    _emit(progress, "start", frames=len(frames), todo=len(todo))
    if not todo:
        _emit(progress, "done", written=0)
        return 0

    # Before the model takes VRAM: a prompt drawn over another format is the
    # artist's mistake, not a reason to load 2 GB of weights first.
    probe = imageio.read_frame(imageio.frame_path(input_pattern, first))
    for key, alpha in alphas.items():
        if alpha.shape != (probe.height, probe.width):
            raise ValueError("The prompt on frame {} is {}x{}, the plate {}x{}".format(
                key, alpha.shape[1], alpha.shape[0], probe.width, probe.height))

    if tracker is None:
        from .model import Tracker

        tracker = Tracker(size)
    from .model import Frames

    linear = None if transfer == "auto" else transfer == "linear"
    plate = Frames(input_pattern, frames, tracker.predictor.image_size, tracker.device, linear)
    free = tracker.free_memory() if hasattr(tracker, "free_memory") else None
    _emit(progress, "setup", width=plate.video_width, height=plate.video_height, size=size,
          keys=keys, mode=mode, free_vram_mb=None if free is None else free // 2 ** 20)
    with open(manifest, "w") as handle:
        json.dump(params, handle, indent=2)

    reference = imageio.read_frame(imageio.frame_path(input_pattern, first))
    writer = ThreadPoolExecutor(max_workers=WRITERS)
    writes = collections.deque()
    seen = set()
    written = 0

    def store(index, logits):
        nonlocal written
        if index in seen:
            return
        seen.add(index)
        frame = frames[index]
        if not done[frame]:
            matte = torch.sigmoid(logits[0, 0].float()).cpu().numpy()
            while len(writes) >= 2 * WRITERS:
                writes.popleft().result()
            path = imageio.frame_path(output_pattern, frame)
            writes.append(writer.submit(imageio.write_channels, path, reference, {layer: matte}, SOFTWARE))
            done[frame] = True
            written += 1
        _emit(progress, "progress", done=len(seen), total=len(frames), frame=frame)

    predictor = tracker.predictor
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if tracker.device.type == "cuda" else torch.autocast("cpu", enabled=False)
    try:
        with torch.inference_mode(), autocast:
            state = tracker.start(plate)
            for key in keys:
                index = key - first
                if mode == "shape":
                    box, points = prompts.shape_prompt(alphas[key])
                    predictor.add_new_points_or_box(
                        state, frame_idx=index, obj_id=1,
                        points=np.array(points, np.float32), labels=np.ones(len(points), np.int32),
                        box=np.array(box, np.float32),
                    )
                else:
                    predictor.add_new_mask(state, frame_idx=index, obj_id=1, mask=prompts.matte_prompt(alphas[key]))
            for reverse in (False, True):
                for index, _objects, logits in predictor.propagate_in_video(state, reverse=reverse):
                    store(index, logits)
        while writes:
            writes.popleft().result()
    finally:
        plate.close()
        writer.shutdown(wait=True)
    _emit(progress, "done", written=written)
    return written
