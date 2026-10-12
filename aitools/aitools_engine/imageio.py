"""Image sequence I/O through OpenImageIO, for every tool's engine.

Plates are read over their display window and handed to the models
display-referred; caches are written as half-float EXRs over the plate's
display window, atomically.
"""

import hashlib
import os
import re

import numpy as np
import OpenImageIO as oiio
import torch

_HASHES = re.compile(r"#+")
_PRINTF = re.compile(r"%(0?\d*)d")
_STAMP_BYTES = 256 * 1024


def has_frame_token(pattern):
    return bool(_PRINTF.search(pattern) or _HASHES.search(pattern))


def frame_path(pattern, frame):
    """Expand a Nuke-style ``####`` or printf ``%04d`` pattern for ``frame``."""
    if _PRINTF.search(pattern):
        return pattern % frame
    match = _HASHES.search(pattern)
    if match:
        width = len(match.group(0))
        return pattern[: match.start()] + str(frame).zfill(width) + pattern[match.end() :]
    return pattern


def _sample_hash(path, size):
    """A blake2b of the first and last 256 KB of a file (whole file when small)."""
    digest = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as handle:
        digest.update(handle.read(_STAMP_BYTES))
        if size > 2 * _STAMP_BYTES:
            handle.seek(-_STAMP_BYTES, os.SEEK_END)
            digest.update(handle.read(_STAMP_BYTES))
    return digest.hexdigest()


def sequence_stamp(pattern, frames, hashes=4):
    """What a cache has to remember about its input to notice the input changed.

    A plate re-rendered to the same path keeps its file names, so a cache
    keyed only by the pattern silently reuses the old pixels' result. Every
    frame is stamped with its size and mtime (ns), which a re-render touches
    for the whole sequence; ``hashes`` frames spread over the range also get
    a hash of their first and last 256 KB, so a rewrite that lands on the
    same byte count within the same second is still noticed. A frame that is
    missing is stamped as such: a cache made before its plate existed never
    counts as resumed.
    """
    frames = list(frames)
    if not frames:
        return []
    if hashes >= len(frames):
        sampled = set(range(len(frames)))
    elif hashes <= 1:
        sampled = {0}
    else:
        sampled = {round(index * (len(frames) - 1) / (hashes - 1)) for index in range(hashes)}
    stamp = []
    for index, frame in enumerate(frames):
        path = frame_path(pattern, frame)
        try:
            info = os.stat(path)
            digest = _sample_hash(path, info.st_size) if index in sampled else None
            stamp.append([frame, info.st_size, info.st_mtime_ns, digest])
        except OSError:
            stamp.append([frame, None, None, None])
    return stamp


class Frame(object):
    """An RGB plate frame in its display window and whether it is linear.

    ``width``/``height`` and ``origin`` describe the display window; the
    cache is written over exactly that window.
    """

    def __init__(self, rgb, spec, is_float, alpha=None):
        self.rgb = rgb
        self.width = spec.full_width
        self.height = spec.full_height
        self.origin = (spec.full_x, spec.full_y)
        self.is_float = is_float
        self.alpha = alpha


def _into_display_window(pixels, spec):
    """Place the data window inside a black display-window canvas.

    Nuke writes EXRs whose data window follows the bbox, so it moves with
    animated content and can be smaller (cropped) or larger (overscan) than
    the format. Models have to see the fixed display window.
    """
    if (spec.x, spec.y, spec.width, spec.height) == (
        spec.full_x, spec.full_y, spec.full_width, spec.full_height
    ):
        return pixels
    canvas = np.zeros((spec.full_height, spec.full_width, pixels.shape[2]), dtype=pixels.dtype)
    left, top = spec.x - spec.full_x, spec.y - spec.full_y
    x0, y0 = max(left, 0), max(top, 0)
    x1 = min(left + spec.width, spec.full_width)
    y1 = min(top + spec.height, spec.full_height)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = pixels[y0 - top:y1 - top, x0 - left:x1 - left]
    return canvas


def _alpha_index(spec):
    """The alpha channel of a spec, or None when it has none."""
    names = [name.split(".")[-1].lower() for name in spec.channelnames]
    if "a" in names:
        return names.index("a")
    if "alpha" in names:
        return names.index("alpha")
    if spec.nchannels == 1:
        return 0
    return None


def read_frame(path, alpha=False):
    """Read a frame in its display window; with ``alpha`` also its alpha.

    With ``alpha=True`` the channels are read in one pass (rgb and alpha
    together; a second pass over a 4K frame is ~250 MB of I/O) and the alpha
    lands in ``frame.alpha`` (None when the file has none).
    """
    image = oiio.ImageInput.open(path)
    if image is None:
        raise IOError("Cannot open {}: {}".format(path, oiio.geterror()))
    index = None
    try:
        spec = image.spec()
        if alpha:
            index = _alpha_index(spec)
        count = max(3, (index or 0) + 1) if index is not None else min(spec.nchannels, 3)
        count = min(count, spec.nchannels)
        pixels = image.read_image(0, 0, 0, count, oiio.FLOAT)
        if pixels is None:
            raise IOError("Cannot read {}: {}".format(path, image.geterror()))
    finally:
        image.close()
    pixels = np.asarray(pixels, dtype=np.float32).reshape(spec.height, spec.width, -1)
    values = None
    if index is not None:
        values = pixels[:, :, index] if index < pixels.shape[2] else None
    rgb = pixels[:, :, :3]
    if rgb.shape[2] == 1:
        rgb = np.repeat(rgb, 3, axis=2)
    elif rgb.shape[2] == 2:
        rgb = np.concatenate([rgb, rgb[:, :, :1]], axis=2)
    rgb = _into_display_window(rgb, spec)
    is_float = spec.format.basetype in (oiio.FLOAT, oiio.HALF, oiio.DOUBLE)
    return Frame(rgb, spec, is_float, alpha=None if values is None else _into_display_window(values[..., None], spec)[..., 0])


def to_display(rgb, linear):
    """Map plate values to the 0-1 display range the models were trained on.

    Linear plates go through a soft highlight knee and the sRGB curve so that
    shadows keep their texture and highlights above 1 don't clip flat.
    ``rgb`` is a torch tensor; the pipeline passes it already on the GPU,
    where this takes a millisecond instead of a quarter second at 4K.
    """
    if not linear:
        return rgb.clamp(0.0, 1.0)
    x = rgb.clamp(min=0.0)
    knee = 0.8
    x = torch.where(x > knee, knee + (1.0 - knee) * (1.0 - torch.exp(-(x - knee) / (1.0 - knee))), x)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.pow(1.0 / 2.4) - 0.055)


def write_channels(path, frame, channels, software, pixel_format=oiio.HALF):
    """Write ``channels`` (name -> [H, W] array) over the display window of ``frame``.

    Everything is stored as half unless ``pixel_format`` says otherwise. Half keeps 1/2048 of relative precision,
    far under any model's error, and the float mantissas of upsampled model
    output barely compress (SmartVectors at 4K: 132 MB a frame in float,
    ~16 MB in half).
    The file is written beside its destination and renamed so an interrupted
    run never leaves a partial frame that a later resume would trust.
    """
    names = list(channels)
    spec = oiio.ImageSpec(frame.width, frame.height, len(names), pixel_format)
    spec.x, spec.y = frame.origin
    spec.full_x, spec.full_y = frame.origin
    spec.channelnames = names
    spec.attribute("compression", "zips")
    spec.attribute("Software", software)
    # (H, W, C) contiguous: OIIO reads past a buffer of any other shape.
    pixels = np.ascontiguousarray(np.stack([channels[name] for name in names], axis=-1), dtype=np.float32)

    temporary = path + ".partial.exr"
    output = oiio.ImageOutput.create(temporary)
    if output is None or not output.open(temporary, spec):
        raise IOError("Cannot write {}: {}".format(temporary, oiio.geterror()))
    try:
        if not output.write_image(pixels):
            raise IOError("Cannot write {}: {}".format(temporary, output.geterror()))
    finally:
        output.close()
    try:
        os.replace(temporary, path)
    except OSError:
        os.remove(temporary)
        raise


def read_alpha(path, name=None):
    """The alpha of an image as [H, W] float32 over its display window.

    Takes the channel ``name`` (e.g. "subjects.id") when given, else the channel
    named A or alpha, or the only channel of a single-channel file.
    """
    image = oiio.ImageInput.open(path)
    if image is None:
        raise IOError("Cannot open {}: {}".format(path, oiio.geterror()))
    try:
        spec = image.spec()
        names = [channel.split(".")[-1].lower() for channel in spec.channelnames]
        if name is not None:
            if name not in spec.channelnames:
                raise IOError("{} has no {} channel".format(path, name))
            channel = list(spec.channelnames).index(name)
        elif "a" in names:
            channel = names.index("a")
        elif "alpha" in names:
            channel = names.index("alpha")
        elif spec.nchannels == 1:
            channel = 0
        else:
            raise IOError("{} has no alpha channel".format(path))
        pixels = image.read_image(0, 0, channel, channel + 1, oiio.FLOAT)
        if pixels is None:
            raise IOError("Cannot read {}: {}".format(path, image.geterror()))
    finally:
        image.close()
    pixels = np.asarray(pixels, dtype=np.float32).reshape(spec.height, spec.width, 1)
    return _into_display_window(pixels, spec)[..., 0]
