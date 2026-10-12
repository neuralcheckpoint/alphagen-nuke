"""SAM 2.1's video predictor, fed from EXR sequences.

SAM 2 reads a folder of JPEGs into memory at 1024x1024; here its frame
loader is swapped for one that reads each plate frame when the tracker asks
for it, through the same display transform as the other tools, so a shot
of any length or bit depth goes in without converting it first.
"""

import os
import queue
import sys
import threading

import torch
import torch.nn.functional as F

from aitools_engine import imageio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(ROOT, "vendor", "sam2")
MODELS = os.path.join(ROOT, "models")

# SAM 2.1 checkpoints and configs, Apache-2.0.
SIZES = {
    "large": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
    "base": ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class Frames(object):
    """The plate as SAM 2 indexes it: frame i -> normalised [3, S, S] tensor.

    Frames are read on demand; the tracker visits each once per direction.
    A worker thread reads one frame ahead, so decoding the next frame (disk
    plus resize, ~25 ms of a 1920x1012 plate) overlaps the model on the
    current one. The caller always receives the frame it asked for, and a
    read error is raised on the caller's thread, where it can be reported.
    """

    _STOP = object()

    def __init__(self, pattern, frames, size, device, linear=None):
        self.pattern = pattern
        self.frames = list(frames)
        self.size = size
        self.device = device
        first = imageio.read_frame(imageio.frame_path(pattern, self.frames[0]))
        self.video_width, self.video_height = first.width, first.height
        self.linear = first.is_float if linear is None else linear
        self.mean = torch.tensor(MEAN, device=device).view(3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(3, 1, 1)
        self._last = None
        # Depth one on each side of the handoff: a request and a result are
        # enough for one lookahead, and the worker can never run away.
        self._asked = None
        self._requests = queue.Queue(maxsize=1)
        self._results = queue.Queue(maxsize=1)
        self._reader = threading.Thread(target=self._read, name="rototrack frame reader", daemon=True)
        self._reader.start()

    def _read(self):
        """Worker loop: read the requested frame and hand it back tagged."""
        while True:
            index = self._requests.get()
            if index is self._STOP:
                return
            try:
                result = self._load(index)
            except Exception as exc:
                result = exc
            self._results.put((index, result))

    def _request(self, index):
        """Ask for ``index``, dropping a lookahead the caller has moved past.

        The tracker jumps (to a key frame, or when it turns around) and the
        queued frame is then stale; replacing it keeps the wait short.
        """
        if self._asked == index:
            return
        self._asked = index
        while True:
            try:
                self._requests.put_nowait(index)
                return
            except queue.Full:
                try:
                    self._requests.get_nowait()
                except queue.Empty:
                    pass

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        if self._last is not None and self._last[0] == index:
            return self._last[1]
        self._request(index)
        while True:
            got, result = self._results.get()
            if got == index:
                break
        if self._asked == index:
            self._asked = None
        # A failed read is raised, never cached: returning it on a second
        # access would hand the caller an exception where a tensor belongs.
        if isinstance(result, Exception):
            raise result
        ahead = index - 1 if self._last is not None and index < self._last[0] else index + 1
        self._last = (index, result)
        if 0 <= ahead < len(self.frames):
            self._request(ahead)
        return result

    def _load(self, index):
        data = imageio.read_frame(imageio.frame_path(self.pattern, self.frames[index]))
        if (data.width, data.height) != (self.video_width, self.video_height):
            raise ValueError("Frame {} has a different resolution".format(self.frames[index]))
        rgb = imageio.to_display(torch.from_numpy(data.rgb).to(self.device), self.linear)
        # SAM 2 squeezes every frame to a square, as its PIL loader does
        # (bicubic, antialiased).
        rgb = F.interpolate(rgb.permute(2, 0, 1)[None], size=(self.size, self.size), mode="bicubic",
                            antialias=True, align_corners=False)[0].clamp(0.0, 1.0)
        return (rgb - self.mean) / self.std

    def close(self):
        """Stop the reader; a run keeps only its own Frames alive."""
        reader = self._reader
        self._reader = None
        if reader is None:
            return
        self._request(self._STOP)
        # Empty the handoff so a lookahead nobody will read can't keep the
        # worker blocked on a full queue instead of seeing the stop.
        while True:
            try:
                self._results.get_nowait()
            except queue.Empty:
                break
        reader.join(timeout=5.0)


def _import_vendor():
    if VENDOR not in sys.path:
        sys.path.insert(0, VENDOR)
    from sam2 import sam2_video_predictor
    from sam2.build_sam import build_sam2_video_predictor

    return sam2_video_predictor, build_sam2_video_predictor


class Tracker(object):
    """SAM 2.1 video predictor with a plate loaded frame by frame."""

    def __init__(self, size="large", device=None):
        if size not in SIZES:
            raise ValueError("Unknown model size {!r}; use one of {}".format(size, sorted(SIZES)))
        checkpoint, config = SIZES[size]
        path = os.path.join(MODELS, checkpoint)
        if not os.path.isfile(path):
            raise IOError("Missing weights {}; run setup.sh".format(path))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._module, build = _import_vendor()
        self.predictor = build(config, path, device=self.device)

    def start(self, frames):
        """A SAM 2 inference state over ``frames`` (a Frames)."""
        module = self._module
        original = module.load_video_frames

        def load(**_kwargs):
            return frames, frames.video_height, frames.video_width

        module.load_video_frames = load
        try:
            # Tracking state on the CPU: a long shot doesn't fill the GPU, and
            # SAM 2 only looks back 7 frames anyway.
            return self.predictor.init_state(video_path=None, offload_state_to_cpu=True)
        finally:
            module.load_video_frames = original

    def free_memory(self):
        if self.device.type != "cuda":
            return None
        return torch.cuda.mem_get_info(self.device)[0]
