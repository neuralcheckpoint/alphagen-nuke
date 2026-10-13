"""Body keypoints for one-click SAM seeds.

SAM alone answers a click on a face with the face. The clicked person's body joints
(YOLOv10 person box, then ViTPose) become extra include clicks, and other people's
joints become exclude clicks, so one click selects the whole person. Both are
ComfyUI's ONNX models, run in ComfyUI's Python because that has onnxruntime.
"""

import json
import math
import os
import subprocess
import sys
import tempfile

SIZE = 640
PERSON = 0  # COCO class id
MIN_SCORE = 0.3
MIN_JOINT = 0.5
BODY = 23  # COCO-WholeBody: 17 body joints + 6 feet; face and hands only add clutter
POSE_SIZE = (192, 256)  # width, height
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def paths(comfy_root):
    """(python, models folder) when ComfyUI has both models, else None (SAM alone)."""
    python = os.path.join(comfy_root, ".venv", "bin", "python")
    models = os.path.join(comfy_root, "models", "detection")
    found = all(os.path.isfile(os.path.join(models, name)) for name in ("yolov10m.onnx", "vitpose-l-wholebody.onnx"))
    return (python, models) if found and os.path.isfile(python) else None


def people(rgb, python, models):
    """People in an HxWx3 uint8 image: {"box": [x1, y1, x2, y2], "joints": [[x, y], ...]}."""
    import numpy as np

    with tempfile.TemporaryDirectory() as directory:
        image = os.path.join(directory, "frame.npy")
        np.save(image, rgb)
        environment = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        result = subprocess.run(
            [python, "-m", "alphagen.person", models, image],
            env=environment, capture_output=True, text=True, timeout=120,
        )
    if result.returncode:
        raise RuntimeError("Person detection failed: " + (result.stderr or result.stdout).strip()[-2000:])
    return json.loads(result.stdout)


def expand(clicks, found, mask):
    """Add the clicked person's joints as includes and everyone else's as excludes.

    clicks: [(x, y, label)]; mask: SAM's HxW answer to those clicks alone. The
    clicked person must have joints inside that mask and near an include click,
    so overlap with someone the detector missed does not select a different body.
    """
    def hits(person):
        return sum(bool(mask[int(y), int(x)]) for x, y in person["joints"])

    includes = [(x, y) for x, y, label in clicks if label]
    candidates = [person for person in found if person["joints"] and hits(person)]
    if not includes or not candidates:
        return list(clicks)

    def distance(person):
        return min(math.hypot(x - tx, y - ty) for x, y in includes for tx, ty in person["joints"])

    target = min(candidates, key=distance)
    # A prop or a missed person can overlap another person's joints without selecting that person.
    if distance(target) > 0.25 * (target["box"][3] - target["box"][1]):
        return list(clicks)
    return list(clicks) + prompts(target, found)


def prompts(target, found):
    """SAM clicks for one detected person: its joints included, everyone else's excluded."""
    added = [(x, y, 1) for x, y in target["joints"]]
    near = 0.05 * (target["box"][3] - target["box"][1])  # a joint this close may be the target's own
    for person in found:
        if person is not target:
            added += [
                (x, y, 0) for x, y in person["joints"]
                if all(math.hypot(x - tx, y - ty) > near for tx, ty, _label in added if _label)
            ]
    return added


def _boxes(session, rgb):
    import numpy as np
    from PIL import Image

    height, width = rgb.shape[:2]
    scale = SIZE / max(height, width)
    resized = Image.fromarray(rgb).resize((round(width * scale), round(height * scale)), Image.BILINEAR)
    pad_x, pad_y = (SIZE - resized.width) // 2, (SIZE - resized.height) // 2
    canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)  # Ultralytics letterbox grey
    canvas[pad_y:pad_y + resized.height, pad_x:pad_x + resized.width] = np.asarray(resized)
    tensor = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

    rows = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]
    kept = []
    for x1, y1, x2, y2, score, label in rows.tolist():  # sorted by score
        box = [max(0.0, (x1 - pad_x) / scale), max(0.0, (y1 - pad_y) / scale),
               min(width - 1.0, (x2 - pad_x) / scale), min(height - 1.0, (y2 - pad_y) / scale)]
        # YOLOv10 still returns weaker duplicates of one person; they would exclude its own joints.
        if int(label) == PERSON and score >= MIN_SCORE and all(_overlap(box, other) < 0.5 for other in kept):
            kept.append(box)
    return kept


def _overlap(a, b):
    """Intersection over the smaller box, so a box inside another counts as a duplicate."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    return width * height / min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))


def _joints(session, rgb, box):
    import numpy as np
    from PIL import Image

    # ViTPose crop: the box grown 1.25x to the model's 3:4 aspect, centred.
    centre_x, centre_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    crop_height = max(box[3] - box[1], (box[2] - box[0]) * POSE_SIZE[1] / POSE_SIZE[0]) * 1.25
    crop_width = crop_height * POSE_SIZE[0] / POSE_SIZE[1]
    left, top = centre_x - crop_width / 2, centre_y - crop_height / 2
    crop = Image.fromarray(rgb).transform(
        POSE_SIZE, Image.EXTENT, (left, top, left + crop_width, top + crop_height), Image.BILINEAR
    )
    tensor = ((np.asarray(crop, np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    heatmaps = session.run(None, {session.get_inputs()[0].name: tensor})[0][0][:BODY]
    rows, columns = heatmaps.shape[1:]
    joints = []
    for heatmap in heatmaps:
        row, column = np.unravel_index(int(heatmap.argmax()), heatmap.shape)
        if heatmap[row, column] >= MIN_JOINT:
            joints.append([left + (column + 0.5) * crop_width / columns, top + (row + 0.5) * crop_height / rows])
    height, width = rgb.shape[:2]
    return [[x, y] for x, y in joints if 0 <= x < width and 0 <= y < height]


def _detect(models, rgb):
    import onnxruntime

    def session(name):
        return onnxruntime.InferenceSession(os.path.join(models, name), providers=["CPUExecutionProvider"])

    found = _boxes(session("yolov10m.onnx"), rgb)
    if not found:
        return []
    pose = session("vitpose-l-wholebody.onnx")
    return [{"box": box, "joints": _joints(pose, rgb, box)} for box in found]


if __name__ == "__main__":
    import numpy as np
    print(json.dumps(_detect(sys.argv[1], np.load(sys.argv[2]))))
