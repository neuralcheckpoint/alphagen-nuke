"""SAM 2.1 subject prompts and isolated RotoTrack subprocess support."""

import argparse
import json
import math
import os
import subprocess
import sys
import traceback


ALPHAGEN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_ROOT = os.path.dirname(ALPHAGEN_ROOT)
ROTOTRACK_ROOT = os.path.join(TOOLS_ROOT, "RotoTrack")
AITTOOLS_ROOT = os.path.join(TOOLS_ROOT, "aitools")
SUBJECT_LAYER = "subject.alpha"
ID_LAYER = "subjects.id"  # analysis map: 0 = nobody, n = detected person n
SOFTWARE = "Alpha Gen subject selection (SAM 2.1)"


def normalize_subjects(subjects, first, last, width, height):
    """Validate SAM prompts before starting either model.

    A prompt is [frame, x, y] (its own subject, include) or
    [frame, x, y, subject, include]. Points sharing a subject number are
    clicks on the same object; include=0 marks an area to leave out.
    Returns [frame, x, y, subject, include] lists.
    """
    if subjects is None:
        return None
    if isinstance(subjects, (str, bytes)):
        raise ValueError("Subjects must be a sequence of [frame, x, y] points")

    normalized = []
    try:
        points = list(subjects)
    except TypeError:
        raise ValueError("Subjects must be a sequence of [frame, x, y] points")

    for index, point in enumerate(points, 1):
        if not isinstance(point, (list, tuple)) or len(point) not in (3, 5):
            raise ValueError("Subject {} must contain frame, x, y and optionally subject, include".format(index))
        values = []
        for name, value in zip(("frame", "x", "y", "subject", "include"), point):
            if isinstance(value, (str, bytes)) or (isinstance(value, bool) and name != "include"):
                raise ValueError("Subject {} {} must be a finite number".format(index, name))
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                raise ValueError("Subject {} {} must be a finite number".format(index, name))
            if not math.isfinite(number):
                raise ValueError("Subject {} {} must be a finite number".format(index, name))
            values.append(number)

        frame_value, x, y = values[:3]
        object_value, include = values[3:] or (index, 1.0)
        frame = int(frame_value)
        if frame != frame_value:
            raise ValueError("Subject {} frame must be an integer".format(index))
        if object_value != int(object_value) or object_value < 1:
            raise ValueError("Subject {} number must be a positive integer".format(index))
        if include not in (0.0, 1.0):
            raise ValueError("Subject {} include must be 0 or 1".format(index))
        if frame < first or frame > last:
            raise ValueError("Subject {} frame {} is outside {}-{}".format(index, frame, first, last))
        if not 0.0 <= x < width or not 0.0 <= y < height:
            raise ValueError(
                "Subject {} point ({}, {}) is outside the {}x{} plate".format(index, x, y, width, height)
            )
        normalized.append([frame, x, y, int(object_value), int(include)])

    for (object_id, frame), clicks in prompt_groups(normalized):
        if not any(label for _x, _y, label in clicks):
            raise ValueError(
                "Subject {} has only exclude points on frame {}; add an include point there".format(object_id, frame)
            )
    return normalized or None


def prompt_groups(subjects):
    """Group normalized prompts into SAM seeds: ((subject, frame), [(x, y, label), ...]), by frame."""
    groups = {}
    for frame, x, y, object_id, include in subjects:
        groups.setdefault((object_id, frame), []).append((x, y, include))
    return sorted(groups.items(), key=lambda item: (item[0][1], item[0][0]))


def runtime():
    """Return the dedicated RotoTrack interpreter and an installed SAM size."""
    interpreter = os.path.join(ROTOTRACK_ROOT, ".venv",
                               "Scripts" if os.name == "nt" else "bin",
                               "python.exe" if os.name == "nt" else "python")
    if not os.path.isfile(interpreter):
        raise RuntimeError("RotoTrack Python environment not found: {}".format(interpreter))

    models = os.path.join(ROTOTRACK_ROOT, "models")
    for size, checkpoint in (
        ("base", "sam2.1_hiera_base_plus.pt"),
        ("large", "sam2.1_hiera_large.pt"),
    ):
        if os.path.isfile(os.path.join(models, checkpoint)):
            return interpreter, size
    raise RuntimeError("No SAM 2.1 checkpoint is installed in {}".format(models))


def cache_fingerprint():
    """Record the SAM checkpoint, worker, and vendored implementation in Alpha Gen's cache."""
    interpreter, size = runtime()
    version_script = """
import importlib.metadata as metadata
import json
import sys

names = ('torch', 'torchvision', 'numpy', 'hydra-core', 'iopath', 'OpenImageIO', 'Pillow', 'tqdm', 'scipy')
packages = {}
for name in names:
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None
print(json.dumps({'python': sys.version.split()[0], 'packages': packages}, sort_keys=True))
"""
    versions = subprocess.run(
        [interpreter, "-c", version_script],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    checkpoint = "sam2.1_hiera_base_plus.pt" if size == "base" else "sam2.1_hiera_large.pt"
    paths = [
        interpreter,
        os.path.join(ROTOTRACK_ROOT, "models", checkpoint),
        os.path.join(ROTOTRACK_ROOT, "rototrack", "model.py"),
        os.path.join(ROTOTRACK_ROOT, "requirements.txt"),
        os.path.join(ROTOTRACK_ROOT, "setup.sh"),
        __file__,
        os.path.join(os.path.dirname(__file__), "person.py"),
    ]
    vendor = os.path.join(ROTOTRACK_ROOT, "vendor", "sam2", "sam2")
    for directory, _subdirs, files in os.walk(vendor):
        for name in files:
            if name.endswith((".py", ".yaml", ".yml")):
                paths.append(os.path.join(directory, name))

    stamps = []
    for path in sorted(set(paths)):
        try:
            info = os.stat(path)
            stamp = [info.st_size, info.st_mtime_ns]
        except OSError:
            stamp = None
        stamps.append([os.path.relpath(path, ROTOTRACK_ROOT), stamp])
    return {"size": size, "files": stamps, "environment": json.loads(versions.stdout)}


def run_tracker(input_pattern, first, last, output_pattern, subjects, *, on_poll=None, detector=None, analyze_key=None):
    """Track all selected people in a child process that owns the SAM model.

    detector: (python, models folder) from person.paths, to grow a click on a
    person into the whole body; None seeds from the clicks alone.
    analyze_key: instead of tracking the clicks, track every person detected
    on this frame into an ID map (ID_LAYER).
    """
    interpreter, size = runtime()
    environment = os.environ.copy()
    search_path = [ALPHAGEN_ROOT, ROTOTRACK_ROOT, AITTOOLS_ROOT]
    if environment.get("PYTHONPATH"):
        search_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(search_path)
    command = [
        interpreter,
        "-m",
        "alphagen.subject",
        "--worker",
        "--input",
        input_pattern,
        "--first",
        str(first),
        "--last",
        str(last),
        "--output",
        output_pattern,
        "--size",
        size,
        "--subjects",
        json.dumps(subjects, separators=(",", ":")),
    ]
    if detector:
        command += ["--detector", json.dumps(list(detector))]
    if analyze_key is not None:
        command += ["--analyze-key", str(analyze_key)]
    process = subprocess.Popen(
        command,
        cwd=ROTOTRACK_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        while process.poll() is None:
            if on_poll is not None:
                on_poll()
            try:
                output, _ = process.communicate(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                continue
        else:
            output, _ = process.communicate()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process.communicate()
        raise

    if process.returncode:
        details = (output or "").strip()[-4000:]
        raise RuntimeError("SAM 2.1 subject tracking failed{}".format(
            ":\n" + details if details else " (exit {})".format(process.returncode)
        ))

def _object_mask(masks):
    import numpy as np
    areas = np.count_nonzero(masks, axis=(1, 2))
    # shortcut: the largest click candidate can merge touching subjects; upgrade to corrective prompts then.
    if not areas.max():
        raise ValueError("SAM 2.1 found no subject at the selected point")
    return masks[int(areas.argmax())]



def _seed_mask(predictor, clicks):
    import numpy as np

    single = len(clicks) == 1
    masks, _scores, _logits = predictor.predict(
        point_coords=np.asarray([[x, y] for x, y, _label in clicks], dtype=np.float32),
        point_labels=np.asarray([label for _x, _y, label in clicks], dtype=np.int32),
        multimask_output=single,
    )
    # One click: whole object, not SAM's highest-IoU body-part candidate.
    # Several clicks/excludes already disambiguate, so SAM's single answer is used.
    return (_object_mask(masks) if single else masks[0]) > 0


def _worker(input_pattern, first, last, output_pattern, subjects, size, detector=None):
    import numpy as np
    import torch

    from aitools_engine import imageio
    from rototrack.model import Frames, Tracker

    from alphagen import person

    frame_numbers = list(range(first, last + 1))
    reference = imageio.read_frame(imageio.frame_path(input_pattern, first))
    tracker = Tracker(size)
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    frames = Frames(input_pattern, frame_numbers, tracker.predictor.image_size, tracker.device)
    seen = set()
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if tracker.device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )
    try:
        with torch.inference_mode(), autocast:
            state = tracker.start(frames)
            seed_predictor = SAM2ImagePredictor(tracker.predictor)
            image_frame = None
            for (object_id, key), clicks in prompt_groups(subjects):
                if key != image_frame:
                    plate = reference if key == first else imageio.read_frame(imageio.frame_path(input_pattern, key))
                    rgb = imageio.to_display(torch.from_numpy(plate.rgb), plate.is_float).clamp(0, 1)
                    rgb = (rgb.numpy() * 255).astype(np.uint8)
                    seed_predictor.set_image(rgb)
                    image_frame, found = key, None
                mask = _seed_mask(seed_predictor, clicks)
                if detector:
                    if found is None:
                        try:
                            found = person.people(rgb, *detector)
                        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                            print("Person detection unavailable, using the clicks alone: {}".format(exc), flush=True)
                            found = detector = []
                    body = person.expand(clicks, found, mask)
                    if len(body) > len(clicks):
                        # The click's own mask keeps what joints miss (hair, props held up to the face).
                        mask = mask | _seed_mask(seed_predictor, body)
                tracker.predictor.add_new_mask(state, frame_idx=key - first, obj_id=object_id, mask=mask)
            del seed_predictor

            for reverse in (False, True):
                for index, _object_ids, logits in tracker.predictor.propagate_in_video(
                    state, reverse=reverse
                ):
                    index = int(index)
                    if index in seen:
                        continue
                    union = torch.sigmoid(logits.float()).amax(dim=0)[0].cpu().numpy()
                    frame_number = frame_numbers[index]
                    imageio.write_channels(
                        imageio.frame_path(output_pattern, frame_number),
                        reference,
                        {SUBJECT_LAYER: union},
                        SOFTWARE,
                    )
                    seen.add(index)
    finally:
        frames.close()

    missing = sorted(set(range(len(frame_numbers))) - seen)
    if missing:
        raise RuntimeError("SAM 2.1 did not return {} frames (first missing index {})".format(
            len(missing), missing[0]
        ))


def _analyze(input_pattern, first, last, output_pattern, key, size, detector):
    """Track everyone the detector finds on frame ``key`` into an ID map, one number per person."""
    import numpy as np
    import torch

    from aitools_engine import imageio
    from rototrack.model import Frames, Tracker

    from alphagen import person

    if not detector:
        raise RuntimeError("Person detection needs ComfyUI's YOLOv10 and ViTPose models")
    frame_numbers = list(range(first, last + 1))
    plate = imageio.read_frame(imageio.frame_path(input_pattern, key))
    rgb = imageio.to_display(torch.from_numpy(plate.rgb), plate.is_float).clamp(0, 1)
    rgb = (rgb.numpy() * 255).astype(np.uint8)
    # shortcut: people are detected on one frame only; someone who enters later needs a point (SAM click).
    found = [found for found in person.people(rgb, *detector) if found["joints"]]
    if not found:
        raise RuntimeError("No person found on frame {}; try another frame or use points".format(key))

    tracker = Tracker(size)
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    frames = Frames(input_pattern, frame_numbers, tracker.predictor.image_size, tracker.device)
    seen = set()
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if tracker.device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )
    try:
        with torch.inference_mode(), autocast:
            state = tracker.start(frames)
            seed_predictor = SAM2ImagePredictor(tracker.predictor)
            seed_predictor.set_image(rgb)
            for object_id, target in enumerate(found, 1):
                mask = _seed_mask(seed_predictor, person.prompts(target, found))
                tracker.predictor.add_new_mask(state, frame_idx=key - first, obj_id=object_id, mask=mask)
            del seed_predictor

            for reverse in (False, True):
                for index, object_ids, logits in tracker.predictor.propagate_in_video(state, reverse=reverse):
                    index = int(index)
                    if index in seen:
                        continue
                    probabilities = torch.sigmoid(logits.float())[:, 0].cpu().numpy()
                    ids = np.asarray(list(object_ids), dtype=np.float32)[probabilities.argmax(axis=0)]
                    ids[probabilities.max(axis=0) < 0.5] = 0.0
                    imageio.write_channels(
                        imageio.frame_path(output_pattern, frame_numbers[index]), plate, {ID_LAYER: ids}, SOFTWARE,
                    )
                    seen.add(index)
    finally:
        frames.close()

    if len(seen) != len(frame_numbers):
        raise RuntimeError("SAM 2.1 did not return every frame of the analysis")
    with open(os.path.join(os.path.dirname(os.path.abspath(output_pattern)), "analysis.json"), "w") as handle:
        json.dump({"first": first, "last": last, "key": key, "people": len(found)}, handle)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="alphagen.subject")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--first", type=int)
    parser.add_argument("--last", type=int)
    parser.add_argument("--output")
    parser.add_argument("--size", choices=("base", "large"), default="base")
    parser.add_argument("--subjects", default="[]")
    parser.add_argument("--detector", default="null")
    parser.add_argument("--analyze-key", type=int)
    args = parser.parse_args(argv)
    if not args.worker:
        parser.error("this module is an internal SAM worker")
    try:
        if args.analyze_key is not None:
            _analyze(args.input, args.first, args.last, args.output, args.analyze_key, args.size,
                     json.loads(args.detector))
            return 0
        subjects = json.loads(args.subjects)
        _worker(args.input, args.first, args.last, args.output, subjects, args.size,
                json.loads(args.detector))
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
