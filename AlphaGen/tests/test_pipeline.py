import json
import os
import re
import sys
from pathlib import Path

import av
import numpy as np
import OpenImageIO as oiio
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent / "aitools"))
sys.path.insert(0, str(PROJECT_ROOT))

from alphagen import pipeline  # noqa: E402
from aitools_engine.imageio import frame_path  # noqa: E402


def test_subject_seed_uses_whole_object_instead_of_body_part():
    masks = np.zeros((3, 8, 8), dtype=bool)
    masks[0, 2:4, 2:4] = True
    masks[1, 1:7, 1:7] = True
    masks[2, 2:5, 2:5] = True
    assert np.array_equal(pipeline.subject._object_mask(masks), masks[1])
    with pytest.raises(ValueError):
        pipeline.subject._object_mask(np.zeros_like(masks))



def _bundled_workflow():
    with open(pipeline.workflow_path(), "r", encoding="utf-8") as handle:
        return json.load(handle)


def test_bundled_workflow_is_closed_and_needs_only_alpha_gen_models():
    workflow = _bundled_workflow()
    for node in workflow.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in workflow
    names = sorted(os.path.basename(path) for path in pipeline._model_paths("root", workflow))
    assert names == sorted([
        "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        "ltx-2.5-audio-vae-bf16.safetensors",
        "ltx-2.5-video-vae-bf16.safetensors",
        pipeline.MODEL,
    ])
    assert not {"TextGenerateLTX2Prompt", "GemmaAPITextEncode", "LoadImage"} & {
        node["class_type"] for node in workflow.values()
    }
    assert all(
        node["inputs"]["text"] == "" for node in workflow.values() if node["class_type"] == "CLIPTextEncode"
    )




@pytest.mark.parametrize("total,free,forced,expected", [
    (8 * 1024 ** 3, 7 * 1024 ** 3, False, True),
    (8 * 1024 ** 3 + 1, 0, False, False),
    (24 * 1024 ** 3, 1 * 1024 ** 3, False, False),
    (24 * 1024 ** 3, 7 * 1024 ** 3, True, True),
])
def test_decode_budget_does_not_mistake_loaded_models_for_a_small_gpu(total, free, forced, expected):
    stats = {"devices": [{"vram_total": total, "vram_free": free}]}
    assert pipeline._small_decode(stats, forced) is expected


def test_decode_budget_rejects_missing_device_memory():
    with pytest.raises(RuntimeError):
        pipeline._small_decode({"devices": []})


def test_prepared_workflow_rejects_a_different_adapter():
    workflow = _bundled_workflow()
    workflow["5004:5606"]["inputs"]["strength_model"] = 0.6
    with pytest.raises(ValueError):
        pipeline.prepare_workflow(workflow, "clip.mp4", "clip", 544, 1)


@pytest.mark.parametrize("count", [122, 144, 227, 245, 331])
def test_long_shots_keep_context_and_cover_every_frame(count):
    first, last = 1001, 1000 + count
    chunks = pipeline.plan_chunks(first, last)
    assert chunks[0]["first"] == first
    assert chunks[-1]["last"] == last
    assert all(chunk["count"] == 121 and chunk["padded"] == 121 for chunk in chunks)
    for earlier, later in zip(chunks, chunks[1:]):
        assert earlier["first"] < later["first"]
        assert earlier["last"] - later["first"] + 1 >= 16
    assert all(
        any(chunk["first"] <= frame <= chunk["last"] for chunk in chunks)
        for frame in range(first, last + 1)
    )


def test_layout_preserves_aspect_and_model_limits():
    fitted = pipeline.layout(3840, 2160, 1088)

    assert (fitted["width"], fitted["height"]) == (1888, 1056)
    assert fitted["width"] <= 1920 and fitted["height"] <= 1088
    assert fitted["width"] % 32 == 0 and fitted["height"] % 32 == 0
    assert fitted["shorter_size"] == 1056


def test_restore_matte_removes_letterbox_padding():
    fitted = pipeline.layout(16, 8, 544)
    padded = np.zeros((fitted["height"], fitted["width"]), dtype=np.uint8)
    left, top = fitted["left"], fitted["top"]
    padded[top:top + fitted["content_height"], left:left + fitted["content_width"]] = 160

    restored = pipeline._restore_matte(padded, fitted, 16, 8)

    assert restored.shape == (8, 16)
    assert np.allclose(restored, 160 / 255.0)


def _write_exr(path, pixels):
    height, width, channels = pixels.shape
    spec = oiio.ImageSpec(width, height, channels, oiio.FLOAT)
    spec.channelnames = ["R", "G", "B"]
    output = oiio.ImageOutput.create(str(path))
    assert output is not None and output.open(str(path), spec)
    try:
        assert output.write_image(np.ascontiguousarray(pixels, dtype=np.float32))
    finally:
        output.close()


class _FakeFrame:
    width = 16
    height = 8
    is_float = False


def _mock_pipeline_io(monkeypatch, tmp_path, *, cache_valid=False, chunk_values=None):
    workflow = tmp_path / "workflow.json"
    workflow.write_text("{}", encoding="utf-8")
    config = {"root": str(tmp_path / "comfy"), "workflow": str(workflow), "host": "127.0.0.1", "port": 8188}
    values = chunk_values or {}
    writes = {}
    calls = []

    monkeypatch.setattr(pipeline.imageio, "read_frame", lambda _path: _FakeFrame())
    monkeypatch.setattr(pipeline.imageio, "sequence_stamp", lambda _pattern, _frames: [])
    monkeypatch.setattr(pipeline, "read_manifest", lambda _path: {"cached": True})
    monkeypatch.setattr(pipeline, "_cache_matches", lambda *_args: cache_valid)
    monkeypatch.setattr(pipeline, "_check_server", lambda _config: {
        "devices": [{"vram_total": 24 * 1024 ** 3}],
    })
    monkeypatch.setattr(pipeline, "_write_input_video", lambda *_args: None)

    def run_workflow(_config, _input, prefix, _resolution, _seed, _status, _low_vram):
        calls.append(prefix)
        return prefix

    def read_matte(path, expected_frames):
        index = int(re.search(r"clip_(\d+)$", path).group(1))
        value = values.get(index, 128)
        return iter([np.full((32, 32), value, dtype=np.uint8) for _ in range(expected_frames)])

    def write_channels(path, _frame, channels, _software):
        frame = int(Path(path).stem.rsplit(".", 1)[1])
        writes[frame] = np.array(channels["alphagen.alpha"], copy=True)
        Path(path).touch()

    monkeypatch.setattr(pipeline, "_run_workflow", run_workflow)
    monkeypatch.setattr(pipeline, "_read_matte", read_matte)
    monkeypatch.setattr(pipeline.imageio, "write_channels", write_channels)
    return config, calls, writes




@pytest.mark.parametrize("cached_low_vram,requested_low_vram,expect_hit", [
    (False, False, True), (True, True, True),
    (False, True, False), (True, False, False),
])
def test_complete_cache_requires_same_decode_override_without_comfyui(
    tmp_path, cached_low_vram, requested_low_vram, expect_hit
):
    input_pattern = str(tmp_path / "plate.####.exr")
    output_pattern = str(tmp_path / "alpha.####.exr")
    pixels = np.full((8, 16, 3), 0.18, dtype=np.float32)
    _write_exr(frame_path(input_pattern, 1001), pixels)
    _write_exr(frame_path(output_pattern, 1001), pixels)
    fitted = pipeline.layout(16, 8, 544)
    root = tmp_path / "missing-comfy"
    workflow = str(root / "workflow.json")
    params = {
        "version": pipeline.CACHE_VERSION,
        "model": pipeline.MODEL,
        "model_stamps": [[str(root / "models" / "loras" / pipeline.MODEL), [1, 1]]],
        "workflow": workflow,
        "workflow_stamp": [1, 1],
        "input": input_pattern,
        "input_stamp": pipeline.imageio.sequence_stamp(input_pattern, [1001]),
        "first": 1001,
        "last": 1001,
        "fps": 24.0,
        "resolution": 544,
        "processed_size": [fitted["width"], fitted["height"]],
        "seed": 1234,
        "layer": "alphagen.alpha",
        "subjects": None,
        "low_vram": cached_low_vram,
        "chunk_frames": pipeline.CHUNK_FRAMES,
        "overlap_frames": pipeline.OVERLAP_FRAMES,
        "subject_mask": None,
        "sam_dependencies": None,
    }
    pipeline._write_manifest(str(tmp_path / "alphagen.json"), params)
    config = {
        "root": str(root),
        "workflow": workflow,
        "host": "127.0.0.1",
        "port": 8188,
    }

    if expect_hit:
        assert pipeline.run(
            input_pattern, 1001, 1001, output_pattern,
            config=config, low_vram=requested_low_vram,
        ) == 0
    else:
        with pytest.raises(RuntimeError):
            pipeline.run(
                input_pattern, 1001, 1001, output_pattern,
                config=config, low_vram=requested_low_vram,
            )
        assert not (tmp_path / "alphagen.json").exists()


def test_partial_multichunk_cache_recomputes_the_whole_shot(monkeypatch, tmp_path):
    config, calls, _writes = _mock_pipeline_io(monkeypatch, tmp_path, cache_valid=True)
    output_dir = tmp_path / "cache"
    output_dir.mkdir()
    input_pattern = str(tmp_path / "plate.####.exr")
    output_pattern = str(output_dir / "alpha.####.exr")
    for frame in range(1001, 1123):
        if frame != 1050:
            Path(frame_path(output_pattern, frame)).touch()

    written = pipeline.run(input_pattern, 1001, 1122, output_pattern, config=config)

    assert written == 122
    assert len(calls) == len(pipeline.plan_chunks(1001, 1122)) == 2
    assert all(Path(frame_path(output_pattern, frame)).is_file() for frame in range(1001, 1123))


def test_run_blends_estimates_at_multiple_seams(monkeypatch, tmp_path):
    config, calls, writes = _mock_pipeline_io(
        monkeypatch, tmp_path, cache_valid=False, chunk_values={1: 51, 2: 128, 3: 204}
    )
    input_pattern = str(tmp_path / "plate.####.exr")
    output_pattern = str(tmp_path / "cache" / "alpha.####.exr")

    written = pipeline.run(input_pattern, 1001, 1331, output_pattern, config=config)

    assert written == 331
    assert len(calls) == 3
    assert len(writes) == 331
    assert np.allclose(writes[1106], 51 / 255.0, atol=1e-3)
    assert np.allclose(writes[1121], 128 / 255.0, atol=1e-3)
    assert np.allclose(writes[1113], (51 * (1 - 7 / 15) + 128 * (7 / 15)) / 255.0, atol=2e-3)
    assert np.allclose(writes[1211], 128 / 255.0, atol=1e-3)
    assert np.allclose(writes[1226], 204 / 255.0, atol=1e-3)


def test_failed_forced_run_leaves_no_valid_manifest(monkeypatch, tmp_path):
    config, _calls, _writes = _mock_pipeline_io(monkeypatch, tmp_path, cache_valid=True)
    output_dir = tmp_path / "cache"
    output_dir.mkdir()
    output_pattern = str(output_dir / "alpha.####.exr")
    manifest = output_dir / "alphagen.json"
    manifest.write_text("old cache", encoding="utf-8")

    def fail_workflow(*_args):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(pipeline, "_run_workflow", fail_workflow)

    with pytest.raises(RuntimeError):
        pipeline.run(
            str(tmp_path / "plate.####.exr"), 1001, 1001, output_pattern,
            config=config, force=True,
        )

    assert not manifest.exists()


@pytest.mark.parametrize("point", [
    [1000, 2, 2],
    [1001.5, 2, 2],
    [1001, float("nan"), 2],
    [1001, 2, float("inf")],
    [1001, -0.01, 2],
    [1001, 16, 2],
    [1001, 2, 8],
    [1001, 2],
    [1001, 2, 2, 0, 1],
    [1001, 2, 2, 1.5, 1],
    [1001, 2, 2, 1, 2],
    [1001, 2, 2, 1, 0],
])
def test_invalid_subject_points_fail_before_comfy_or_sam(monkeypatch, tmp_path, point):
    monkeypatch.setattr(pipeline.imageio, "read_frame", lambda _path: _FakeFrame())
    monkeypatch.setattr(
        pipeline, "_check_server", lambda _config: pytest.fail("ComfyUI must not be contacted")
    )

    with pytest.raises(ValueError):
        pipeline.run(
            str(tmp_path / "plate.####.exr"),
            1001,
            1002,
            str(tmp_path / "alpha.####.exr"),
            config={"root": str(tmp_path)},
            subjects=[point],
        )

def test_clicks_on_one_subject_seed_sam_together():
    subjects = pipeline.subject.normalize_subjects(
        [[1005, 1, 1, 2, 1], [1001, 2, 2], [1005, 3, 3, 2, 0], [1001, 4, 4, 1, True]], 1001, 1009, 16, 8,
    )
    assert subjects[1] == [1001, 2.0, 2.0, 2, 1]
    assert pipeline.subject.prompt_groups(subjects) == [
        ((1, 1001), [(4.0, 4.0, 1)]),
        ((2, 1001), [(2.0, 2.0, 1)]),
        ((2, 1005), [(1.0, 1.0, 1), (3.0, 3.0, 0)]),
    ]


def test_click_grows_to_the_clicked_persons_joints():
    mask = np.zeros((100, 100), bool)
    mask[10:30, 10:30] = True  # SAM's answer to the click: the face
    blond = {"box": [0, 0, 60, 100], "joints": [[20, 20], [20, 60], [30, 90]]}
    other = {"box": [50, 0, 100, 100], "joints": [[80, 20], [21, 61]]}  # second joint overlaps the blond's
    clicks = [(20.0, 20.0, 1)]
    assert pipeline.person.expand(clicks, [other, blond], mask) == clicks + [
        (20, 20, 1), (20, 60, 1), (30, 90, 1), (80, 20, 0),
    ]
    # Clicked someone the detector missed: no joints in the click's mask, clicks unchanged.
    assert pipeline.person.expand(clicks, [other], mask) == clicks
    assert pipeline.person.expand(clicks, [], mask) == clicks


def test_input_video_is_padded_and_has_silent_audio(tmp_path):
    pattern = str(tmp_path / "plate.####.exr")
    _write_exr(frame_path(pattern, 1001), np.full((8, 16, 3), 0.18, dtype=np.float32))
    _write_exr(frame_path(pattern, 1002), np.full((8, 16, 3), 0.32, dtype=np.float32))
    chunk = pipeline.plan_chunks(1001, 1002)[0]
    fitted = pipeline.layout(16, 8, 544)
    fitted["source_width"], fitted["source_height"] = 16, 8
    output = tmp_path / "clip.mp4"

    pipeline._write_input_video(pattern, chunk, fitted, str(output), 24.0)
    with av.open(str(output)) as video:
        assert (video.streams.video[0].width, video.streams.video[0].height) == (32, 32)
        assert len(video.streams.audio) == 1
        assert len(list(video.decode(video=0))) == 9


def test_person_map_keeps_only_the_picked_people(monkeypatch, tmp_path):
    config, _calls, writes = _mock_pipeline_io(monkeypatch, tmp_path)
    ids = np.zeros((8, 16), dtype=np.float32)
    ids[:, :8], ids[:, 8:] = 1, 2
    monkeypatch.setattr(pipeline, "read_manifest",
                        lambda path: {"first": 1001, "last": 1001} if path.endswith("analysis.json") else {})
    monkeypatch.setattr(pipeline.imageio, "read_alpha",
                        lambda _path, name=None: ids if name == pipeline.subject.ID_LAYER else None)
    input_pattern = str(tmp_path / "plate.####.exr")
    output_pattern = str(tmp_path / "cache" / "alpha.####.exr")
    id_map = {"pattern": str(tmp_path / "people" / "id.####.exr"), "ids": [2]}

    assert pipeline.run(input_pattern, 1001, 1001, output_pattern, config=config, id_map=id_map) == 1
    assert writes[1001][:, 0].max() < 0.01  # person 1 is cut; only the 4 px dilation leaks in
    assert np.allclose(writes[1001][:, 13:], 128 / 255.0, atol=1e-3)

    monkeypatch.setattr(pipeline, "read_manifest", lambda _path: {"first": 1001, "last": 1000})
    with pytest.raises(ValueError, match="does not cover"):
        pipeline.run(input_pattern, 1001, 1001, output_pattern, config=config, id_map=id_map)


def test_one_detected_person_is_prompted_with_its_joints_against_the_others():
    first = {"box": [0, 0, 10, 100], "joints": [[5, 10], [5, 50]]}
    second = {"box": [50, 0, 60, 100], "joints": [[55, 10]]}
    assert pipeline.person.prompts(first, [first, second]) == [(5, 10, 1), (5, 50, 1), (55, 10, 0)]
