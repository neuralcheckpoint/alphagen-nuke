import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def plugin(monkeypatch):
    monkeypatch.setitem(sys.modules, "nuke", MagicMock())
    monkeypatch.setitem(sys.modules, "aitools_nuke", MagicMock())
    path = Path(__file__).resolve().parents[1] / "nuke_plugin" / "alphagen_nuke.py"
    spec = importlib.util.spec_from_file_location("alphagen_nuke_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_excluding_every_mapped_person_does_not_compute_automatically(plugin, monkeypatch):
    node = MagicMock()
    monkeypatch.setattr(plugin, "_range_and_subjects", lambda _node: (1001, 1009, [[1001, 10, 20, 1, 0]]))
    monkeypatch.setattr(plugin, "_has_people_map", lambda _node: True)
    monkeypatch.setattr(plugin, "_active_numbers", lambda _node: [1])
    monkeypatch.setattr(plugin, "_hit", lambda _node, _number: 2)
    engine = MagicMock(side_effect=AssertionError("an empty selection must not reach the engine"))
    monkeypatch.setattr(plugin, "_engine", engine)
    assert plugin.compute(node) is None
    engine.assert_not_called()
    plugin.nuke.message.assert_called_once()


def test_finishing_preview_rechecks_points_instead_of_marking_stale_selection_fresh(plugin, monkeypatch):
    node = MagicMock()
    knobs = {name: MagicMock() for name in ("show_preview", "preview_fresh")}
    node.__getitem__.side_effect = knobs.__getitem__
    refresh = MagicMock()
    monkeypatch.setattr(plugin, "refresh_selection", refresh)
    monkeypatch.setattr(plugin, "_has_people_map", lambda _node: True)
    job = SimpleNamespace(result={"written": 9, "seconds": 1.0})
    plugin._preview_finished(job, node)
    refresh.assert_called_once_with(node)
    node["preview_fresh"].setValue.assert_not_called()


def test_id_hit_is_sampled_without_filtering_neighbor_ids(plugin):
    node = MagicMock()
    node.knobs.return_value = {"analysis_pattern": object()}
    node["analysis_pattern"].value.return_value = "id.####.exr"
    node["subject_point_1"].getValue.return_value = (10.9, 20.1)
    node["subject_frame_1"].value.return_value = 1005
    people = plugin.nuke.toNode.return_value
    people.sample.return_value = 3
    plugin._update_hit(node, 1)
    people.sample.assert_called_once_with("subjects.id", 10.5, 20.5, frame=1005)
    node["subject_hit_1"].setValue.assert_called_once_with(3)


def test_only_exclude_points_on_people_mean_everyone_else(plugin):
    points = [[1001, 10, 20, 1, 0]]
    assert plugin._split_points(points, [2], everyone=[1, 2, 3]) == ([], [1, 3])
    # An include on another person, or a SAM include, keeps the explicit selection.
    both = [[1001, 10, 20, 1, 1], [1001, 30, 20, 2, 0]]
    assert plugin._split_points(both, [1, 2], everyone=[1, 2, 3]) == ([], [1])
    sam = [[1001, 50, 60, 1, 1], [1001, 30, 20, 2, 0]]
    assert plugin._split_points(sam, [0, 2], everyone=[1, 2, 3]) == ([sam[0]], [])
    # Excluding the only person there is still selects nobody.
    assert plugin._split_points(points, [1], everyone=[1]) == ([], [])


@pytest.mark.parametrize("profile,expected", [(None, (1088, True)), (0, (1088, True)),
                                              (1, (544, True)), (2, (768, False))])
def test_quality_presets_and_legacy_manual_settings(plugin, profile, expected):
    node = MagicMock()
    knobs = {name: MagicMock() for name in ("resolution", "low_vram")}
    knobs["resolution"].getValue.return_value = 2
    knobs["low_vram"].value.return_value = True
    if profile is not None:
        knobs["quality_profile"] = MagicMock()
        knobs["quality_profile"].getValue.return_value = profile
    node.knobs.return_value = knobs
    node.__getitem__.side_effect = knobs.__getitem__
    assert plugin.processing_settings(node) == expected


@pytest.mark.parametrize("profile,enabled", [(0, True), (1, False), (2, False)])
def test_preset_controls_preserve_manual_values(plugin, profile, enabled):
    node = MagicMock()
    node.knobs.return_value = {name: object() for name in ("alphagen_version", "resolution", "low_vram")}
    knob = MagicMock()
    knob.name.return_value = "quality_profile"
    knob.getValue.return_value = profile
    plugin.subject_knob_changed(node, knob)
    for name in ("resolution", "low_vram"):
        node[name].setEnabled.assert_called_with(enabled)
        node[name].setValue.assert_not_called()


def test_quality_upgrade_leaves_existing_settings_alone(plugin):
    node = MagicMock()
    node.knobs.return_value = {"alphagen_version": object(), "resolution": object(), "low_vram": object()}
    plugin.add_quality_knob(node)
    plugin.nuke.Enumeration_Knob.assert_called_once_with(
        "quality_profile", "quality profile", plugin.QUALITY_PROFILES,
    )
    node["resolution"].setValue.assert_not_called()
    node["low_vram"].setValue.assert_not_called()


@pytest.mark.parametrize("profile,resolution,small", [(1, "544", True), (2, "768", False)])
def test_compute_sends_effective_preset_to_engine(plugin, monkeypatch, profile, resolution, small):
    node = MagicMock()
    node.name.return_value = "AlphaGen1"
    node.node.return_value = None
    node.knobs.return_value = {"quality_profile": object()}
    node["quality_profile"].getValue.return_value = profile
    monkeypatch.setattr(plugin, "_range_and_subjects", lambda _node: (1001, 1072, []))
    monkeypatch.setattr(plugin, "_has_people_map", lambda _node: False)
    monkeypatch.setattr(plugin, "_engine", lambda _node: "/python")
    monkeypatch.setattr(plugin, "_source_pattern", lambda *_args: ("plate.####.exr", False))
    plugin.aitools_nuke.cache_pattern.return_value = "/tmp/alpha.####.exr"
    plugin.compute(node)
    command = plugin.aitools_nuke.EngineJob.call_args.args[1]
    assert command[command.index("--resolution") + 1] == resolution
    assert ("--low-vram" in command) is small
