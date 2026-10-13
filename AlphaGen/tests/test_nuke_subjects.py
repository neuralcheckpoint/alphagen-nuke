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
