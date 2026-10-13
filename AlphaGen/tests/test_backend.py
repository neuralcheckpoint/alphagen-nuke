import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent / "aitools"))
sys.path.insert(0, str(PROJECT_ROOT))

from alphagen import backend  # noqa: E402


@pytest.fixture
def config():
    return {"root": "/comfy", "host": "127.0.0.1", "port": 8191}


def test_start_reuses_online_backend(monkeypatch, config):
    monkeypatch.setattr(backend, "online", lambda _config: True)
    run = MagicMock()
    monkeypatch.setattr(backend.subprocess, "run", run)
    assert backend.control("start", config=config) == 0
    run.assert_not_called()


def test_stop_refuses_busy_queue(monkeypatch, config):
    monkeypatch.setattr(backend, "online", lambda _config: True)
    monkeypatch.setattr(backend.shutil, "which", lambda _name: "/bin/comfy")
    monkeypatch.setattr(backend.pipeline, "_request", lambda *_args, **_kwargs: {"queue_running": [1], "queue_pending": []})
    run = MagicMock()
    monkeypatch.setattr(backend.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="busy"):
        backend.control("stop", config=config)
    run.assert_not_called()


@pytest.mark.parametrize("action,states", [("start", [False, True]), ("stop", [True, False])])
def test_control_checks_requested_state(monkeypatch, config, action, states):
    monkeypatch.setattr(backend, "online", MagicMock(side_effect=states))
    monkeypatch.setattr(backend.shutil, "which", lambda _name: "/bin/comfy")
    monkeypatch.setattr(backend.pipeline, "_require_comfy_idle", lambda _config: None)
    run = MagicMock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(backend.subprocess, "run", run)
    events = []
    assert backend.control(action, config=config, progress=events.append) == 0
    command = run.call_args.args[0]
    assert command[:3] == ["/bin/comfy", "--workspace", "/comfy"]
    assert "8191" in command
    assert events[-1]["event"] == "backend"


def test_status_rejects_unrelated_http_service(monkeypatch, config):
    monkeypatch.setattr(backend.pipeline, "_request", lambda *_args, **_kwargs: {"ok": True})
    with pytest.raises(RuntimeError, match="unexpected HTTP"):
        backend.online(config)


def test_status_rejects_http_error_instead_of_calling_it_offline(monkeypatch, config):
    def request(*_args, **_kwargs):
        raise HTTPError("http://localhost", 404, "missing", None, None)
    monkeypatch.setattr(backend.pipeline, "_request", request)
    with pytest.raises(RuntimeError, match="unexpected HTTP"):
        backend.online(config)
