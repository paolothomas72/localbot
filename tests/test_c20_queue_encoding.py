"""BRIEF C20: the queue reads `ollama` output as UTF-8 (bytes invalid in cp1252 and UTF-8 must not kill anything).

No ollama, no network, no GPU: subprocess.run is wrapped so the queue's own keyword arguments are kept
but the command becomes a python child that prints raw bad bytes."""
from __future__ import annotations

import subprocess
import sys
import threading

import pytest

import scripts.model_queue as mq

REAL_RUN = subprocess.run


@pytest.fixture
def rig(monkeypatch):
    seen: list[dict] = []
    thread_errors: list = []
    state = {"code": 0}

    def fake_run(cmd, **kwargs):
        seen.append(dict(kwargs))
        script = ("import sys; "
                  "sys.stdout.buffer.write(b'pulling \\x8f layer\\n'); sys.stdout.buffer.flush(); "
                  "sys.stderr.buffer.write(b'pulling \\x8f layer\\n'); sys.stderr.buffer.flush(); "
                  f"sys.exit({state['code']})")
        return REAL_RUN([sys.executable, "-c", script], **kwargs)

    monkeypatch.setattr(mq.subprocess, "run", fake_run)
    monkeypatch.setattr(threading, "excepthook", lambda args: thread_errors.append(args))

    def no_network(*a, **k):
        raise OSError("no network in tests")

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    return seen, thread_errors, state


def test_pull_ok_survives_bad_bytes(rig):
    seen, errs, state = rig
    state["code"] = 0
    assert mq._pull_default("m") == (True, "ok")
    assert errs == []


def test_pull_failure_keeps_message(rig):
    seen, errs, state = rig
    state["code"] = 1
    ok, msg = mq._pull_default("m")
    assert ok is False
    assert "pulling" in msg
    assert errs == []


def test_show_modelfile_is_text(rig):
    seen, errs, state = rig
    state["code"] = 0
    data = mq._show_default("m")
    assert isinstance(data["modelfile"], str)
    assert "pulling" in data["modelfile"]
    assert errs == []


def test_delete_no_exception(rig):
    seen, errs, state = rig
    state["code"] = 0
    mq._delete_default("m")
    assert errs == []


def test_three_calls_pass_utf8_replace(rig):
    seen, errs, state = rig
    mq._pull_default("m")
    mq._show_default("m")
    mq._delete_default("m")
    assert len(seen) == 3
    for kw in seen:
        assert kw.get("encoding") == "utf-8"
        assert kw.get("errors") == "replace"
