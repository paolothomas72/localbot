"""BRIEF C5 F1: under the product rules the model never sees a secret-named file.

The real chat, the real loop and a real disk, with a scripted fake model and a scripted container (the World of
``tests/test_chat.py``). The canary is a made-up key in ``.env``; it must be in no message sent to the model, no
event, no screen line and no record. Exam rules stay as they were."""
from __future__ import annotations

import json
import os

import pytest

from localbot.config import get_profile
from localbot.loop import run_task
from tests.fakes import ScriptedTransport, call, provider, reply
from tests.test_chat import SECRET, World

NOTICE = ("looks like a secrets file. Local Bot does not show its contents to the model. "
          "If you need something from it, ask the user.")
SEED = {".env": f"API_KEY={SECRET}\n", "notes.txt": "N\n", "sub/x.txt": "x\n", "config/.env": f"K={SECRET}\n"}


def _chat(tmp_path, step, seed=None, new_work=None):
    w = World(tmp_path, ["1", "what is in the file?", "/quit"], seed=dict(SEED if seed is None else seed),
              replies=[reply(calls=[step]), reply("I cannot see it.")])
    if new_work is not None:
        base = w.deps.new_work
        w.deps.new_work = lambda path: new_work(base(path))
    assert w.run() == 0, w.text
    return w


def _nowhere(w, canary=SECRET):
    """The canary is in none of: every body sent to the model, every event, the screen, the record."""
    assert canary not in json.dumps(w.transport.bodies), "the model was sent the secret"
    assert canary not in json.dumps(w.events()), "an event holds the secret"
    assert canary not in w.text, "the screen shows the secret"
    assert canary not in w.record_text(), "the record holds the secret"


def _tool_message(w):
    """The text the model got back for its one tool call (the last request body's last tool message)."""
    msgs = [m for m in w.transport.bodies[-1]["messages"] if m.get("role") == "tool"]
    assert len(msgs) == 1, msgs
    return msgs[0]["content"]


# --- 1 + 2. read, every spelling ------------------------------------------------------------------------------

PATHS = [".env", "./.env", "sub/../.env", ".ENV", ".env ", "config/.env"]


@pytest.mark.parametrize("path", PATHS, ids=["plain", "dot-slash", "dotdot", "upper", "trailing-space", "subfolder"])
def test_c5_read_of_a_secret_named_file_gives_the_model_a_notice_never_the_contents(tmp_path, path):
    w = _chat(tmp_path, call("read", path=path))
    _nowhere(w)
    got = _tool_message(w)
    assert got.startswith("ERROR: harness:") and NOTICE in got, got
    refusals = w.events("refusal")
    assert len(refusals) == 1 and refusals[0]["tool"] == "read"
    assert "  ! refused: read" in w.text
    assert (w.work / ".env").read_text(encoding="utf-8") == f"API_KEY={SECRET}\n"


def test_c5_the_notice_names_the_file_the_model_asked_for(tmp_path):
    w = _chat(tmp_path, call("read", path="config/.env"))
    assert f"ERROR: harness: config/.env {NOTICE}" in _tool_message(w)


# --- 3. edit ------------------------------------------------------------------------------------------------------


def test_c5_edit_of_a_secret_named_file_gets_the_notice_not_updated_and_the_file_is_unchanged(tmp_path):
    before = f"API_KEY={SECRET}\n".encode("utf-8")
    w = _chat(tmp_path, call("edit", path=".env", old_string="API_KEY=", new_string="API_KEY=changed-"))
    got = _tool_message(w)
    assert got.startswith("ERROR: harness:") and NOTICE in got and "Updated" not in got, got
    assert (w.work / ".env").read_bytes() == before
    _nowhere(w)


# --- 4. write (already refused; this guards it) --------------------------------------------------------------------


def test_c5_write_to_a_secret_named_file_is_still_refused(tmp_path):
    w = _chat(tmp_path, call("write", path=".env", content="API_KEY=other\n"))
    got = _tool_message(w)
    assert got.startswith("ERROR:") and "Wrote" not in got, got
    assert (w.work / ".env").read_text(encoding="utf-8") == f"API_KEY={SECRET}\n"
    _nowhere(w)


# --- 5. a harmless-named link to .env ------------------------------------------------------------------------------


def test_c5_a_link_with_a_harmless_name_to_env_is_refused_and_leaks_nothing(tmp_path):
    made = []

    def link(work):
        try:
            os.symlink(work / ".env", work / "notes2.txt")
        except (OSError, NotImplementedError, AttributeError) as exc:
            pytest.skip(f"this OS cannot make symbolic links here: {type(exc).__name__}")
        made.append(1)
        return work

    w = _chat(tmp_path, call("read", path="notes2.txt"), new_work=link)
    assert made
    got = _tool_message(w)
    assert got.startswith("ERROR:") and "1\t" not in got, got
    _nowhere(w)


# --- 6. no regression ----------------------------------------------------------------------------------------------


def test_c5_read_of_a_normal_file_is_exactly_as_before(tmp_path):
    w = _chat(tmp_path, call("read", path="notes.txt"))
    assert _tool_message(w).rstrip("\n").endswith("1\tN") and "ERROR" not in _tool_message(w)
    assert w.events("refusal") == []


def test_c5_read_of_a_template_name_like_env_example_is_not_treated_as_a_secret(tmp_path):
    w = _chat(tmp_path, call("read", path=".env.example"), seed={".env.example": "API_KEY=changeme\n"})
    assert "changeme" in _tool_message(w) and w.events("refusal") == []


# --- 7. exam rules unchanged ---------------------------------------------------------------------------------------


def test_c5_under_the_exam_rules_read_of_env_gives_what_it_gave_before(disk):
    p = get_profile("core4")
    ws = disk.ws({".env": f"API_KEY={SECRET}\n", "notes.txt": "N\n"})
    res = run_task(p, provider(ScriptedTransport([reply(calls=[call("read", path=".env")]), reply("done")]),
                               p.provider), "fake:1", "what is in .env?", ws)
    assert res.tool_results == [f"1\tAPI_KEY={SECRET}"]   # recorded by running the same call on c17748c
