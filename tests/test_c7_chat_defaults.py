"""BRIEF C7 part A: the chat's own defaults (reply limit 1024, 12 steps; context 8192 and thinking unchanged) and the
25-step cap. Only the chat changes: the profiles in config.py (exam, gate, LONG) keep their values.

The real chat, the real loop and a real disk with a scripted fake model (the World of ``tests/test_chat.py``)."""
from __future__ import annotations

import hashlib
import json

import pytest

from localbot import __main__ as cli
from localbot import chat
from localbot.config import get_profile
from tests.fakes import call, reply
from tests.test_chat import World


def _world(tmp_path, replies, keys=None):
    return World(tmp_path, keys or ["1", "clean up this folder", "/quit"], replies=replies)


def _header(w):
    (rec,) = w.records()
    assert rec[0]["kind"] == "header"
    return rec[0]


def _scenario():
    return [reply("looking", calls=[call("bash", command="ls")]), reply("Done here.")]


def test_c7_no_flags_sends_reply_limit_1024_and_records_12_steps_and_no_think(tmp_path):
    w = _world(tmp_path, _scenario())
    assert w.run() == 0, w.text
    assert len(w.transport.bodies) == 2
    for body in w.transport.bodies:
        assert body["options"] == {"num_ctx": 8192, "temperature": 0.2, "num_predict": 1024}
        assert "think" not in body
    h = _header(w)
    assert (h["reply_limit"], h["context"], h["steps"], h["think"]) == (1024, 8192, 12, "default")


def test_c7_no_flags_allows_twelve_steps_for_one_message(tmp_path):
    cut = [reply(thinking="t" * 100, done_reason="length", ev=1024) for _ in range(12)]
    w = _world(tmp_path, cut)
    assert w.run() == 0, w.text
    assert len(w.transport.bodies) == 12 and "max_steps" in w.text


def test_c7_flags_still_win_over_the_chat_defaults(tmp_path):
    w = _world(tmp_path, _scenario())
    assert chat.chat_main("main", None, w.deps, reply_limit=512, steps=8) == 0, w.text
    assert all(b["options"]["num_predict"] == 512 for b in w.transport.bodies)
    h = _header(w)
    assert (h["reply_limit"], h["steps"]) == (512, 8)


def test_c7_twenty_five_steps_is_allowed(tmp_path):
    w = _world(tmp_path, _scenario())
    assert chat.chat_main("main", None, w.deps, steps=25) == 0, w.text
    assert _header(w)["steps"] == 25


@pytest.mark.parametrize("n", [26, 40, 1000])
def test_c7_more_than_25_steps_is_refused_plainly_and_nothing_starts(tmp_path, n):
    w = _world(tmp_path, [reply("never asked")])
    assert chat.chat_main("main", None, w.deps, steps=n) == 2
    assert "steps" in w.text and "25" in w.text and str(n) in w.text and "Nothing was started" in w.text, w.text
    assert "max_turns" not in w.text and "num_predict" not in w.text
    assert not w.shells and not w.transport.bodies and "start" not in w.log
    assert not (w.home / ".localbot").exists()


def test_c7_the_command_line_refuses_26_steps_before_anything_starts(tmp_path, monkeypatch):
    # the real command line, the real chat_main: the plain message comes from the chat, before any start
    w = _world(tmp_path, [reply("never asked")])
    monkeypatch.setattr("localbot.chat.default_deps", lambda: w.deps)
    assert cli.main(["chat", "--steps", "26"]) == 2
    assert "25" in w.text and "Nothing was started" in w.text
    assert not w.shells and not w.transport.bodies and "start" not in w.log


def test_c7_help_text_shows_the_new_defaults(capsys):
    with pytest.raises(SystemExit):
        cli.main(["chat", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "(default 1024)" in out and "(default 12, at most 25)" in out and "(default 8192)" in out
    assert "default 512" not in out and "(default 8)" not in out


def test_c7_the_profiles_themselves_are_unchanged():
    # exam, gate and LONG runs use these profiles directly: the chat's defaults must not leak into them
    for name in ("core4", "core4_product", "parity_lab6"):
        p = get_profile(name)
        assert (p.provider.num_predict, p.loop.max_turns, p.provider.num_ctx) == (512, 8, 8192), name


def test_c7_the_new_no_flag_bodies_differ_from_the_c5_ones_only_in_the_reply_limit(tmp_path):
    # tests/test_c6_chat_settings.py's GOLDEN_BODIES was re-recorded for C7; put num_predict back to 512 and the bodies
    # hash to the C5 tip's golden again (ee6498fc..., recorded on 522201c), so nothing else in them changed
    w = _world(tmp_path, _scenario())
    assert w.run() == 0, w.text
    bodies = json.loads(json.dumps(w.transport.bodies))
    assert {b["options"]["num_predict"] for b in bodies} == {1024}
    for b in bodies:
        b["options"]["num_predict"] = 512
    s = json.dumps(bodies, sort_keys=True, ensure_ascii=False)
    assert hashlib.sha256(s.encode()).hexdigest() == "ee6498fc5f0e0113d0b464e49d2e8d0a0981c150a3fd11e1c13990a728351bfe"
