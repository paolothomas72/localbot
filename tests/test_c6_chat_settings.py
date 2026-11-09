"""BRIEF C6 parts A and B: the chat's settings (reply limit, context, steps, thinking), the gpu_fit line in the
record, and the cut note that says it is not from the user.

The real chat, the real loop and a real disk, with a scripted fake model and a scripted container (the World of
``tests/test_chat.py``). Nothing here talks to Ollama or a container."""
from __future__ import annotations

import hashlib
import json

import pytest

from localbot import __main__ as cli
from localbot import chat
from localbot.config import get_profile
from localbot.loop import CUT_NUDGE, run_task
from tests.fakes import FakeShell, ScriptedTransport, call, provider, reply
from tests.test_chat import SECRET, World
from tests import test_c5_cut_events as c5

# The request bodies of the scenario below, recorded by running it on 522201c (the C5 tip: what a72b2db sends).
# C7: re-recorded on purpose: the chat's own default reply limit is now 1024. The only difference from the C5 bodies is
# num_predict (tests/test_c7_chat_defaults.py puts 512 back and gets the C5 hash ee6498fc...).
GOLDEN_BODIES = ("2", "6057f5c27fe8cd07d0ac7806898f0f2f5c56a224e2fbdc502171b2cf82c93afe")
EXAM_CUT_SHA = "ca78169dfb3a7059a02ad05c6626c84f61209e2f0063f7d2959e94ef2d6c6d16"   # exam rules, 8 cut replies, on 522201c

NOTE_THINKING = ("[harness note, not from the user] Your reply hit the {n}-token limit before any tool call or answer "
                 "(your thinking used it up). Think less. Make one tool call now, or give the user a short answer.")
NOTE_TEXT = ("[harness note, not from the user] Your reply hit the output limit ({n} tokens) and was cut off, so "
             "nothing in it was run. Send a shorter reply. Write a big file in parts: write the first part, then add "
             "the rest in later steps.")


def _world(tmp_path, replies, keys=None, **kw):
    return World(tmp_path, keys or ["1", "clean up this folder", "/quit"], replies=replies, **kw)


def _scenario_replies():
    return [reply("looking", calls=[call("bash", command="ls")]), reply("Done here.")]


def _header(w):
    (rec,) = w.records()
    assert rec[0]["kind"] == "header"
    return rec[0]


# --- 1. each setting reaches the request and the record header ---------------------------------------------------


def test_c6_the_settings_reach_every_request_body_and_the_record_header(tmp_path):
    w = _world(tmp_path, _scenario_replies())
    assert chat.chat_main("main", None, w.deps, reply_limit=1024, context=16384, steps=5, think=False) == 0, w.text
    for body in w.transport.bodies:
        assert body["options"]["num_predict"] == 1024 and body["options"]["num_ctx"] == 16384
        assert body["think"] is False
    h = _header(w)
    assert (h["reply_limit"], h["context"], h["steps"], h["think"]) == (1024, 16384, 5, "off")


def test_c6_think_on_sends_true(tmp_path):
    w = _world(tmp_path, _scenario_replies())
    assert chat.chat_main("main", None, w.deps, think=True) == 0, w.text
    assert all(b["think"] is True for b in w.transport.bodies)
    assert _header(w)["think"] == "on"


def test_c6_the_steps_setting_limits_the_steps_of_one_message(tmp_path):
    cut = [reply(thinking="t" * 100, done_reason="length", ev=512) for _ in range(3)]
    w = _world(tmp_path, cut)
    assert chat.chat_main("main", None, w.deps, steps=3) == 0, w.text
    assert len(w.transport.bodies) == 3 and "max_steps" in w.text


# --- 2. no flags: the same request bodies as before ----------------------------------------------------------------


def test_c6_without_flags_the_request_bodies_are_those_of_the_c5_tip(tmp_path):
    w = _world(tmp_path, _scenario_replies())
    assert w.run() == 0, w.text
    assert "think" not in w.transport.bodies[0]
    assert w.transport.bodies[0]["options"] == {"num_ctx": 8192, "temperature": 0.2, "num_predict": 1024}  # C7
    s = json.dumps(w.transport.bodies, sort_keys=True, ensure_ascii=False)
    assert (str(len(w.transport.bodies)), hashlib.sha256(s.encode()).hexdigest()) == GOLDEN_BODIES
    h = _header(w)
    assert (h["reply_limit"], h["context"], h["steps"], h["think"]) == (1024, 8192, 12, "default")  # C7


# --- 3. bad values stop before anything starts ---------------------------------------------------------------------


@pytest.mark.parametrize("kw,words", [
    ({"reply_limit": 8192}, ("reply limit", "context")),
    ({"reply_limit": 9000}, ("reply limit", "context")),
    ({"reply_limit": 1024, "context": 1024}, ("reply limit", "context")),
    ({"reply_limit": 0}, ("reply limit",)),
    ({"context": 0}, ("context",)),
    ({"steps": 0}, ("steps",)),
    ({"steps": -2}, ("steps",)),
])
def test_c6_a_bad_value_stops_before_the_container_starts_with_a_plain_message(tmp_path, kw, words):
    w = _world(tmp_path, [reply("never asked")])
    assert chat.chat_main("main", None, w.deps, **kw) == 2
    assert all(x in w.text for x in words), w.text
    assert "num_predict" not in w.text and "num_ctx" not in w.text and "max_turns" not in w.text
    assert not w.shells and not w.transport.bodies and "start" not in w.log
    assert not (w.home / ".localbot").exists()


def test_c6_the_command_line_passes_the_flags_and_leaves_a_plain_chat_alone(monkeypatch):
    got = []
    monkeypatch.setattr("localbot.chat.chat_main", lambda *a, **k: got.append((a, k)) or 0)
    assert cli.main(["chat"]) == 0
    assert cli.main(["chat", "--bot", "b", "--model", "m:1", "--reply-limit", "1024", "--context", "16384",
                     "--steps", "6", "--think", "off"]) == 0
    assert cli.main(["chat", "--think", "on"]) == 0
    assert got[0] == (("main", None), {})                      # exactly the call of before
    assert got[1] == (("b", "m:1"), {"reply_limit": 1024, "context": 16384, "steps": 6, "think": False})
    assert got[2] == (("main", None), {"think": True})


@pytest.mark.parametrize("argv", [["--reply-limit", "lots"], ["--context", "1.5"], ["--steps", "x"],
                                  ["--think", "maybe"], ["--think"]])
def test_c6_text_where_a_number_or_on_off_belongs_is_refused_by_the_command_line(monkeypatch, argv, capsys):
    started = []
    monkeypatch.setattr("localbot.chat.chat_main", lambda *a, **k: started.append(1) or 0)
    with pytest.raises(SystemExit) as e:
        cli.main(["chat", *argv])
    assert e.value.code == 2 and not started
    assert "usage" in capsys.readouterr().err.lower()


# --- 4. thinking the model cannot switch: said plainly, no retry ----------------------------------------------------


def test_c6_a_think_error_from_ollama_is_said_plainly_and_the_chat_stops_without_a_retry(tmp_path):
    err = RuntimeError('HTTP Error 400: Bad Request: {"error":"\\"qwen3:8b\\" does not support thinking"}')
    w = _world(tmp_path, [err, reply("must never be asked for")], keys=["1", "hello", "again?", "/quit"])
    assert chat.chat_main("main", None, w.deps, think=False) == 1
    assert len(w.transport.bodies) == 1                          # no second try, with or without the field
    assert "does not support thinking" in w.text and "--think" in w.text
    assert w.shells and w.log.count("close") == w.log.count("start") == 1
    (rec,) = w.records()
    assert rec[-1]["kind"] == "end" and rec[-1]["reason"] == "think-error"
    assert [p for p in w.keys.prompts if p == "you> "] == ["you> "]  # it did not ask for another message


def test_c6_an_error_without_the_think_setting_is_left_to_the_loop_as_before(tmp_path):
    err = RuntimeError('HTTP Error 400: Bad Request: {"error":"nope"}')
    w = _world(tmp_path, [err], keys=["1", "hello", "/quit"])
    assert w.run() == 0
    assert "think-error" not in w.record_text() and "provider_error" in w.text


# --- 5. the gpu_fit line --------------------------------------------------------------------------------------------


def _gpu_lines(w):
    (rec,) = w.records()
    return [r for r in rec if r["kind"] == "gpu_fit"], rec


def test_c6_one_gpu_fit_line_after_the_first_reply(tmp_path):
    w = _world(tmp_path, _scenario_replies())
    w.routed.ps_models = [{"name": "qwen3:8b", "size": 6_000_000_000, "size_vram": 6_000_000_000}]
    assert w.run() == 0, w.text
    lines, rec = _gpu_lines(w)
    assert len(lines) == 1
    g = lines[0]
    assert (g["model_bytes"], g["on_card_bytes"], g["fits"], g["doubtful"]) == (6_000_000_000, 6_000_000_000, "yes", False)
    kinds = [r["kind"] for r in rec]
    assert kinds.index("header") < kinds.index("gpu_fit") < kinds.index("summary")


def test_c6_part_of_the_model_off_the_card_is_a_no(tmp_path):
    w = _world(tmp_path, _scenario_replies())
    w.routed.ps_models = [{"name": "qwen3:8b", "size": 9_000_000_000, "size_vram": 6_000_000_000}]
    assert w.run() == 0
    (g,), _ = _gpu_lines(w)
    assert (g["model_bytes"], g["on_card_bytes"], g["fits"]) == (9_000_000_000, 6_000_000_000, "no")


def test_c6_a_model_ollama_does_not_list_is_unknown(tmp_path):
    w = _world(tmp_path, _scenario_replies())
    assert w.run() == 0
    (g,), _ = _gpu_lines(w)
    assert g["fits"] == "unknown" and g["model_bytes"] is None


def test_c6_a_reading_far_below_the_size_on_disk_is_marked_doubtful(tmp_path):
    w = _world(tmp_path, _scenario_replies())
    w.routed.ps_models = [{"name": "qwen3:8b", "size": 1_140_000_000, "size_vram": 1_140_000_000}]  # disk size is 5.2 GB
    assert w.run() == 0
    (g,), _ = _gpu_lines(w)
    assert g["fits"] == "yes" and g["doubtful"] is True


def test_c6_stats_shows_the_settings_and_the_card(tmp_path):
    w = _world(tmp_path, _scenario_replies(), keys=["1", "hello", "/stats", "/quit"])
    w.routed.ps_models = [{"name": "qwen3:8b", "size": 6_000_000_000, "size_vram": 6_000_000_000}]
    assert chat.chat_main("main", None, w.deps, reply_limit=1024, steps=8, think=False) == 0, w.text  # C7: steps pinned
    assert "reply limit: 1024" in w.text and "context: 8192" in w.text and "steps: 8" in w.text
    assert "thinking: off" in w.text and "fits on the card: yes" in w.text


# --- 6. the cut note (product rules) -----------------------------------------------------------------------------


def _second_body_last_message(w):
    return w.transport.bodies[1]["messages"][-1]


def test_c6_a_cut_reply_with_only_thinking_gets_the_note_that_says_it_is_not_from_the_user(tmp_path):
    w = _world(tmp_path, [reply(thinking="t" * 900, done_reason="length", ev=512), reply("ok")])
    assert chat.chat_main("main", None, w.deps, reply_limit=512) == 0, w.text  # C7: the C6 limit pinned
    m = _second_body_last_message(w)
    assert m["role"] == "user" and m["content"] == NOTE_THINKING.format(n=512)


def test_c6_a_cut_reply_with_visible_text_keeps_the_shorter_reply_advice_with_the_same_start(tmp_path):
    w = _world(tmp_path, [reply("a long start of an answer", thinking="t", done_reason="length", ev=512), reply("ok")])
    assert chat.chat_main("main", None, w.deps, reply_limit=512) == 0, w.text  # C7: the C6 limit pinned
    assert _second_body_last_message(w)["content"] == NOTE_TEXT.format(n=512)


def test_c6_the_note_names_the_limit_in_use(tmp_path):
    w = _world(tmp_path, [reply(thinking="t", done_reason="length", ev=1024), reply("ok")])
    assert chat.chat_main("main", None, w.deps, reply_limit=1024) == 0, w.text
    assert "the 1024-token limit" in _second_body_last_message(w)["content"]


def test_c6_under_the_exam_rules_the_cut_note_is_the_old_text_byte_for_byte(disk):
    assert CUT_NUDGE == ("harness: your reply hit the output limit ({n} tokens) and was cut off, so nothing in it was "
                         "run. Send a shorter reply. Write a big file in parts: write the first part, then add the "
                         "rest in later steps.")
    p = get_profile("core4")
    ws = disk.ws(dict(c5.SEED))
    res = run_task(p, provider(ScriptedTransport(c5.SCENARIOS["cut"]()), p.provider), "fake:1", "clean up the folder",
                   ws, shell=FakeShell({}))
    s = json.dumps(res.messages, sort_keys=True, ensure_ascii=False, default=str)
    assert hashlib.sha256(s.encode()).hexdigest() == EXAM_CUT_SHA
    assert [m["content"] for m in res.messages if m["role"] == "user"][-1] == CUT_NUDGE.format(n=512)


# --- 7. missing from the C5 review: a canary in a cut reply -----------------------------------------------------


def test_c6_a_cut_reply_holding_a_secret_shows_it_in_no_event_screen_line_or_record(tmp_path):
    cut = reply(f"the key is API_KEY={SECRET} and so on", thinking="t", done_reason="length", ev=512)
    w = _world(tmp_path, [cut, reply("ok")], seed={".env": f"API_KEY={SECRET}\n", "notes.txt": "N\n"})
    assert w.run() == 0, w.text
    assert SECRET not in json.dumps(w.events()), "an event holds the secret"
    assert SECRET not in w.text, "the screen shows the secret"
    assert SECRET not in w.record_text(), "the record holds the secret"
    assert any(e["event"] == "cut" for e in w.events())


def test_c6_a_cut_reply_s_own_words_are_in_no_event_screen_line_or_record(tmp_path):
    # reviewer (C6): the test above uses the .env key, which the masking removes anyway, so it cannot see model text
    # put into the cut event. This marker is in no secrets file: only "the cut event carries no model text" keeps it out.
    marker = "PLAIN-MARKER-7731"
    cut = reply(f"I will now say {marker} and go on", thinking="t", done_reason="length", ev=512)
    w = _world(tmp_path, [cut, reply("ok")], seed={"notes.txt": "N\n"})
    assert w.run() == 0, w.text
    assert any(e["event"] == "cut" for e in w.events())
    assert marker not in json.dumps(w.events()), "an event holds the model's own words"
    assert marker not in w.text, "the screen shows the model's own words"
    assert marker not in w.record_text(), "the record holds the model's own words"
