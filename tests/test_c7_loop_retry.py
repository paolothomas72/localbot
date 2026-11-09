"""BRIEF C7 part B: a second chance for a reply that loops (repeats the same lines until it is stopped).

Only when the caller asks for it (``run_task(..., repeat_retries=N)``; the chat passes 2). Exam, gate and LONG runs
pass nothing: a looping reply ends the task as ``repetition`` at once, exactly as on 80f2049 (golden below).

The real chat, the real loop and a real disk, with a scripted fake model (the World of ``tests/test_chat.py``)."""
from __future__ import annotations

import hashlib
import json
import re

import pytest

from localbot import chat
from localbot import loop as loopmod
from localbot.config import get_profile
from localbot.loop import run_task
from tests.fakes import FakeShell, ScriptedTransport, call, provider, reply
from tests.test_chat import World

MARKER = "PLAIN-MARKER-4417"
LOOP = "\n".join([f"I will tidy the folder now {MARKER}."] * 12)
GOLDEN_LOOP = "\n".join(["I will tidy the folder now."] * 12)

NOTE_1 = ("[harness note, not from the user] Your last reply repeated the same lines over and over and was stopped. "
          "Give a short answer, or make one tool call.")
NOTE_2 = ("[harness note, not from the user] Your last reply repeated the same lines over and over and was stopped. "
          "Answer in one or two sentences, or make exactly one tool call.")

# Recorded on 80f2049 (before C7) with outbox/c7_golden_80f2049.py: core4 and core4_product give the same result,
# with and without on_event (the scenario: one read, then a reply of 12 equal lines).
GOLDEN_80F2049 = json.loads(r'''{"messages_sha256": "2891a16218a6919188614cf504bd0bddbea38114c6e79cb55b95a8312f38df03", "n_messages": 5, "reason": "repetition", "segments": [{"end_reason": "repetition", "steps": 2, "turn": 1}], "summary": {"arg_errors": [], "arg_extra": [], "call_source": ["native", "none"], "compactions": [], "container_scans": [], "delete_attempt": false, "error": "", "eval_count": 60, "false_success": [], "git_changed": false, "no_call_reason": null, "nudges": {"cut": 0, "parse": 0, "stuck": 0}, "reason": "repetition", "text_call_attempts": [], "tool_errors": [], "tools": ["read"], "veto_texts": [], "vetoes": 0}}''')


def _core(res):
    blob = json.dumps(res.messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
    s = res.summary()
    for k in ("elapsed_s", "provider_s", "eval_s", "tok_s", "provider", "event_errors"):
        s.pop(k, None)
    return {"reason": res.reason, "n_messages": len(res.messages), "messages_sha256": hashlib.sha256(blob).hexdigest(),
            "segments": [{k: v for k, v in g.items() if k != "wall_s"} for g in res.segments], "summary": s}


def _direct(disk, profile_name, replies, **kw):
    p = get_profile(profile_name)
    ws = disk.ws({"notes.txt": "N\n"})
    tr = ScriptedTransport(replies)
    res = run_task(p, provider(tr, p.provider), "fake:1", "clean up the folder", ws, shell=FakeShell({}), **kw)
    return res, tr


# --- 1. exam rules (and any caller that does not ask): exactly as on 80f2049 ---------------------------------------


@pytest.mark.parametrize("profile_name", ["core4", "core4_product"])
@pytest.mark.parametrize("watch", [False, True])
@pytest.mark.parametrize("given", [False, True])
def test_c7_without_retries_a_loop_ends_repetition_at_once_exactly_as_on_80f2049(disk, profile_name, watch, given):
    events = []
    kw = {"on_event": events.append} if watch else {}
    if given:
        kw["repeat_retries"] = 0
    res, tr = _direct(disk, profile_name, [reply(calls=[call("read", path="notes.txt")]), reply(GOLDEN_LOOP),
                                           reply("never asked")], **kw)
    assert _core(res) == GOLDEN_80F2049
    assert len(tr.bodies) == 2
    assert not any("harness note" in str(m.get("content")) for m in res.messages)
    if watch:
        assert [e["event"] for e in events] == ["tool", "message_end"]


@pytest.mark.parametrize("bad", [-1, True, "2", 1.5, None])
def test_c7_repeat_retries_must_be_a_whole_number_of_zero_or_more(disk, bad):
    with pytest.raises(ValueError):
        _direct(disk, "core4_product", [reply("x")], repeat_retries=bad)


# --- 2. the loop itself, with retries ------------------------------------------------------------------------------


def test_c7_with_retries_the_loop_events_carry_numbers_only_and_the_loop_text_is_not_kept(disk):
    events = []
    res, tr = _direct(disk, "core4_product", [reply(LOOP), reply(LOOP), reply("Short answer.")],
                      repeat_retries=2, on_event=events.append)
    assert res.reason == "done"
    loops = [e for e in events if e["event"] == "loop"]
    assert [(e["count"], e["retries"], e["limit"], e["retry"], e["next_limit"]) for e in loops] == [
        (1, 2, 512, True, None), (2, 2, 512, True, 256)]
    assert all(set(e) == {"event", "message", "step", "count", "retries", "limit", "retry", "next_limit"} for e in loops)
    assert MARKER not in json.dumps(res.messages) and MARKER not in json.dumps(events)
    assert [m["content"] for m in res.messages if m["role"] == "user"][1:] == [NOTE_1, NOTE_2]
    placeholders = [m for m in res.messages if m["role"] == "assistant" and m["content"] == loopmod.LOOP_PLACEHOLDER]
    assert len(placeholders) == 2 and all(set(m) == {"role", "content"} for m in placeholders)
    assert MARKER not in loopmod.LOOP_PLACEHOLDER


# --- 3. the chat: loop 1, loop 2, loop 3 ---------------------------------------------------------------------------


def _chat(tmp_path, replies, keys=None, **kw):
    w = World(tmp_path, keys or ["1", "clean up this folder", "/quit"], replies=replies)
    assert chat.chat_main("main", None, w.deps, **kw) == 0, w.text
    return w


def _msgs(body):
    return body["messages"]


def test_c7_chat_loop_1_gets_note_1_and_the_looping_reply_is_not_in_the_next_request(tmp_path):
    w = _chat(tmp_path, [reply(LOOP), reply("Short answer.")])
    b = w.transport.bodies
    assert len(b) == 2
    assert _msgs(b[1])[-1] == {"role": "user", "content": NOTE_1}
    assert _msgs(b[1])[-2] == {"role": "assistant", "content": loopmod.LOOP_PLACEHOLDER}
    assert MARKER not in json.dumps(b[1])
    assert b[1]["options"]["num_predict"] == 1024                         # loop 1 does not shrink the reply limit
    assert "  ! reply repeated itself; asked for a short answer (1 of 2 retries)" in w.text.splitlines()
    assert "Short answer." in w.text and "(this message ended" not in w.text
    (ev,) = w.events("loop")
    assert (ev["count"], ev["limit"], ev["retry"]) == (1, 1024, True)


def test_c7_chat_loop_2_gets_the_firmer_note_and_only_the_next_call_gets_half_the_reply_limit(tmp_path):
    w = _chat(tmp_path, [reply(LOOP), reply(LOOP), reply(calls=[call("bash", command="ls")]), reply("Done.")])
    b = w.transport.bodies
    assert len(b) == 4
    assert [x["options"]["num_predict"] for x in b] == [1024, 1024, 512, 1024]
    assert _msgs(b[2])[-1] == {"role": "user", "content": NOTE_2}
    assert MARKER not in json.dumps(b)
    lines = w.text.splitlines()
    assert "  ! reply repeated itself; asked for a short answer (1 of 2 retries)" in lines
    assert ("  ! reply repeated itself again; asked for a very short answer, next reply at most 512 tokens "
            "(2 of 2 retries)") in lines
    assert [(e["count"], e["limit"], e["next_limit"]) for e in w.events("loop")] == [(1, 1024, None), (2, 1024, 512)]


@pytest.mark.parametrize("limit,half", [(300, 256), (2048, 1024), (512, 256), (200, 200)])
def test_c7_half_the_reply_limit_is_never_below_256_nor_above_the_limit(tmp_path, limit, half):
    w = _chat(tmp_path, [reply(LOOP), reply(LOOP), reply("Done.")], reply_limit=limit)
    assert [x["options"]["num_predict"] for x in w.transport.bodies] == [limit, limit, half]


def test_c7_chat_loop_3_ends_the_message_as_repetition_in_plain_words_with_real_choices(tmp_path):
    w = _chat(tmp_path, [reply(LOOP), reply(LOOP), reply(LOOP), reply("never asked")])
    b = w.transport.bodies
    assert len(b) == 3 and b[2]["options"]["num_predict"] == 512
    assert [(e["count"], e["retry"]) for e in w.events("loop")] == [(1, True), (2, True), (3, False)]
    (end,) = w.events("message_end")
    assert end["end_reason"] == "repetition"
    text = w.text
    assert "  (this message ended: repetition; the reply repeated itself 3 times)" in text.splitlines()
    assert "kept repeating" in text and "send your message again" in text and "/model" in text
    after = text[text.index("kept repeating"):text.index("(That conversation has ended")]
    assert set(re.findall(r"(?<![\w/])/[a-z]+", after)) <= {"/new", "/model", "/undo", "/stats", "/help"}
    assert set(re.findall(r"(?<![\w/])/[a-z]+", after)) >= {"/model"}
    (rec,) = w.records()
    assert [r["summary"]["reason"] for r in rec if r["kind"] == "summary"] == ["repetition"]


def test_c7_each_message_gets_its_own_two_retries(tmp_path):
    keys = ["1", "first", "second", "/quit"]
    w = _chat(tmp_path, [reply(LOOP), reply("a"), reply(LOOP), reply("b")], keys=keys)
    assert [e["count"] for e in w.events("loop")] == [1, 1]
    assert w.text.count("(1 of 2 retries)") == 2
    assert [x["options"]["num_predict"] for x in w.transport.bodies] == [1024] * 4


def test_c7_a_cut_on_the_halved_call_names_the_halved_limit(tmp_path):
    w = _chat(tmp_path, [reply(LOOP), reply(LOOP), reply(thinking="t" * 50, done_reason="length", ev=512),
                         reply("Done.")])
    b = w.transport.bodies
    assert [x["options"]["num_predict"] for x in b] == [1024, 1024, 512, 1024]
    (cut,) = w.events("cut")
    assert cut["limit"] == 512
    assert "the 512-token limit" in _msgs(b[3])[-1]["content"]


# --- 4. the looping reply's own words are nowhere --------------------------------------------------------------


@pytest.mark.parametrize("n_loops", [1, 2, 3])
def test_c7_a_plain_marker_in_a_looping_reply_is_in_no_event_screen_line_record_line_or_later_request(tmp_path, n_loops):
    replies = [reply(LOOP)] * n_loops + [reply("Fine.")]
    w = _chat(tmp_path, replies)
    assert w.events("loop"), "no loop event"
    assert MARKER not in json.dumps(w.events()), "an event holds the model's own words"
    assert MARKER not in w.text, "the screen shows the model's own words"
    assert MARKER not in w.record_text(), "the record holds the model's own words"
    assert MARKER not in json.dumps(w.transport.bodies), "a later request holds the model's own words"
