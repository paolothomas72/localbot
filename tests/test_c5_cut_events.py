"""BRIEF C5 F2: a reply that is cut at the output limit, or that cannot be understood, is now an event and a line on
the screen. Events only: the nudges, limits, order and the messages sent to the model are those of c17748c."""
from __future__ import annotations

import hashlib
import json

import pytest

from localbot import chat
from localbot.config import get_profile
from localbot.loop import run_task
from tests.fakes import FakeShell, ScriptedTransport, call, provider, reply
from tests.test_chat import SECRET, World

PRODUCT = get_profile("core4_product")
LIMIT = PRODUCT.provider.num_predict_for("fake:1")
SEED = {".env": f"API_KEY={SECRET}\n", "notes.txt": "N\n"}
THINK = "t" * 1800

# --- the scenarios (a script of model replies each) -----------------------------------------------------------


def cut_thinking_only():
    return [reply(thinking=THINK, done_reason="length", ev=512) for _ in range(8)]


def unreadable():
    bad = '{"name": "read", "arguments": {"path": "' + SECRET + '"'   # call-shaped, cut off: nothing parses
    return [reply(bad), reply(bad), reply("ok, giving up on that.")]


def fenced():
    return [reply('```json\n<tool_call>{"name": "read", "arguments": {"path": "notes.txt"}}</tool_call>\n```'),
            reply("ok.")]


def normal():
    return [reply(calls=[call("read", path="notes.txt")]), reply("it says N.")]


SCENARIOS = {"cut": cut_thinking_only, "unreadable": unreadable, "fenced": fenced, "normal": normal}


def _core(res):
    """What the run did, by the fields the brief names; the messages sent to the model as a count and a fingerprint."""
    blob = json.dumps(res.messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return {"reason": res.reason, "n_messages": len(res.messages), "messages_sha256": hashlib.sha256(blob).hexdigest(),
            "said": res.said, "tool_results": res.tool_results,
            "nudges": res.nudges, "vetoes": res.vetoes, "veto_texts": res.veto_texts,
            "turns": [[t.call_source, t.calls] for t in res.turns]}


def run_direct(disk, scenario, on_event=None):
    ws = disk.ws(dict(SEED))
    script = SCENARIOS[scenario]()
    return run_task(PRODUCT, provider(ScriptedTransport(script), PRODUCT.provider), "fake:1", "clean up the folder",
                    ws, shell=FakeShell({}), on_event=on_event)


GOLDEN = json.loads(r'''{"cut": {"messages_sha256": "e2d842789501d5b9139b58302da765dd2da0807a5b371a6f28c0a6ce9ced1028", "n_messages": 18, "nudges": {"cut": 8, "parse": 0, "stuck": 0}, "reason": "max_steps", "said": ["", "", "", "", "", "", "", ""], "tool_results": [], "turns": [["none", []], ["none", []], ["none", []], ["none", []], ["none", []], ["none", []], ["none", []], ["none", []]], "veto_texts": [], "vetoes": 0}, "fenced": {"messages_sha256": "27cb1e7d09777a90b4136031825ccd0526fac57e4e70bed50ce48a37bdddc893", "n_messages": 5, "nudges": {"cut": 0, "parse": 1, "stuck": 0}, "reason": "done", "said": ["```json\n<tool_call>{\"name\": \"read\", \"arguments\": {\"path\": \"notes.txt\"}}</tool_call>\n```", "ok."], "tool_results": [], "turns": [["none", []], ["none", []]], "veto_texts": [], "vetoes": 0}, "normal": {"messages_sha256": "cef2d4a71e6c8e4143e6a362c687768fe42d4a9d32fe805c34dc9ad0c9e11e20", "n_messages": 5, "nudges": {"cut": 0, "parse": 0, "stuck": 0}, "reason": "done", "said": ["", "it says N."], "tool_results": ["1\tN"], "turns": [["native", ["read"]], ["none", []]], "veto_texts": [], "vetoes": 0}, "unreadable": {"messages_sha256": "ed8b9ef678e64e8cb5ad67c52550ff81a21b0c95ed096f96e4126538bed5ba50", "n_messages": 7, "nudges": {"cut": 0, "parse": 2, "stuck": 0}, "reason": "done", "said": ["{\"name\": \"read\", \"arguments\": {\"path\": \"[REDACTED]\"", "{\"name\": \"read\", \"arguments\": {\"path\": \"[REDACTED]\"", "ok, giving up on that."], "tool_results": [], "turns": [["none", []], ["none", []], ["none", []]], "veto_texts": [], "vetoes": 2}}''')   # recorded by running run_direct without on_event on c17748c. C6: only the 'cut' messages_sha256 was re-recorded, because the product-rules cut note changed on purpose (tests/test_c6_chat_settings.py holds the unchanged exam-rules hash)


def run_chat(tmp_path, scenario, message="clean up the folder"):
    w = World(tmp_path, ["1", message, "/quit"], replies=SCENARIOS[scenario](), seed=dict(SEED))
    assert chat.chat_main("main", None, w.deps, reply_limit=512, steps=8) == 0, w.text  # C7: the C5 settings pinned
    return w


def lines(w, start):
    return [ln for ln in w.text.splitlines() if ln.startswith(start)]


# --- 1. every reply cut, thinking only ------------------------------------------------------------------------


def test_c5_eight_cut_replies_make_eight_events_eight_lines_and_a_count_on_the_end_line(tmp_path):
    w = run_chat(tmp_path, "cut")
    cuts = w.events("cut")
    assert len(cuts) == 8
    summary = [r["summary"] for rec in w.records() for r in rec if r["kind"] == "summary"][0]
    for i, ev in enumerate(cuts):
        assert ev["step"] == i and ev["limit"] == 512 and ev["done_reason"] == "length" and ev["tokens"] == 512
        assert ev["visible"] is False and ev["thinking_tokens"] == summary["provider"][i]["thinking_tokens_est"] > 0
    shown = lines(w, "  ! reply cut")
    assert shown == ["  ! reply cut at 512 tokens (all of it was thinking, nothing visible); asked for a shorter one"] * 8
    assert "  (this message ended: max_steps; 8 replies cut, 0 not understood, 0 tools run)" in w.text.splitlines()


def test_c5_the_counts_on_the_end_line_start_again_for_each_message(tmp_path):
    w = World(tmp_path, ["1", "clean up", "and again", "/quit"], replies=cut_thinking_only() + cut_thinking_only(),
              seed=dict(SEED))
    assert chat.chat_main("main", None, w.deps, reply_limit=512, steps=8) == 0, w.text  # C7: the C5 settings pinned
    ends = [ln for ln in w.text.splitlines() if ln.startswith("  (this message ended")]
    assert ends == ["  (this message ended: max_steps; 8 replies cut, 0 not understood, 0 tools run)"] * 2


def test_c5_a_cut_reply_with_visible_text_says_so(tmp_path):
    script = [reply("I will start by looking at", done_reason="length", ev=512), reply("ok, done.")]
    w = World(tmp_path, ["1", "clean up", "/quit"], replies=script, seed=dict(SEED))
    assert chat.chat_main("main", None, w.deps, reply_limit=512, steps=8) == 0, w.text  # C7: the C5 settings pinned
    ev = w.events("cut")
    assert len(ev) == 1 and ev[0]["visible"] is True
    assert lines(w, "  ! reply cut") == ["  ! reply cut at 512 tokens; asked for a shorter one"]


# --- 2. replies that cannot be understood -------------------------------------------------------------------


def test_c5_unreadable_call_shaped_text_is_an_event_and_a_line_and_the_text_is_nowhere(tmp_path):
    w = run_chat(tmp_path, "unreadable")
    evs = w.events("parse_fail")
    assert [(e["step"], e["reason"]) for e in evs] == [(0, "unreadable"), (1, "unreadable")]
    assert set(evs[0]) == {"event", "message", "step", "reason"}   # fixed fields only: no model text
    assert lines(w, "  ! reply not understood") == ["  ! reply not understood (no tool call or answer found); asked again"] * 2
    assert SECRET not in json.dumps(w.events()) and SECRET not in w.text and SECRET not in w.record_text()


def test_c5_a_call_inside_a_fence_is_reported_as_fenced(tmp_path):
    w = run_chat(tmp_path, "fenced")
    assert [(e["step"], e["reason"]) for e in w.events("parse_fail")] == [(0, "fenced")]
    assert len(lines(w, "  ! reply not understood")) == 1


# --- 3. a normal reply -------------------------------------------------------------------------------------------


def test_c5_a_normal_run_has_no_new_lines_and_no_new_events(tmp_path):
    w = run_chat(tmp_path, "normal")
    assert w.events("cut") == [] and w.events("parse_fail") == []
    assert lines(w, "  ! ") == [] and "this message ended" not in w.text


# --- 4. no behavior change -----------------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_c5_events_change_nothing_the_model_or_the_summary_sees(disk, scenario):
    plain = _core(run_direct(disk, scenario))
    events = []
    watched = _core(run_direct(disk, scenario, on_event=events.append))
    assert watched == plain
    assert json.loads(json.dumps(plain)) == GOLDEN[scenario]   # and it is what c17748c gave


def test_c5_the_events_a_run_sends_are_the_new_kinds_only_where_a_reply_was_cut_or_unreadable(disk):
    seen = {}
    for scenario in SCENARIOS:
        kinds = []
        run_direct(disk, scenario, on_event=lambda e, k=kinds: k.append(e["event"]))
        seen[scenario] = kinds
    assert seen["cut"].count("cut") == 8 and seen["unreadable"].count("parse_fail") == 2
    assert "cut" not in seen["normal"] and "parse_fail" not in seen["normal"]
    assert "cut" not in seen["unreadable"] and "parse_fail" not in seen["cut"]
