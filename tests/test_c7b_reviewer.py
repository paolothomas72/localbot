"""C7b (reviewer chat 25): five behaviours of C7 that were right but had no test (each one found by a deliberate break
that no test noticed). Same fakes as ``tests/test_c7_loop_retry.py`` and ``tests/test_c7_ladder.py``."""
from __future__ import annotations

import pytest

from localbot import chat
from tests.fakes import call, reply
from tests.test_c7_ladder import Scripted, ladder
from tests.test_c7_loop_retry import LOOP, _chat, _direct


def test_c7b_a_halved_limit_never_carries_into_the_next_message(tmp_path):
    # Loop 2 on the message's LAST step: the halved call would have been for this message. The next message of the
    # person starts with the whole reply limit again.
    w = _chat(tmp_path, [reply(LOOP), reply(LOOP), reply("b")], keys=["1", "first", "second", "/quit"], steps=2)
    assert [e["end_reason"] for e in w.events("message_end")] == ["max_steps", "done"]
    assert [x["options"]["num_predict"] for x in w.transport.bodies] == [1024, 1024, 1024]


@pytest.mark.parametrize("ending", ["done", "max_steps"])
def test_c7b_the_repeating_lines_are_shown_only_when_a_message_ended_repeating(tmp_path, ending):
    if ending == "done":
        w = _chat(tmp_path, [reply(LOOP), reply("Short answer.")])
    else:
        w = _chat(tmp_path, [reply(LOOP), reply(LOOP)], steps=2)
    (end,) = w.events("message_end")
    assert end["end_reason"] == ending
    for line in chat.REPEAT_LINES:
        assert line not in w.text


def test_c7b_a_bare_provider_gets_its_own_reply_limit_back_after_the_halved_call(disk):
    # run_task with the provider itself (no chat meter around it): the halved call swaps the provider's settings for
    # that one call; the call after it must send the profile's own limit again.
    res, tr = _direct(disk, "core4_product", [reply(LOOP), reply(LOOP), reply(calls=[call("read", path="notes.txt")]),
                                              reply("Done.")], repeat_retries=2)
    assert res.reason == "done"
    assert [b["options"]["num_predict"] for b in tr.bodies] == [512, 512, 256, 512]


def test_c7b_slow_is_measured_against_level_1_not_the_level_before(tmp_path):
    # 60 tokens/s is 80 % of the level before (75) but only 60 % of level 1 (100): slow.
    speeds = {512: 100.0, 1024: 75.0, 2048: 60.0}
    s = Scripted(lambda rp, ctx, think, steps: [dict(doubtful=True, tokens_per_second=speeds.get(rp, 100.0))] * 3)
    rows, levels = ladder.run_ladder(3, s, say=lambda t: None)
    first = {lv["reply_limit"]: lv["stop"] for lv in levels if lv["context"] == 8192 and lv["think"] == "default"}
    assert first == {512: "", 1024: "", 2048: ladder.SLOW_STOP}


@pytest.mark.parametrize("text", [
    '{"type": "function", "function": {"name": "bash", "arguments": "{\\"command\\": \\"ls\\"}"}}',
    '{"tool_calls": [{"function": {"name": "read", "arguments": {"path": "notes.txt"}}}]}',
])
def test_c7b_tool_calls_nested_in_a_function_object_are_seen(text):
    assert ladder.looks_like_text_tool_call(text) is True


def test_c7b_every_try_is_given_its_steps(tmp_path):
    seen = []

    def run_try(rp, ctx, think, n, steps):  # no default: a try that is not given its steps fails here
        seen.append(steps)
        return dict(ended_why="done", answered="yes", worked=True, seconds=5.0, tokens_per_second=100.0, fits="yes")
    ladder.run_ladder(1, run_try, say=lambda t: None)
    assert seen and set(seen) == {12}
