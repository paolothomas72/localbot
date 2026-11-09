"""BRIEF C7 part C: the ladder measures what matters: ``worked`` (not ``reached``), tool calls written as text, asks,
loops, a steps climb, the card's memory from nvidia-smi, and "slow: maybe off the card".

No Ollama, no container, no GPU, no nvidia-smi: made-up session records, a fake ``run_try`` and the chat on the fakes
of ``tests/test_chat.py`` with a fake card reader."""
from __future__ import annotations

import csv
import dataclasses
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

from localbot import chat, telemetry
from localbot.telemetry import VramReading
from tests.fakes import call, reply
from tests.test_chat import World

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "chat_ladder.py"
spec = importlib.util.spec_from_file_location("chat_ladder_c7", SCRIPT)
ladder = importlib.util.module_from_spec(spec)
sys.modules["chat_ladder_c7"] = ladder
spec.loader.exec_module(ladder)

SHAPE_ACTION = '{"action": "bash", "action_input": "ls -F"}'
SHAPE_NAME_ARGS = '```json\n{"name": "read", "arguments": {"path": "notes.txt"}}\n```'
SHAPE_TAG = '<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>'


# --- 1. worked and text tool calls, from made-up records ----------------------------------------------------------


@pytest.mark.parametrize("text", [SHAPE_ACTION, SHAPE_NAME_ARGS, SHAPE_TAG, "Sure. " + SHAPE_ACTION,
                                  '[{"name": "bash", "arguments": {"command": "ls"}}]', "<TOOL_CALL>ls</TOOL_CALL>"])
def test_c7_tool_calls_written_as_text_are_seen(text):
    assert ladder.looks_like_text_tool_call(text) is True


@pytest.mark.parametrize("text", ["All tidy: I left .env alone and kept old_draft.txt.",
                                  "The action I took was to list the files; the arguments were none.",
                                  '{"total": 3, "files": ["a"]}', "", "   "])
def test_c7_ordinary_answers_are_not_tool_calls(text):
    assert ladder.looks_like_text_tool_call(text) is False


def _record(tmp_path, events, end_reason="done", gpu=None, name="rec.jsonl"):
    lines = [{"kind": "header", "bot": "ladder", "model": "m:1", "reply_limit": 1024, "context": 8192, "steps": 12,
              "think": "default"}]
    if gpu is not False:
        lines.append({"kind": "gpu_fit", "model_bytes": 8_000_000_000, "on_card_bytes": 8_000_000_000, "fits": "yes",
                      "doubtful": False, **(gpu or {})})
    lines += [{"kind": "event", "event": e} for e in events]
    lines.append({"kind": "event", "event": {"event": "message_end", "end_reason": end_reason, "steps": 3, "wall_s": 9}})
    lines.append({"kind": "summary", "conversation": 1, "summary": {"reason": end_reason, "elapsed_s": 9.0,
                                                                     "tok_s": 50.0, "provider": [{}] * 3}})
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    return p


def test_c7_a_real_answer_after_done_worked(tmp_path):
    row = ladder.parse_record(_record(tmp_path, [{"event": "tool", "status": "ran"},
                                                 {"event": "answer", "text": "All tidy."}]))
    assert row["worked"] is True and row["text_tool_calls"] == 0


@pytest.mark.parametrize("shape", [SHAPE_ACTION, SHAPE_NAME_ARGS, SHAPE_TAG])
def test_c7_an_answer_that_is_a_tool_call_written_as_text_did_not_work(tmp_path, shape):
    row = ladder.parse_record(_record(tmp_path, [{"event": "answer", "text": shape}]))
    assert row["worked"] is False and row["text_tool_calls"] == 1
    assert row["answered"] == "yes"          # the old column still says yes: that is why it was not enough


@pytest.mark.parametrize("end,events", [
    ("max_steps", [{"event": "answer", "text": "All tidy."}]),
    ("repetition", [{"event": "answer", "text": "All tidy."}]),
    ("done", []),
    ("done", [{"event": "answer", "text": "  "}]),
])
def test_c7_not_done_or_no_answer_did_not_work(tmp_path, end, events):
    assert ladder.parse_record(_record(tmp_path, events, end_reason=end))["worked"] is False


def test_c7_asks_loops_and_steps_are_counted(tmp_path):
    events = [{"event": "ask", "paths": ["old_draft.txt"]}, {"event": "ask_answer", "answer": "no"},
              {"event": "ask", "paths": ["notes.txt"]},
              {"event": "loop", "count": 1, "limit": 1024, "retry": True},
              {"event": "answer", "text": "Done."}]
    row = ladder.parse_record(_record(tmp_path, events))
    assert (row["asked_about_files"], row["loops"], row["steps_used"], row["worked"]) == (2, 1, 3, True)


def test_c7_a_timed_out_try_did_not_work(tmp_path):
    row = ladder.parse_record(_record(tmp_path, [{"event": "answer", "text": "ok"}]), timed_out_after=600)
    assert row["worked"] is False


def test_c7_the_card_columns_come_from_the_gpu_fit_line(tmp_path):
    row = ladder.parse_record(_record(tmp_path, [], gpu={"card_used_mib": 11000, "card_total_mib": 12227}))
    assert (row["card_used_gb"], row["card_total_gb"]) == (11.53, 12.82)


@pytest.mark.parametrize("gpu", [False, {}, {"card_used_mib": None, "card_total_mib": None}])
def test_c7_no_card_reading_leaves_the_card_columns_empty(tmp_path, gpu):
    row = ladder.parse_record(_record(tmp_path, [], gpu=gpu))
    assert row["card_used_gb"] is None and row["card_total_gb"] is None
    text = ladder._csv_text([dict(row, level=1, **{"try": 1})])
    got = next(csv.DictReader(io.StringIO(text)))
    assert got["card_used_gb"] == "" and got["card_total_gb"] == ""


def test_c7_no_record_row_has_every_new_column():
    row = ladder.no_record_row(1024, 8192, "default", 1, 3.0, steps=16)
    assert row["worked"] is False and row["steps_limit"] == 16
    assert all(c in row for c in ("text_tool_calls", "asked_about_files", "loops", "card_used_gb", "card_total_gb"))


# --- 2. the chat records the card (nvidia-smi) after the first reply -------------------------------------------


def _try_world(tmp_path, vram):
    w = World(tmp_path, ["1", "hello", "/quit"], replies=[reply("hi")])
    w.deps = dataclasses.replace(w.deps, vram=vram)
    return w


def test_c7_the_chat_reads_the_card_with_the_existing_telemetry_reader():
    assert chat.read_vram is telemetry.read_vram          # no second nvidia-smi parser
    assert chat.Deps.__dataclass_fields__["vram"].default is None   # tests never run the real nvidia-smi


def test_c7_with_nvidia_smi_the_gpu_fit_line_holds_the_card_used_and_total(tmp_path):
    calls = []
    w = _try_world(tmp_path, lambda: calls.append(1) or VramReading(used_mib=9000, total_mib=12227, util_pct=40))
    assert w.run() == 0, w.text
    (rec,) = w.records()
    (g,) = [r for r in rec if r["kind"] == "gpu_fit"]
    assert (g["card_used_mib"], g["card_total_mib"]) == (9000, 12227) and calls == [1]


def test_c7_stats_shows_the_card_memory_when_nvidia_smi_gave_it(tmp_path):
    w = World(tmp_path, ["1", "hello", "/stats", "/quit"], replies=[reply("hi")])
    w.deps = dataclasses.replace(w.deps, vram=lambda: VramReading(used_mib=9000, total_mib=12227))
    assert w.run() == 0, w.text
    assert "card memory used 9.4 of 12.8 GB (nvidia-smi)" in w.text


@pytest.mark.parametrize("vram", [None, lambda: None, lambda: (_ for _ in ()).throw(OSError("no nvidia-smi"))])
def test_c7_without_nvidia_smi_nothing_breaks_and_the_card_is_unknown(tmp_path, vram):
    w = _try_world(tmp_path, vram)
    assert w.run() == 0, w.text
    (rec,) = w.records()
    (g,) = [r for r in rec if r["kind"] == "gpu_fit"]
    assert g["card_used_mib"] is None and g["card_total_mib"] is None
    assert "hi" in w.text


def test_c7_one_try_passes_the_steps_to_the_chat(tmp_path):
    w = World(tmp_path, [], replies=[reply("left it alone")], seed={})
    args = ladder.parse_args(["--one-try", "--model", "qwen3:8b", "--task", "clean up this folder",
                              "--reply-limit", "1024", "--context", "8192", "--steps", "16"])
    assert ladder.one_try(args, deps=w.deps) == 0, w.text
    rec = next((w.home / ".localbot" / "bots" / "ladder" / "sessions").glob("*.jsonl"))
    assert ladder.parse_record(rec)["steps_limit"] == 16


def test_c7_every_try_command_passes_steps_explicitly():
    assert ladder.child_command("m:1", "t", 1024, 8192, "default")[-2:] == ["--steps", "12"]
    cmd = ladder.child_command("m:1", "t", 1024, 8192, "off", steps=20)
    assert cmd[cmd.index("--steps") + 1] == "20" and cmd[cmd.index("--think") + 1] == "off"


# --- 3. the climb: best = most worked; the steps climb; slow and doubtful ---------------------------------------


def _row(**kw):
    base = dict(reply_limit=512, context=8192, think="default", steps_limit=12, steps_used=3, replies_cut=0,
                cut_by_thinking=0, tools_run=2, ended_why="done", answered="yes", worked=True, text_tool_calls=0,
                asked_about_files=0, loops=0, seconds=30.0, tokens_per_second=100.0, model_gb=6.0, on_card_gb=6.0,
                fits="yes", doubtful=False, card_used_gb=None, card_total_gb=None, key_in_record=False)
    base.update(kw)
    return base


class Scripted:
    """A fake ``run_try``: ``plan(rp, ctx, think, steps)`` -> list of row overrides (one per try)."""

    def __init__(self, plan):
        self.plan, self.calls = plan, []

    def __call__(self, rp, ctx, think, n, steps=12):
        self.calls.append((rp, ctx, think, steps, n))
        return _row(reply_limit=rp, context=ctx, think=think, steps_limit=steps, **self.plan(rp, ctx, think, steps)[n - 1])


def _levels(s):
    out = []
    for rp, ctx, think, steps, _n in s.calls:
        if not out or out[-1] != (rp, ctx, think, steps):
            out.append((rp, ctx, think, steps))
    return out


def test_c7_best_is_the_most_tries_that_worked_not_the_most_that_reached(tmp_path):
    def plan(rp, ctx, think, steps):
        worked = {512: 1, 1024: 3, 2048: 2, 4096: 0}.get(rp, 3)
        return [dict(worked=i < worked, seconds=90 if rp == 1024 else 20) for i in range(3)]
    s = Scripted(plan)
    ladder.run_ladder(3, s, say=lambda t: None)
    assert {lv[0] for lv in _levels(s) if lv[1] != 8192} == {1024}     # slower, but every try really worked


def test_c7_no_steps_climb_when_no_try_ran_out_of_steps(tmp_path):
    s = Scripted(lambda rp, ctx, think, steps: [dict()] * 3)
    rows, levels = ladder.run_ladder(3, s, say=lambda t: None)
    assert {lv[3] for lv in _levels(s)} == {12} and len(levels) == 8
    assert all(c[3] == 12 for c in s.calls)


def test_c7_the_steps_climb_goes_on_only_while_a_try_ended_max_steps(tmp_path):
    def plan(rp, ctx, think, steps):
        here = (rp, ctx, think) == (512, 8192, "default")
        out_of_steps = here and steps in (12, 16)
        return [dict(ended_why="max_steps" if (out_of_steps and i == 0) else "done",
                     worked=here and not (out_of_steps and i == 0)) for i in range(3)]
    s = Scripted(plan)
    ladder.run_ladder(3, s, say=lambda t: None)
    steps_levels = [lv for lv in _levels(s) if lv[3] != 12 and lv[2] == "default"]
    assert steps_levels == [(512, 8192, "default", 16), (512, 8192, "default", 20)]


def test_c7_the_steps_climb_stops_at_25(tmp_path):
    def plan(rp, ctx, think, steps):
        return [dict(ended_why="max_steps", worked=False, seconds=10 if (rp, ctx) == (512, 8192) else 50)] * 3
    s = Scripted(plan)
    rows, levels = ladder.run_ladder(3, s, say=lambda t: None)
    assert [lv[3] for lv in _levels(s) if lv[2] == "default" and lv[:2] == (512, 8192)] == [12, 16, 20, 25]
    assert max(c[3] for c in s.calls) == 25 and len(levels) == ladder.LEVELS_MAX == 11


def test_c7_the_think_off_level_uses_the_best_row_of_all_climbs_with_its_steps(tmp_path):
    def plan(rp, ctx, think, steps):
        if think == "default" and (rp, ctx) == (512, 8192):
            if steps == 12:
                return [dict(ended_why="max_steps", worked=False)] * 3
            return [dict(worked=True, seconds=5)] * 3
        return [dict(worked=False)] * 3
    s = Scripted(plan)
    ladder.run_ladder(3, s, say=lambda t: None)
    assert s.calls[-1][:4] == (512, 8192, "off", 16)


def test_c7_slow_and_doubtful_marks_the_level_and_stops_that_climb(tmp_path):
    def plan(rp, ctx, think, steps):
        return [dict(doubtful=True, tokens_per_second=60.0 if rp == 2048 else 100.0)] * 3
    s = Scripted(plan)
    rows, levels = ladder.run_ladder(3, s, say=lambda t: None)
    assert [lv[0] for lv in _levels(s) if lv[1] == 8192 and lv[2] == "default"] == [512, 1024, 2048]
    slow = [lv for lv in levels if lv["reply_limit"] == 2048]
    assert slow[0]["stop"] == "slow: maybe off the card"
    assert all(lv[0] != 2048 for lv in _levels(s) if lv[1] != 8192 or lv[2] == "off")   # never the best one


def test_c7_slow_alone_is_not_a_stop_when_the_card_reading_is_trusted(tmp_path):
    s = Scripted(lambda rp, ctx, think, steps: [dict(doubtful=False, tokens_per_second=60.0 if rp == 2048 else 100.0)] * 3)
    rows, levels = ladder.run_ladder(3, s, say=lambda t: None)
    assert len(levels) == 8 and not any(lv["stop"] for lv in levels)


def test_c7_exactly_70_percent_is_not_slow(tmp_path):
    s = Scripted(lambda rp, ctx, think, steps: [dict(doubtful=True, tokens_per_second=70.0 if rp == 2048 else 100.0)] * 3)
    rows, levels = ladder.run_ladder(3, s, say=lambda t: None)
    assert not any(lv["stop"] for lv in levels)


# --- 4. the files ----------------------------------------------------------------------------------------------


def test_c7_csv_has_the_new_columns_and_the_md_says_what_worked_proves(tmp_path):
    rows, levels = ladder.run_ladder(1, Scripted(lambda rp, ctx, think, steps: [dict()]), say=lambda t: None)
    csv_path, md_path = ladder.write_outputs(rows, levels, tmp_path, "m", "t")
    got = list(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8"))))
    for c in ("worked", "text_tool_calls", "asked_about_files", "loops", "steps_used", "card_used_gb", "card_total_gb"):
        assert c in got[0], c
    md = " ".join(md_path.read_text(encoding="utf-8").split())
    assert "`worked` is a proxy" in md and "cannot tell whether the answer was good" in md
    assert "`reached` is weak" in md


def test_c7_the_estimate_counts_the_steps_climb(tmp_path):
    text = ladder.estimate_text(tries=3, limit_s=600)
    assert "11 levels" in text and "33 tries" in text and "5.5 hours" in text
