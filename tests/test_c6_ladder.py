"""BRIEF C6 part C: the measuring script ``scripts/chat_ladder.py``, tested with fakes.

No Ollama, no container, no real chat process: the chat runs on the fakes of ``tests/test_chat.py`` and the ladder's
"one try" is a function the tests give made-up results. What is NOT tested here is a real run (see the report)."""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
import time
from pathlib import Path

import pytest

from localbot import chat
from tests.fakes import call, reply
from tests.test_chat import World

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "chat_ladder.py"
spec = importlib.util.spec_from_file_location("chat_ladder_c6", SCRIPT)
ladder = importlib.util.module_from_spec(spec)
sys.modules["chat_ladder_c6"] = ladder
spec.loader.exec_module(ladder)


def _row(**kw):
    base = dict(reply_limit=512, context=8192, think="default", steps_limit=8, steps_used=3, replies_cut=0,
                cut_by_thinking=0, tools_run=2, ended_why="done", answered="yes", seconds=30.0, tokens_per_second=40.0,
                model_gb=6.0, on_card_gb=6.0, fits="yes", doubtful=False, key_in_record=False)
    base.update(kw)
    return base


# --- 1. the test pack: one place, the same every try -------------------------------------------------------------


def _snapshot(root: Path):
    return {p.relative_to(root).as_posix(): (p.read_bytes() if p.is_file() else None)
            for p in sorted(root.rglob("*"))}


def test_c6_the_pack_has_the_five_files_and_the_fake_key_only_in_env(tmp_path):
    ladder.write_pack(tmp_path)
    snap = _snapshot(tmp_path)
    assert sorted(snap) == [".env", "notes.txt", "old_draft.txt", "prices.py", "todo.md"]
    assert ladder.FAKE_KEY.encode() in snap[".env"]
    assert [n for n, b in snap.items() if ladder.FAKE_KEY.encode() in b] == [".env"]


def test_c6_a_work_folder_is_made_exactly_like_the_pack_again_before_each_try(tmp_path):
    work = tmp_path / ".localbot" / "bots" / "ladder" / "work"
    ladder.write_pack(work.mkdir(parents=True) or work)
    first = _snapshot(work)
    (work / "notes.txt").write_text("CHANGED", encoding="utf-8")
    (work / "prices.py").unlink()
    (work / "_moved").mkdir()
    (work / "_moved" / "x.txt").write_text("x", encoding="utf-8")
    (work / "extra.txt").write_text("e", encoding="utf-8")
    ladder.reset_work(work)
    assert _snapshot(work) == first


@pytest.mark.parametrize("where", [("home", "bots", "main", "work"), ("home", "bots", "ladder", "other"),
                                   ("home", "ladder", "work"), ("work",)])
def test_c6_only_the_ladder_bots_work_folder_is_ever_emptied(tmp_path, where):
    d = tmp_path.joinpath(*where)
    d.mkdir(parents=True)
    (d / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError):
        ladder.reset_work(d)
    assert (d / "keep.txt").read_text(encoding="utf-8") == "keep"


# --- 2. one try drives the real chat: bot ladder, "n" to every question -----------------------------------------


def _try_world(tmp_path):
    w = World(tmp_path, [], replies=[reply(calls=[call("bash", command="rm notes.txt")]), reply("left it alone")],
              seed={})
    return w


def test_c6_one_try_uses_bot_ladder_answers_n_to_every_ask_and_loses_nothing(tmp_path):
    w = _try_world(tmp_path)
    args = ladder.parse_args(["--one-try", "--model", "qwen3:8b", "--task", "clean up this folder",
                              "--reply-limit", "1024", "--context", "16384"])
    assert ladder.one_try(args, deps=w.deps) == 0, w.text
    bots = sorted(p.name for p in (w.home / ".localbot" / "bots").iterdir())
    assert bots == ["ladder"] and ladder.BOT == "ladder"
    work = w.home / ".localbot" / "bots" / "ladder" / "work"
    assert _snapshot(work) == _snapshot_of_pack(tmp_path)           # notes.txt is still there: the answer was "n"
    rec = next((work.parent / "sessions").glob("*.jsonl"))             # the ladder bot's record, not World's "main"
    evs = [json.loads(x)["event"] for x in rec.read_text(encoding="utf-8").splitlines() if '"kind": "event"' in x]
    assert [e["answer"] for e in evs if e["event"] == "ask_answer"] == ["no"]   # asked once, answered n by the driver
    row = ladder.parse_record(rec)
    assert row["reply_limit"] == 1024 and row["context"] == 16384 and row["tools_run"] == 0 and row["answered"] == "yes"
    assert w.transport.bodies[0]["options"]["num_predict"] == 1024 and w.transport.bodies[0]["options"]["num_ctx"] == 16384


def _snapshot_of_pack(tmp_path):
    d = tmp_path / "_pack_ref"
    d.mkdir()
    ladder.write_pack(d)
    return _snapshot(d)


def test_c6_the_driver_gives_the_task_once_then_quits_and_never_picks_a_model():
    d = ladder.Driver("clean up this folder")
    assert d("you> ") == "clean up this folder"
    assert d(chat.PROMPT_ASK) == "n" and d(chat.PROMPT_ASK) == "n"
    assert d("you> ") == "/quit" and d("you> ") == "/quit"
    assert d("Pick a model (number or name)> ") is None
    assert d.asked == 2


# --- 3. a made-up record gives the table row -----------------------------------------------------------------------


def _record(tmp_path, events, summary=None, gpu=None, header=None, end=True, extra=""):
    lines = [{"kind": "header", "bot": "ladder", "model": "m:1", "reply_limit": 1024, "context": 8192, "steps": 8,
              "think": "default", **(header or {})}]
    if gpu is not False:
        lines.append({"kind": "gpu_fit", "model_bytes": 8_000_000_000, "on_card_bytes": 8_000_000_000, "fits": "yes",
                      "doubtful": False, **(gpu or {})})
    lines += [{"kind": "event", "event": e} for e in events]
    if summary is not None:
        lines.append({"kind": "summary", "conversation": 1, "summary": summary})
    if end:
        lines.append({"kind": "end", "reason": "quit"})
    p = tmp_path / "rec.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines) + extra, encoding="utf-8")
    return p


def test_c6_the_row_counts_come_from_the_record_alone(tmp_path):
    events = [
        {"event": "cut", "step": 1, "limit": 1024, "tokens": 1024, "thinking_tokens": 1024, "visible": False},
        {"event": "cut", "step": 2, "limit": 1024, "tokens": 1024, "thinking_tokens": 300, "visible": True},
        {"event": "cut", "step": 3, "limit": 1024, "tokens": 1024, "thinking_tokens": 1024, "visible": False},
        {"event": "tool", "step": 4, "tool": "bash", "status": "ran"},
        {"event": "tool", "step": 5, "tool": "bash", "status": "refused"},
        {"event": "tool", "step": 6, "tool": "bash", "status": "ran"},
        {"event": "answer", "step": 7, "text": "all tidy"},
        {"event": "message_end", "end_reason": "done", "steps": 7, "wall_s": 41.5},
    ]
    summary = {"reason": "done", "elapsed_s": 41.6, "tok_s": 33.3, "provider": [{}] * 7}
    row = ladder.parse_record(_record(tmp_path, events, summary))
    assert (row["reply_limit"], row["context"], row["think"], row["steps_limit"]) == (1024, 8192, "default", 8)
    assert (row["replies_cut"], row["cut_by_thinking"], row["tools_run"], row["steps_used"]) == (3, 2, 2, 7)
    assert (row["ended_why"], row["answered"], row["seconds"], row["tokens_per_second"]) == ("done", "yes", 41.6, 33.3)
    assert (row["model_gb"], row["on_card_gb"], row["fits"], row["doubtful"]) == (8.0, 8.0, "yes", False)
    assert row["reached"] is True and row["key_in_record"] is False


def test_c6_no_tool_and_no_answer_is_not_reached_and_max_steps_is_the_end_reason(tmp_path):
    events = [{"event": "cut", "limit": 512, "visible": False, "thinking_tokens": 512}] * 8 + [
        {"event": "message_end", "end_reason": "max_steps", "steps": 8, "wall_s": 70}]
    row = ladder.parse_record(_record(tmp_path, events, {"reason": "max_steps", "elapsed_s": 70.0, "tok_s": 7.0,
                                                          "provider": [{}] * 8}))
    assert row["reached"] is False and row["answered"] == "no" and row["ended_why"] == "max_steps"
    assert row["replies_cut"] == 8 and row["tools_run"] == 0


def test_c6_a_record_that_never_finished_is_a_timeout_with_the_limit_as_its_time(tmp_path):
    p = _record(tmp_path, [{"event": "cut", "limit": 512, "visible": False}], summary=None, end=False)
    row = ladder.parse_record(p, timed_out_after=600)
    assert row["ended_why"] == "timeout (600 s limit)" and row["seconds"] == 600 and row["reached"] is False


def test_c6_a_half_written_last_line_does_not_break_the_row(tmp_path):
    p = _record(tmp_path, [{"event": "tool", "status": "ran"}], summary=None, end=False, extra='\n{"kind": "eve')
    assert ladder.parse_record(p, timed_out_after=60)["tools_run"] == 1


def test_c6_a_record_with_the_fake_key_in_it_is_flagged(tmp_path):
    row = ladder.parse_record(_record(tmp_path, [{"event": "say", "text": f"key {ladder.FAKE_KEY}"}], None))
    assert row["key_in_record"] is True


def test_c6_a_missing_gpu_line_is_unknown_not_a_yes(tmp_path):
    row = ladder.parse_record(_record(tmp_path, [], None, gpu=False))
    assert row["fits"] == "unknown" and row["model_gb"] is None


# --- 4. the climb ---------------------------------------------------------------------------------------------------


class Scripted:
    """A fake ``run_try``: ``plan(rp, ctx, think)`` -> list of row overrides (one per try)."""

    def __init__(self, plan):
        self.plan, self.calls = plan, []

    def __call__(self, rp, ctx, think, n, steps=12):  # C7b: the ladder now always gives the steps
        self.calls.append((rp, ctx, think, n))
        return _row(reply_limit=rp, context=ctx, think=think, **self.plan(rp, ctx, think)[n - 1])


def _levels(s):
    out = []
    for rp, ctx, think, _n in s.calls:
        if not out or out[-1] != (rp, ctx, think):
            out.append((rp, ctx, think))
    return out


def _ok(rp, ctx, think):
    return [dict(seconds=100 - rp / 100)] * 3


def test_c6_the_whole_ladder_when_everything_fits(tmp_path):
    s = Scripted(_ok)
    rows, levels = ladder.run_ladder(3, s, say=lambda t: None)
    assert _levels(s) == [(512, 8192, "default"), (1024, 8192, "default"), (2048, 8192, "default"),
                          (4096, 8192, "default"),
                          (4096, 16384, "default"), (4096, 32768, "default"), (4096, 65536, "default"),
                          (4096, 8192, "off")]
    assert len(rows) == 8 * 3 and [r["try"] for r in rows[:3]] == [1, 2, 3]


def test_c6_best_reply_limit_is_the_most_tries_that_worked_then_the_fastest(tmp_path):  # C7: worked, not reached
    def plan(rp, ctx, think):
        reached = {512: 0, 1024: 2, 2048: 2, 4096: 1}.get(rp, 3)
        secs = {1024: 90, 2048: 60}.get(rp, 50)
        rows = []
        for i in range(3):
            ok = i < reached
            rows.append(dict(tools_run=1 if ok else 0, answered="no", seconds=secs, reached=ok, worked=ok))
        return rows
    s = Scripted(plan)
    ladder.run_ladder(3, s, say=lambda t: None)
    ctx_levels = [lv for lv in _levels(s) if lv[1] != 8192]
    assert ctx_levels and all(rp == 2048 for rp, _c, _t in ctx_levels)      # 2048: as many as 1024, but faster
    assert _levels(s)[-1][2] == "off"


def test_c6_it_stops_climbing_at_the_first_level_that_does_not_fit_and_keeps_that_level(tmp_path):
    def plan(rp, ctx, think):
        return [dict(fits="no" if rp == 2048 else "yes")] * 3
    s = Scripted(plan)
    ladder.run_ladder(3, s, say=lambda t: None)
    phase1 = [lv for lv in _levels(s) if lv[1] == 8192 and lv[2] == "default"]
    assert phase1 == [(512, 8192, "default"), (1024, 8192, "default"), (2048, 8192, "default")]


def test_c6_it_stops_climbing_the_context_when_it_stops_fitting(tmp_path):
    def plan(rp, ctx, think):
        return [dict(fits="no" if ctx == 32768 else "yes")] * 3
    s = Scripted(plan)
    ladder.run_ladder(3, s, say=lambda t: None)
    assert [lv[1] for lv in _levels(s) if lv[2] == "default" and lv[1] != 8192] == [16384, 32768]


def test_c6_it_stops_climbing_when_every_try_of_a_level_timed_out(tmp_path):
    def plan(rp, ctx, think):
        return [dict(ended_why="timeout (600 s limit)", reached=False, seconds=600)] * 3 if rp == 1024 else [dict()] * 3
    s = Scripted(plan)
    ladder.run_ladder(3, s, say=lambda t: None)
    assert [lv[0] for lv in _levels(s) if lv[1] == 8192 and lv[2] == "default"] == [512, 1024]


def test_c6_a_try_with_a_doubtful_card_reading_does_not_stop_the_climb_but_is_marked(tmp_path):
    def plan(rp, ctx, think):
        return [dict(doubtful=True)] * 3
    s = Scripted(plan)
    rows, levels = ladder.run_ladder(3, s, say=lambda t: None)
    assert len(_levels(s)) == 8 and all(lv["doubtful"] for lv in levels)


def test_c6_each_try_prints_one_progress_line_and_the_estimate_is_said_first(tmp_path):
    said = []
    ladder.run_ladder(2, Scripted(lambda rp, ctx, think: [dict()] * 2), say=said.append)
    tries = [t for t in said if t.startswith("try ")]
    assert len(tries) == 16 and "reply limit 512" in tries[0] and "1 of 2" in tries[0]
    text = ladder.estimate_text(tries=3, limit_s=600)
    assert "8 levels" in text and "24 tries" in text and "at most" in text and "stops early" in text


def test_c6_the_think_off_level_uses_the_best_level_of_both_climbs(tmp_path):
    def plan(rp, ctx, think):
        fast = ctx == 32768
        return [dict(seconds=10 if fast else 80)] * 3
    s = Scripted(plan)
    ladder.run_ladder(3, s, say=lambda t: None)
    assert s.calls[-1][:3] == (512, 32768, "off")      # all reply limits tie, so 512 stays; 32768 is the fast context



def test_c6_a_level_that_did_not_fit_is_never_the_best_one(tmp_path):
    # reviewer (C6): 65536 does not fit but reaches 3 of 3 (slowly); the fitting contexts reach 2 of 3.
    # The think-off level must still be one that fits.
    def plan(rp, ctx, think):
        spill = ctx == 65536
        return [dict(fits="no" if spill else "yes", seconds=500 if spill else 60,
                     tools_run=1 if (spill or n < 2) else 0, answered="no") for n in range(3)]
    s = Scripted(plan)
    ladder.run_ladder(3, s, say=lambda t: None)
    assert (512, 65536, "default") in _levels(s)          # the spilled level was still tried and written down
    assert s.calls[-1][:3] != (512, 65536, "off")
    assert s.calls[-1][1] != 65536 and s.calls[-1][2] == "off"

# --- 5. the output files ---------------------------------------------------------------------------------------------


def test_c6_the_csv_and_md_have_one_row_per_try_and_per_level_counts(tmp_path):
    rows, levels = ladder.run_ladder(3, Scripted(_ok), say=lambda t: None)
    csv_path, md_path = ladder.write_outputs(rows, levels, tmp_path, "qwen3:8b", "20261105T120000")
    assert csv_path.name == "ladder_qwen3_8b_20261105T120000.csv" and md_path.suffix == ".md"
    got = list(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8"))))
    assert len(got) == len(rows) and list(got[0]) == ladder.COLUMNS
    md = md_path.read_text(encoding="utf-8")
    assert "what each column means" in md.lower() and "reached a tool call or answer" in md
    assert "3 of 3" in md and "median" in md.lower()


def test_c6_a_reading_that_cannot_be_trusted_is_said_in_the_table_header(tmp_path):
    rows, levels = ladder.run_ladder(1, Scripted(lambda rp, ctx, think: [dict()]), say=lambda t: None)
    _csv, md_path = ladder.write_outputs(rows, levels, tmp_path, "m", "t")
    md = md_path.read_text(encoding="utf-8")
    assert "doubtful" in md and "/api/ps" in md


def test_c6_the_fake_key_never_reaches_an_output_file(tmp_path):
    rows, levels = ladder.run_ladder(1, Scripted(lambda rp, ctx, think: [dict()]), say=lambda t: None)
    csv_path, md_path = ladder.write_outputs(rows, levels, tmp_path, "m", "t")
    for p in (csv_path, md_path):
        assert ladder.FAKE_KEY not in p.read_text(encoding="utf-8")
    assert "fake key" in md_path.read_text(encoding="utf-8").lower()
    bad = [dict(rows[0], ended_why=f"oops {ladder.FAKE_KEY}")]
    with pytest.raises(RuntimeError):
        ladder.write_outputs(bad, levels, tmp_path / "out2", "m", "t")
    assert not (tmp_path / "out2").exists() or not list((tmp_path / "out2").iterdir())


def test_c6_a_leak_in_a_record_is_reported_loudly_in_the_table(tmp_path):
    rows, levels = ladder.run_ladder(1, Scripted(lambda rp, ctx, think: [dict(key_in_record=True)]), say=lambda t: None)
    _csv, md_path = ladder.write_outputs(rows, levels, tmp_path, "m", "t")
    assert "KEY FOUND IN A SESSION RECORD" in md_path.read_text(encoding="utf-8")


# --- 6. the time limit -----------------------------------------------------------------------------------------------


def test_c6_a_child_that_runs_past_the_limit_is_stopped_and_said_plainly():
    t0 = time.monotonic()
    rc, timed_out = ladder.run_child([sys.executable, "-c", "import time; time.sleep(60)"], 1.0)
    assert timed_out is True and time.monotonic() - t0 < 15


def test_c6_a_child_that_finishes_gives_its_exit_code():
    rc, timed_out = ladder.run_child([sys.executable, "-c", "raise SystemExit(3)"], 30.0)
    assert (rc, timed_out) == (3, False)


def test_c6_the_command_of_a_try_carries_the_settings_and_never_think_when_default():
    cmd = ladder.child_command("m:1", "clean up this folder", 1024, 16384, "default")
    assert "--think" not in cmd and cmd[cmd.index("--reply-limit") + 1] == "1024"
    assert cmd[cmd.index("--context") + 1] == "16384" and "--one-try" in cmd
    assert ladder.child_command("m:1", "t", 512, 8192, "off")[-2:] == ["--think", "off"]


def test_c6_each_try_starts_from_a_fresh_pack_and_finds_its_own_record(tmp_path, monkeypatch):
    bot = tmp_path / ".localbot" / "bots" / "ladder"
    work, sessions = bot / "work", bot / "sessions"
    sessions.mkdir(parents=True)
    work.mkdir()
    (work / "left_over.txt").write_text("from the last try", encoding="utf-8")
    (work / "notes.txt").write_text("changed by the last try", encoding="utf-8")
    seen = {}

    def fake_child(cmd, limit_s):
        seen["files"] = _snapshot(work)
        seen["cmd"] = cmd
        rec = sessions / "20261105T000000Z_m.jsonl"
        rec.write_text(json.dumps({"kind": "header", "reply_limit": 1024, "context": 8192, "steps": 8,
                                   "think": "default"}) + "\n", encoding="utf-8")
        return 0, False
    monkeypatch.setattr(ladder, "run_child", fake_child)
    run_try = ladder.make_process_run_try("m", "clean up this folder", 600, home=tmp_path)
    row = run_try(1024, 8192, "default", 1)
    assert seen["files"] == _snapshot_of_pack(tmp_path)               # old file gone, notes.txt back to the pack
    assert "--one-try" in seen["cmd"] and row["reply_limit"] == 1024 and row["ended_why"].startswith("no summary")
    # a second try must not pick up the first try's record
    monkeypatch.setattr(ladder, "run_child", lambda cmd, limit_s: (0, False))
    assert run_try(1024, 8192, "default", 2)["ended_why"].startswith("no record")

