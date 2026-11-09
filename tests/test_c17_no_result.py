"""BRIEF C17: a try that never ran is "no result" (never a pass), "passed without acting" is counted on its
own line, the quick check says "load failed", and the ladder stops when nothing loads.

No Ollama, no GPU, no container: scripted providers behind the real OllamaProvider, made-up result files and
made-up ladder rows."""
from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

import scripts.chat_ladder as ladder
import scripts.model_queue as mq
from localbot.config import get_profile
from localbot.exam.runner import load_pack, run_pack
from tests.fakes import ReactiveTransport, RoutedTransport, ScriptedTransport, call, provider, reply
from tests.test_runner import fake_sampler

P6 = get_profile("parity_lab6")
ERR500 = "request failed: HTTP Error 500: Internal Server Error"


def _http500():
    return urllib.error.HTTPError("http://127.0.0.1:11434/api/chat", 500, "Internal Server Error", {}, None)


def _go(disk, pack, tasks, chat, n=1):
    rt = RoutedTransport(ReactiveTransport(chat) if callable(chat) else ScriptedTransport(chat))
    lines: list[str] = []
    res, dest = run_pack(pack, P6, "fake:1", provider(rt, P6.provider), n=n, tasks=tasks,
                         vram_reader=lambda: None, sampler_factory=fake_sampler,
                         results_dir=disk.base / "results", trials_root=disk.base / "_trials", log=lines.append)
    return res, dest, lines


def _always_500(_messages):
    raise _http500()


# --- A. no result (runner) ---------------------------------------------------------------------------------------


def test_1_always_500_privacy_pack_is_no_result_not_a_pass(disk):
    tasks = load_pack("privacy")
    res, dest, lines = _go(disk, "privacy", tasks, _always_500)
    assert res["passed"] == 0 and res["total"] == len(tasks)
    assert res["no_result_tasks"] == res["total"] and res["scored_total"] == 0
    for t in res["tasks"].values():
        tr = t["trials"][0]
        assert t["no_result"] is True and t["pass"] is False
        assert tr["no_result"] is True and tr["pass"] is False
        assert tr["lab_judge_pass"] is False and tr["core4_judge_pass"] is False
    assert res["passed_lab_judge"] == 0
    assert any("no_result=" + str(len(tasks)) in line for line in lines)
    assert mq.parse_pack_scores(dest.parent, "fake:1")["scores"]["privacy"] == "no result"


def test_2_fails_once_then_answers_is_scored_as_today(disk):
    t = next(x for x in load_pack("work") if x["id"] == "e_calc")
    good = [reply(calls=[call("write", path="ans.txt", content="21")]), reply("21")]
    res, _, _ = _go(disk, "work", [t], [_http500(), _http500()] + good)
    tr = res["tasks"]["e_calc"]["trials"][0]
    assert tr["pass"] and tr["tag"] == "r" and tr["no_result"] is False  # the re-run is what is scored
    assert tr["infra_attempt"]["no_result"] is False  # the first attempt was replaced, not counted
    assert res["tasks"]["e_calc"]["no_result"] is False
    assert res["no_result_tasks"] == 0 and res["scored_total"] == res["total"] == 1 and res["passed"] == 1


def test_2b_most_tries_no_result_makes_the_task_no_result(disk):
    t = next(x for x in load_pack("work") if x["id"] == "e_calc")
    # 3 tries: t1 good, t2 and t3 are 500 after the re-run too
    good = [reply(calls=[call("write", path="ans.txt", content="21")]), reply("21")]
    script = good + [_http500()] * 8
    res, _, _ = _go(disk, "work", [t], script, n=3)
    task = res["tasks"]["e_calc"]
    assert [x["no_result"] for x in task["trials"]] == [False, True, True]
    assert task["no_result"] is True and task["pass"] is False
    assert res["no_result_tasks"] == 1 and res["scored_total"] == 0 and res["passed"] == 0


# --- C. passed without acting (runner) ---------------------------------------------------------------------------


def test_3_no_tool_calls_on_a_privacy_task_still_passes_and_is_counted(disk):
    t = next(x for x in load_pack("privacy") if x["id"] == "p_escape_write")
    res, _, _ = _go(disk, "privacy", [t], lambda m: reply("I will not do that."))
    task = res["tasks"]["p_escape_write"]
    tr = task["trials"][0]
    assert tr["pass"] is True and tr["acted"] is False and tr["no_result"] is False  # the judging is unchanged
    assert task["passed_without_acting"] == 1 and res["passed_without_acting"] == 1
    assert res["passed"] == 1 and res["no_result_tasks"] == 0


def test_3b_a_pass_that_acted_is_not_counted_as_without_acting(disk):
    t = next(x for x in load_pack("work") if x["id"] == "e_calc")
    good = [reply(calls=[call("write", path="ans.txt", content="21")]), reply("21")]
    res, _, _ = _go(disk, "work", [t], good)
    tr = res["tasks"]["e_calc"]["trials"][0]
    assert tr["pass"] and tr["acted"] is True
    assert res["tasks"]["e_calc"]["passed_without_acting"] == 0 and res["passed_without_acting"] == 0


# --- C/B. the queue reads the result files -----------------------------------------------------------------------


def _write_result(out: Path, pack: str, tasks: dict, model: str = "m:1") -> None:
    passed = sum(1 for t in tasks.values() if t["pass"])
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{pack}_x.json").write_text(json.dumps({"pack": pack, "model": model, "passed": passed,
                                                     "total": len(tasks), "tasks": tasks}), encoding="utf-8")


def _trial(passed=True, calls=1, infra=False, **extra):
    return {"pass": passed, "infra": infra, "calls": [{"name": "read"}] * calls, **extra}


def test_queue_score_counts_scored_tasks_only_and_reports_the_rest(tmp_path):
    tasks = {
        "a": {"pass": True, "trials": [_trial()]},
        "b": {"pass": True, "trials": [_trial(calls=0)], "passed_without_acting": 1},
        "c": {"pass": False, "no_result": True, "trials": [_trial(False, 0, no_result=True)]},
        "d": {"pass": False, "trials": [_trial(False)]},
    }
    _write_result(tmp_path, "privacy", tasks)
    got = mq.parse_pack_scores(tmp_path, "m:1")
    assert got["scores"]["privacy"] == "2/3"
    assert got["no_result_tasks"] == {"privacy": 1}
    assert got["passed_without_acting"] == {"privacy": 1}


def test_queue_old_file_apriel_style_reads_as_no_result_normal_file_unchanged(tmp_path):
    """Made-up old-format files (no new fields): infra tries kept after the re-run, all passing on an empty folder."""
    apriel = {f"t{i}": {"pass": True, "trials": [_trial(True, 0, infra=True)] * 3} for i in range(8)}
    normal = {f"t{i}": {"pass": i < 6, "trials": [_trial(i < 6, 2)] * 3} for i in range(8)}
    a, n = tmp_path / "a", tmp_path / "n"
    _write_result(a, "privacy", apriel, "apriel")
    _write_result(n, "privacy", normal, "norm")
    got_a, got_n = mq.parse_pack_scores(a, "apriel"), mq.parse_pack_scores(n, "norm")
    assert got_a["scores"]["privacy"] == "no result"
    assert got_n["scores"]["privacy"] == "6/8" and "privacy" not in got_n.get("no_result_tasks", {})
    assert got_n["passed_without_acting"] == {"privacy": 0}


def _fake_run_try(row):
    return lambda model, task, limit_s, home=None: (lambda *a, **k: dict(row))


BASE_ROW = {"tools_run": 0, "text_tool_calls": 0, "seconds": 3, "worked": False, "answered": "no",
            "key_in_record": False}


def test_4_quick_check_says_load_failed_and_why(monkeypatch):
    err = ERR500 + " " + "x" * 300
    row = {**BASE_ROW, "ended_why": "provider_error", "error": err}
    monkeypatch.setattr(mq.ladder, "make_process_run_try", _fake_run_try(row))
    out = mq._quick_check_default("m:1")
    assert out["result"] == "load failed: " + err[:160]
    assert out["native"] is False and out["text_tool"] is False


@pytest.mark.parametrize("row,expect", [
    ({"ended_why": "done", "error": ""}, "no tool call"),
    ({"ended_why": "done", "tools_run": 2}, "native tool call"),
    ({"ended_why": "timeout (180 s limit)", "error": ""}, "error/timeout"),
    ({"ended_why": "no record (exit code 1)"}, "error/timeout"),
])
def test_4b_other_quick_check_results_are_unchanged(monkeypatch, row, expect):
    monkeypatch.setattr(mq.ladder, "make_process_run_try", _fake_run_try({**BASE_ROW, **row}))
    assert mq._quick_check_default("m:1")["result"] == expect


def test_4c_parse_record_keeps_the_provider_error(tmp_path):
    lines = [{"kind": "header", "bot": "ladder", "model": "m:1", "reply_limit": 512, "context": 8192, "steps": 12,
              "think": "default"},
             {"kind": "event", "event": {"event": "message_end", "end_reason": "provider_error", "steps": 1, "wall_s": 2}},
             {"kind": "summary", "conversation": 1, "summary": {"reason": "provider_error", "elapsed_s": 2.0,
                                                                  "error": ERR500, "provider": []}}]
    p = tmp_path / "r.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    row = ladder.parse_record(p)
    assert row["ended_why"] == "provider_error" and row["error"] == "HTTP Error 500: Internal Server Error"


# --- D. the ladder ------------------------------------------------------------------------------------------------


def _lrow(**kw):
    base = dict(reply_limit=512, context=8192, think="default", steps_limit=12, steps_used=3, replies_cut=0,
                cut_by_thinking=0, tools_run=2, ended_why="done", answered="yes", worked=True, text_tool_calls=0,
                asked_about_files=0, loops=0, seconds=30.0, tokens_per_second=100.0, model_gb=6.0, on_card_gb=6.0,
                fits="yes", doubtful=False, card_used_gb=None, card_total_gb=None, key_in_record=False)
    base.update(kw)
    return base


class Scripted:
    def __init__(self, plan):
        self.plan, self.calls = plan, []

    def __call__(self, rp, ctx, think, n, steps=12):
        self.calls.append((rp, ctx, think, steps, n))
        return _lrow(reply_limit=rp, context=ctx, think=think, steps_limit=steps, **self.plan(rp, ctx, think, steps, n))


FAIL = dict(ended_why="provider_error", error=(ERR500 + " " + "y" * 300)[:160], tools_run=0,
            answered="no", worked=False, fits="unknown", seconds=2.0, tokens_per_second=None)  # error cut as parse_record does


def test_6_always_500_stops_after_the_first_step(tmp_path):
    s = Scripted(lambda rp, ctx, think, steps, n: dict(FAIL))
    rows, levels = ladder.run_ladder(3, s, say=lambda t: None)
    assert len(levels) == 1 and len(s.calls) == 3  # one step, its three tries, nothing higher
    assert levels[0]["stop"] == "load_failed"
    assert levels[0]["load_error"] == FAIL["error"] and len(levels[0]["load_error"]) == 160
    csv_path, md_path = ladder.write_outputs(rows, levels, tmp_path, "m", "t")
    assert FAIL["error"] in md_path.read_text(encoding="utf-8")
    assert FAIL["error"] in csv_path.read_text(encoding="utf-8")


def test_7_fails_once_at_a_step_then_answers_the_ladder_climbs_on():
    s = Scripted(lambda rp, ctx, think, steps, n: dict(FAIL) if n == 1 else {})
    rows, levels = ladder.run_ladder(3, s, say=lambda t: None)
    assert len(levels) > 1 and all(lv["stop"] == "" for lv in levels)


def test_7b_a_failed_load_that_is_not_http_500_is_not_a_load_failure():
    slow = dict(FAIL, error="timed out after 180s")
    s = Scripted(lambda rp, ctx, think, steps, n: dict(slow))
    _rows, levels = ladder.run_ladder(3, s, say=lambda t: None)
    assert len(levels) > 1 and levels[0]["stop"] != "load_failed"
