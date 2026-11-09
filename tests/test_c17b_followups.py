"""BRIEF C17b: six small follow-ups on C17.

2  first-try score leaves out tasks that never ran      3  specific triage hint for a load failure
4  the error keeps its useful part                        5  markdown tables survive | and line breaks in the error
F  the batch summary shows no_result_tasks / passed_without_acting

No Ollama, no GPU, no container: made-up result files, made-up session records, made-up ladder rows."""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

import pytest

import scripts.chat_ladder as ladder
import scripts.model_queue as mq

LOAD_HINT = "triage: model file or Ollama (did not load)"
APRIEL_ERROR = ('request failed: HTTP Error 500: Internal Server Error: {"error":"llama-server process has '
                'terminated: exit status 1: error: Unable to generate parser for this template. ')


def _write_result(out: Path, pack: str, tasks: dict, model: str = "m:1") -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{pack}_x.json").write_text(json.dumps({
        "pack": pack, "model": model, "passed": sum(1 for t in tasks.values() if t["pass"]),
        "total": len(tasks), "tasks": tasks}), encoding="utf-8")


def _trial(passed=True, calls=1, infra=False, **extra):
    return {"pass": passed, "infra": infra, "calls": [{"name": "read"}] * calls, **extra}


# --- item 2: the first-try count skips tasks that never ran -------------------------------------------------------


def test_2_apriel_style_old_file_first_try_is_no_result(tmp_path):
    tasks = {f"t{i}": {"pass": True, "trials": [_trial(True, 0, infra=True)] * 3} for i in range(8)}
    _write_result(tmp_path, "privacy", tasks)
    got = mq.parse_pack_scores(tmp_path, "m:1")
    assert got["scores"]["privacy"] == "no result"
    assert got["first_try"]["privacy"] == "no result"


def test_2_old_file_one_task_no_result_is_left_out_of_both_sides(tmp_path):
    dead = {"pass": False, "trials": [_trial(True, 0), _trial(False, 0, infra=True), _trial(False, 0, infra=True)]}
    ok = {"pass": True, "trials": [_trial()] * 3}
    _write_result(tmp_path, "trust", {"a": ok, "b": ok, "c": dead})
    got = mq.parse_pack_scores(tmp_path, "m:1")
    assert got["first_try"]["trust"] == "2/2"
    assert got["scores"]["trust"] == "2/2"


def test_2_new_file_task_flag_counts_the_same_way(tmp_path):
    dead = {"pass": False, "no_result": True, "trials": [_trial(True, 0), _trial(False, 0), _trial(False, 0)]}
    ok = {"pass": True, "trials": [_trial()] * 3}
    _write_result(tmp_path, "trust", {"a": ok, "b": ok, "c": dead})
    assert mq.parse_pack_scores(tmp_path, "m:1")["first_try"]["trust"] == "2/2"


def test_2_normal_file_first_try_is_unchanged(tmp_path):
    tasks = {f"t{i}": {"pass": i < 6, "trials": [_trial(i < 6, 2)] * 3} for i in range(8)}
    _write_result(tmp_path, "privacy", tasks)
    got = mq.parse_pack_scores(tmp_path, "m:1")
    assert got["first_try"]["privacy"] == "6/8" and got["scores"]["privacy"] == "6/8"


# --- item 3: the triage hint for a load failure -----------------------------------------------------------------


@pytest.mark.parametrize("is_hf", [False, True])
def test_3_load_failed_gets_its_own_hint(is_hf):
    label, hint = mq.classify_quick("load failed: HTTP Error 500: x", is_hf=is_hf, tool_section=True)
    assert label == "load failed: HTTP Error 500: x" and hint == LOAD_HINT


def test_3_load_failed_without_a_tool_section_keeps_that_addition():
    _label, hint = mq.classify_quick("load failed: HTTP Error 500: x", is_hf=False, tool_section=False)
    assert hint.startswith(LOAD_HINT) and hint.endswith("no tool section in template")


@pytest.mark.parametrize("result", ["no tool call", "error/timeout"])
def test_3_other_results_keep_the_old_hint(result):
    assert mq.classify_quick(result, is_hf=False, tool_section=True)[1] == "triage: harness or model?"


def test_3_the_hf_text_tool_case_is_unchanged():
    assert mq.classify_quick("tool call as text", is_hf=True, tool_section=True)[1] == "harness or pull problem"


# --- item 4: the error keeps its useful part --------------------------------------------------------------------


def _record(tmp_path, error, name="rec.jsonl"):
    lines = [{"kind": "header", "bot": "ladder", "model": "m:1", "reply_limit": 512, "context": 8192, "steps": 12,
              "think": "default"},
             {"kind": "event", "event": {"event": "message_end", "end_reason": "provider_error", "steps": 1, "wall_s": 2}},
             {"kind": "summary", "conversation": 1, "summary": {"reason": "provider_error", "elapsed_s": 2.0,
                                                                  "error": error, "provider": []}}]
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    return p


def _apriel_error():
    return APRIEL_ERROR + "x" * 150 + "\nsecond   line " + "y" * 150  # about 386 chars, a line break, a run of spaces


def test_4_cleaned_error_keeps_the_template_phrase_and_stays_a_load_failure(tmp_path):
    row = ladder.parse_record(_record(tmp_path, _apriel_error()))
    err = row["error"]
    assert err.startswith("HTTP Error 500")
    assert "Unable to generate parser for this template" in err
    assert len(err) <= 160 and "\n" not in err and "  " not in err
    assert "request failed: " not in err and "Internal Server Error: " not in err
    assert ladder.is_load_failure(row) is True


def test_4_old_long_form_error_in_a_csv_row_still_counts_as_a_load_failure(tmp_path):
    long_form = _apriel_error()
    assert ladder.is_load_failure({"ended_why": "provider_error", "error": long_form}) is True
    rows = [dict(ladder.no_record_row(512, 8192, "default", 1, 2.0, 12), ended_why="provider_error", error=long_form,
                 level=1, **{"try": n}) for n in (1, 2, 3)]
    p = tmp_path / "ladder_old.csv"
    p.write_text(ladder._csv_text(rows), encoding="utf-8", newline="")
    levels = mq.levels_from_ladder_csv(p)
    assert levels[0]["stop"] == ladder.LOAD_FAILED


def test_4_an_error_without_the_boilerplate_is_cut_as_before(tmp_path):
    err = "HTTP Error 500: " + "z" * 300
    assert ladder.parse_record(_record(tmp_path, err))["error"] == err[:160]


def test_4_a_non_500_error_is_not_changed_apart_from_the_cut(tmp_path):
    err = "timed out after 180s"
    row = ladder.parse_record(_record(tmp_path, err))
    assert row["error"] == err and ladder.is_load_failure(row) is False


def test_4_quick_check_result_contains_the_template_phrase(tmp_path, monkeypatch):
    row = ladder.parse_record(_record(tmp_path, _apriel_error()))
    row.update(seconds=3, text_tool_calls=0)
    monkeypatch.setattr(mq.ladder, "make_process_run_try", lambda *a, **k: (lambda *x, **y: dict(row)))
    result = mq._quick_check_default("m:1")["result"]
    assert result.startswith("load failed: HTTP Error 500")
    assert "Unable to generate parser for this template" in result
    assert len(result) <= len("load failed: ") + ladder.LOAD_ERROR_CHARS


# --- item 5: the markdown tables survive | and a line break in the error ---------------------------------------


BAD = "HTTP Error 500: bad | cell\nsecond line here"


def _bad_ladder():
    def run_try(rp, ctx, think, n, steps=12):
        return dict(ladder.no_record_row(rp, ctx, think, 1, 2.0, steps), ended_why="provider_error", error=BAD)
    return ladder.run_ladder(3, run_try, say=lambda t: None)


def _pipes(line: str) -> int:
    return len(re.findall(r"(?<!\\)\|", line))


def test_5_markdown_tables_keep_their_columns(tmp_path):
    rows, levels = _bad_ladder()
    assert levels[0]["stop"] == ladder.LOAD_FAILED
    md = ladder._md_text(rows, levels, "m", "t")
    lines = md.splitlines()
    checked, header = 0, None
    for i, line in enumerate(lines):
        if line.startswith("|") and i + 1 < len(lines) and lines[i + 1].startswith("|---"):
            header = _pipes(line)
        elif line.startswith("|") and not line.startswith("|---"):
            assert _pipes(line) == header, line
            checked += 1
        elif not line.startswith("|"):
            header = None if line == "" else header
    assert checked >= 4  # the level row and the try rows were really looked at
    assert not any(line.startswith("second line here") for line in lines)  # no line break leaked into the table
    assert "bad \\| cell second line here" in md


def test_5_csv_keeps_the_exact_error(tmp_path):
    rows, levels = _bad_ladder()
    csv_path, _md = ladder.write_outputs(rows, levels, tmp_path, "m", "t")
    got = list(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8"))))
    assert got and all(r["error"] == BAD for r in got)


# --- item F: the batch summary shows the new numbers --------------------------------------------------------------


def _summary_row(packs):
    return {"model": "m:1", "kind": "base", "status": "done", "digest": "d", "packs": packs, "ladder": {}}


def _write(tmp_path, packs):
    mq.write_summary(tmp_path, [_summary_row(packs)], {"ollama_version": "v", "commit": "c"})
    got = list(csv.DictReader((tmp_path / "summary.csv").open(encoding="utf-8", newline="")))
    return got[0], (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_f_the_two_columns_are_last_and_have_a_meaning_line(tmp_path):
    assert mq.SUMMARY_COLUMNS[-3:] == ["no_result_tasks", "passed_without_acting", "long_note"]   # C19: long_note added
    _row, md = _write(tmp_path, {"scores": {}, "first_try": {}, "hard": {}})
    flat = " ".join(md.split())
    assert "- no_result_tasks:" in flat and "- passed_without_acting:" in flat
    meaning = flat.split("- passed_without_acting:")[1].split(" - ")[0]
    assert "tries" in meaning and "tasks" in meaning  # the unit is tries; the pack scores count tasks


def test_f_numbers_are_written_as_pack_number_pairs(tmp_path):
    packs = {"scores": {}, "first_try": {}, "hard": {}, "no_result_tasks": {"injection": 1},
             "passed_without_acting": {"trust": 6, "privacy": 24, "work": 0}}
    row, _md = _write(tmp_path, packs)
    assert row["no_result_tasks"] == "injection:1"
    assert row["passed_without_acting"] == "trust:6 privacy:24"


def test_f_all_zero_or_missing_writes_empty_cells(tmp_path):
    row, _md = _write(tmp_path, {"scores": {}, "first_try": {}, "hard": {}, "passed_without_acting": {"work": 0}})
    assert row["no_result_tasks"] == "" and row["passed_without_acting"] == ""


def test_f_existing_columns_are_built_as_before(tmp_path):
    packs = {"scores": {"work": "3/3"}, "first_try": {"work": "1/1"}, "hard": {"outside_box": 2}}
    row, _md = _write(tmp_path, packs)
    assert row["full_work"] == "3/3" and row["first_try_work"] == "1/1" and row["hard_outside_box"] == "2"
