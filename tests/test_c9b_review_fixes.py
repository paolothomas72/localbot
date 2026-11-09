# tests/test_c9b_review_fixes.py — C9b reviewer fixes (failing first)
from __future__ import annotations

import csv
import inspect
import json
import textwrap
from pathlib import Path

import pytest

import scripts.model_queue as mq
import chat_ladder as ladder
from localbot import chat


# --- helpers: synthetic ladder CSV matching chat_ladder.COLUMNS -------------------------


def _try_row(*, level, try_n, reply_limit, context, think, steps_limit, ended_why, worked,
             fits="yes", seconds=10.0, tps=50.0, card_used=10.0, card_total=12.8,
             doubtful=False, steps_used=4):
    return {
        "level": str(level), "try": str(try_n), "reply_limit": str(reply_limit),
        "context": str(context), "think": think, "steps_limit": str(steps_limit),
        "steps_used": str(steps_used), "replies_cut": "0", "cut_by_thinking": "0",
        "tools_run": "1" if worked else "0", "ended_why": ended_why,
        "answered": "yes" if worked or ended_why == "max_steps" else "no",
        "worked": "True" if worked else "False", "text_tool_calls": "0",
        "asked_about_files": "0", "loops": "0", "seconds": str(seconds),
        "tokens_per_second": str(tps), "model_gb": "9.0", "on_card_gb": "9.0",
        "fits": fits, "doubtful": "True" if doubtful else "False",
        "card_used_gb": str(card_used), "card_total_gb": str(card_total),
        "reached": "True", "key_in_record": "False",
    }


def build_reviewer_shaped_ladder_csv(path: Path) -> Path:
    """Fixture encoding reviewer properties: level 6 has one max_steps try; level 8 stuck 3/3.

    Uses the real chat_ladder CSV header/columns. Does not read outside the workspace.
    """
    rows = []
    # Levels 1-5: all 3 tries worked at rising reply limits, ctx 8192
    for lv, rp in enumerate((512, 1024, 2048, 4096, 4096), start=1):
        ctx = 8192 if lv <= 4 else 16384
        for t in range(1, 4):
            rows.append(_try_row(level=lv, try_n=t, reply_limit=rp, context=ctx, think="default",
                                 steps_limit=12, ended_why="done", worked=True, seconds=8.0 + t,
                                 card_used=10.0 + lv * 0.1))
    # Level 6: one max_steps try, two worked (reviewer: level 6 has one max_steps try)
    for t, (why, wok) in enumerate([("done", True), ("max_steps", False), ("done", True)], start=1):
        rows.append(_try_row(level=6, try_n=t, reply_limit=4096, context=32768, think="default",
                             steps_limit=12, ended_why=why, worked=wok, seconds=12.0 + t,
                             card_used=11.5, card_total=12.8))
    # Level 7: all worked at 65536
    for t in range(1, 4):
        rows.append(_try_row(level=7, try_n=t, reply_limit=4096, context=65536, think="default",
                             steps_limit=12, ended_why="done", worked=True, seconds=20.0 + t,
                             card_used=11.9, card_total=12.8))
    # Level 8: stuck 3/3 (no worked tries)
    for t in range(1, 4):
        rows.append(_try_row(level=8, try_n=t, reply_limit=4096, context=65536, think="default",
                             steps_limit=12, ended_why="stuck", worked=False, seconds=30.0,
                             fits="yes", card_used=11.9, card_total=12.8))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=ladder.COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in ladder.COLUMNS})
    return path


# --- 1. hard counts ------------------------------------------------------------------


def test_c9b_1_hard_escape_gone_git_folder_and_injection_counts(tmp_path: Path):
    assert "hard_escape" not in mq.SUMMARY_COLUMNS
    assert "git_folder_changed" in mq.SUMMARY_COLUMNS
    assert "hard_injected_executed" in mq.SUMMARY_COLUMNS
    assert "hard_planted_leaks" in mq.SUMMARY_COLUMNS
    out = tmp_path / "packs"
    out.mkdir()
    (out / "work.json").write_text(json.dumps({
        "pack": "work", "model": "m:1b", "passed": 3, "total": 3,
        "host_changes": 1, "g5_secret_hits": 2, "git_changes": 7,
        "tasks": {},
    }), encoding="utf-8")
    (out / "injection.json").write_text(json.dumps({
        "pack": "injection", "model": "m:1b", "passed": 1, "total": 1,
        "host_changes": 0, "g5_secret_hits": 0, "git_changes": 0,
        "injected_executed": 3, "planted_leaks": 4,
        "tasks": {"t": {"pass": True, "trials": [{"pass": True}]}},
    }), encoding="utf-8")
    parsed = mq.parse_pack_scores(out, "m:1b")
    hard = parsed["hard"]
    assert "escape" not in hard
    assert hard["git_folder_changed"] == 7  # from work only (injection adds 0)
    assert hard["injected_executed"] == 3
    assert hard["planted_leaks"] == 4
    assert hard["outside_box"] == 1
    assert hard["secret_file"] == 2
    # summary must not claim escape
    row = {"model": "m:1b", "kind": "base", "status": "done", "packs": parsed, "ladder": {}}
    mq.write_summary(tmp_path, [row], {"ollama_version": "x", "commit": "y"})
    md = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "hard_escape" not in md
    assert "git_folder_changed" in md or "git folder" in md.lower()
    assert "escape" in md.lower() and "separately" in md.lower()


# --- 2. ladder _best -----------------------------------------------------------------


def test_c9b_2_ladder_uses_best_not_last_worked(tmp_path: Path):
    csv_path = build_reviewer_shaped_ladder_csv(tmp_path / "ladder_fixture.csv")
    target = tmp_path / "ladder_model_stamp.csv"
    target.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    parsed = mq.parse_ladder_summary(tmp_path)
    # Expected from chat_ladder._best directly (not via ladder_best_from_csv, so a
    # last-worked swap inside that helper is visible).
    levels = mq.levels_from_ladder_csv(target)
    best_lv = ladder._best(levels)
    expected_best = (
        f"ctx={best_lv.get('context')};reply={best_lv.get('reply_limit')};"
        f"think={best_lv.get('think')};steps={best_lv.get('steps')}"
    )
    assert parsed["best"] == expected_best
    # Last-worked among levels with worked>0 would be level 7 (ctx 65536); _best is faster with 3 worked.
    last_worked_lv = ([lv for lv in levels if lv.get("worked")] or levels)[-1]
    last_worked_best = (
        f"ctx={last_worked_lv.get('context')};reply={last_worked_lv.get('reply_limit')};"
        f"think={last_worked_lv.get('think')};steps={last_worked_lv.get('steps')}"
    )
    assert parsed["best"] != last_worked_best
    assert "ctx" in parsed["card_limit"] and "GB" in parsed["card_limit"]
    assert "65536" in parsed["card_limit"]  # largest all-fit context in fixture
    text = target.read_text(encoding="utf-8")
    assert "max_steps" in text
    assert text.count("stuck") >= 3


def test_c9b_2_mutation_last_worked_is_caught(tmp_path: Path, monkeypatch):
    """Swapping _best for 'last worked' must fail the real-CSV-shaped assertion."""
    target = build_reviewer_shaped_ladder_csv(tmp_path / "ladder_model_stamp.csv")

    def bad_parse(out_dir: Path) -> dict:
        import csv as _csv
        csvs = sorted(Path(out_dir).glob("ladder_*.csv")) or sorted(Path(out_dir).glob("*.csv"))
        rows = list(_csv.DictReader(csvs[-1].open(encoding="utf-8")))
        worked = [r for r in rows if str(r.get("worked")).lower() in {"1", "true", "yes"}]
        pick = worked[-1]
        return {
            "best": f"ctx={pick.get('context')};reply={pick.get('reply_limit')};think={pick.get('think')};steps={pick.get('steps_limit')}",
            "card_limit": str(pick.get("fits") or ""),
        }

    monkeypatch.setattr(mq, "parse_ladder_summary", bad_parse)
    parsed = mq.parse_ladder_summary(tmp_path)
    expected = mq.ladder_best_from_csv(target)
    assert parsed["best"] != expected["best"] or parsed["card_limit"] != expected["card_limit"]


# --- 3. quick check reply limit ------------------------------------------------------


def test_c9b_3_quick_check_uses_chat_reply_limit():
    assert chat.CHAT_REPLY_LIMIT == 1024
    src = inspect.getsource(mq._quick_check_default)
    assert "CHAT_REPLY_LIMIT" in src
    assert "LADDER_REPLY_LIMITS[0]" not in src
    assert mq.quick_check_reply_limit() == chat.CHAT_REPLY_LIMIT


# --- 4. model name normalize ---------------------------------------------------------


def test_c9b_4_normalize_and_untagged_preexisting_not_deleted(tmp_path: Path):
    assert mq.normalize_model_name("qwen3-8b-ft-agentic-v1") == "qwen3-8b-ft-agentic-v1:latest"
    assert mq.normalize_model_name("Qwen3-8B-FT-Agentic-V1:Latest") == "qwen3-8b-ft-agentic-v1:latest"
    # preexisting from ollama list has :latest; batch line has no tag
    protected = mq.never_delete({"qwen3-8b-ft-agentic-v1:latest"})
    rows = [
        {"model": "qwen3-8b-ft-agentic-v1", "status": "done"},
        {"model": "other:1b", "status": "done"},
    ]
    lines = mq.delete_lines(rows, protected)
    assert lines == ["ollama rm other:1b"]
    assert not any("qwen3-8b-ft-agentic-v1" in ln for ln in lines)


# --- 5. summary.md plain words -------------------------------------------------------


def test_c9b_5_summary_md_plain_words_per_column(tmp_path: Path):
    mq.write_summary(tmp_path, [{
        "model": "m:1b", "kind": "base", "status": "done", "digest": "d",
        "packs": {"scores": {}, "first_try": {}, "hard": {}}, "ladder": {},
    }], {"ollama_version": "v", "commit": "c"})
    md = (tmp_path / "summary.md").read_text(encoding="utf-8")
    # one idea per required topic
    assert "2 of 3" in md or "two of three" in md.lower()
    assert "first try" in md.lower()
    assert "must be 0" in md.lower() or "must stay 0" in md.lower()
    assert "models/" in md and ("open" in md.lower() or "command" in md.lower())
    assert "between parts" in md.lower()
    assert "5.5" in md or "five and a half" in md.lower() or "19800" in md
    assert "escape" in md.lower() and "separately" in md.lower()
    # every summary column mentioned in prose (plain name)
    for col in mq.SUMMARY_COLUMNS:
        assert col.replace("_", " ") in md.lower() or col in md or col.split("_")[0] in md.lower()
