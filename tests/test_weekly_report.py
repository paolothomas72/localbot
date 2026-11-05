"""X4b: the Monday report. Part 1 is plain words; part 2 is the evidence."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
JARGON = ("num_ctx", "majority", "judge", "core4", "parity", "n=3",
          "G1", "G2", "G3", "G4", "G5", "G6")


def _load():
    path = REPO / "scripts" / "weekly_report.py"
    spec = importlib.util.spec_from_file_location("weekly_report", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(tmp: Path, **extra) -> dict:
    result = tmp / "result.json"
    result.write_text("{}", encoding="utf-8", newline="\n")
    row = {
        "gate": "LONG", "model": "small-model", "pack": "long", "profile": "core4_product",
        "hardware": "12 GB, 8k context",
        "passed": 3, "total": 4, "wall_s": 12.5, "tok_s": 40.0,
        "lab_judge_passed": 3, "host_changes": 0, "g5_secret_hits": 0, "secret_seen": 0,
        "gpu_lane": "gpu", "timing_comparable": True, "context_peak": 800,
        "output_tokens": 100, "turns": 6, "vram_peak_mib": 4000, "time_per_task_s": 3.1,
        "end_reasons": {"done": 3, "max_steps": 1},
        "result_file": str(result),
        "status": "ok",
        "misses": [
            {"task": "l_one", "cause": "the file was empty", "label": "verified"},
            {"task": "l_two", "cause": "a note in the log", "label": "reported"},
        ],
    }
    row.update(extra)
    return {
        "commit": "c9c3f39", "ollama_version": "0.0-test", "gpu": "12 GB card",
        "generated": "2026-11-03", "runs": [] if extra.get("_empty") else [row],
        "not_tested": ["the host test suite", "the container attack pack"],
    }


def test_a_week_with_runs_links_every_number_and_lists_sha256(tmp_path):
    mod = _load()
    data = _run(tmp_path)
    text = mod.render(data)
    assert "## Evidence" in text
    plain, evidence = text.split("## Evidence", 1)
    assert "small-model" in plain and "12 GB, 8k context" in plain
    assert "safety held" in plain.lower() or "Safety held" in plain
    result = Path(data["runs"][0]["result_file"])
    assert result.name in text
    digest = hashlib.sha256(result.read_bytes()).hexdigest()
    assert digest in evidence
    for number in ("3", "4", "12.5", "40.0", "800", "100", "6", "4000", "3.1"):
        assert f"[{number}](" in text


def test_a_week_with_no_runs_does_not_copy_last_week(tmp_path):
    mod = _load()
    previous = "# Last week\n\n| old-model | kept going |\n"
    text = mod.render(_run(tmp_path, _empty=True), previous=previous)
    assert "no runs this week" in text.lower()
    assert "old-model" not in text


def test_verified_and_reported_misses_are_labelled_differently(tmp_path):
    mod = _load()
    evidence = mod.render(_run(tmp_path)).split("## Evidence", 1)[1]
    assert "**verified**" in evidence and "**reported**" in evidence
    assert evidence.index("**verified**") != evidence.index("**reported**")


def test_plain_part_does_not_use_gate_jargon(tmp_path):
    mod = _load()
    plain = mod.render(_run(tmp_path)).split("## Evidence", 1)[0]
    for word in JARGON:
        assert word not in plain, word


def test_template_has_both_parts():
    template = REPO / "docs" / "reports" / "WEEKLY_TEMPLATE.md"
    if not template.exists():
        pytest.skip("needs a private file that is not in the public copy: docs/reports/WEEKLY_TEMPLATE.md")
    text = template.read_text(encoding="utf-8")
    assert "Part 1" in text and "Part 2" in text
