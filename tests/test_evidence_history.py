"""BRIEF_004 P6: ``scripts/evidence_history.py`` recomputes the evidence page's history totals from
``docs/runs.json`` and the committed result files. These tests pin the published numbers; if one does not
come out, the counting rule is NOT changed to fit it (stop and report)."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "evidence_history.py"


@pytest.fixture(scope="module")
def eh():
    spec = importlib.util.spec_from_file_location("evidence_history_t", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def summary(eh):
    return eh.summarize(REPO)


def test_the_history_totals_are_the_published_ones(summary):
    assert summary["files"] == 222 and summary["tries"] == 6228
    assert round(summary["hours"], 1) == 29.2
    assert summary["models"] == 24 and summary["void"] == 6
    assert summary["ollama"] == {"0.40.12": 51, "0.40.13": 73, "0.40.14": 71, None: 27}


def test_the_final_run_61fb534(summary):
    f = summary["final"]
    assert f["code"] == "61fb534" and f["files"] == 71 and f["tries"] == 2016
    assert f["minutes"] == 620 and f["minutes_raw"] != 620 and 619 < f["minutes_raw"] < 620
    assert f["lanes"] == {"gpu": 64, "spill": 5, "unknown": 2}


def test_the_rounds_add_up_to_the_totals(summary):
    rows = summary["rounds"]
    assert sum(r["files"] for r in rows) == 222 and sum(r["tries"] for r in rows) == 6228
    assert round(sum(r["hours"] for r in rows), 1) == 29.2
    assert [r["code"] for r in rows][-1] == "61fb534"


def test_the_markdown_prints_the_numbers(eh, summary):
    md = eh.render(summary)
    for text in ("**222**", "**6,228**", "**29.2 hours**", "**24** models", "**6**", "0.40.12 (51 files)",
                 "0.40.13 (73", "0.40.14 (71", "not recorded (27", "61fb534", "71 files", "2,016 tries",
                 "10 h 20 min", "gpu 64", "spill 5", "unknown 2"):
        assert text in md, text
    assert ("pao" + "lo") not in md.lower() and "\r" not in md  # spelled in pieces: this file is exported too


def test_it_is_read_only_stdlib_only_and_takes_a_root(tmp_path):
    src = SCRIPT.read_text(encoding="utf-8")
    assert "\r\n" not in src
    for banned in ('"w"', "'w'", "write_text", "write_bytes", "unlink", "rmtree", "subprocess", "import requests"):
        assert banned not in src, banned
    r = subprocess.run([sys.executable, "-I", str(SCRIPT), str(REPO)], capture_output=True, text=True,
                       encoding="utf-8", cwd=tmp_path)
    assert r.returncode == 0 and "**222**" in r.stdout, r.stderr[-300:]
    r2 = subprocess.run([sys.executable, "-I", str(SCRIPT)], capture_output=True, text=True, encoding="utf-8",
                        cwd=tmp_path)
    assert r2.returncode == 0 and r2.stdout == r.stdout  # no argument = the checkout it sits in


def test_a_missing_result_file_stops_with_exit_2_and_a_message(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "results" / "stage1").mkdir(parents=True)
    (tmp_path / "docs" / "runs.json").write_text(
        json.dumps({"runs": [{"run": "r", "gate": "G3", "code": "x", "files": ["nope.json"]}], "void": []}),
        encoding="utf-8")
    r = subprocess.run([sys.executable, "-I", str(SCRIPT), str(tmp_path)], capture_output=True, text=True,
                       encoding="utf-8")
    assert r.returncode == 2 and "nope.json" in r.stderr and r.stdout == ""