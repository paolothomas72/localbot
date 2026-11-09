"""BRIEF C17c: the batch summary.md table survives a | or a line break inside a cell.

The quick_check cell of a model that never loaded holds the start of the provider's error, and a template error can
quote template text, which uses |. The CSV is quoted by the csv writer; the markdown table needs the same cell rule
as the ladder file (chat_ladder._md_cell)."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import scripts.model_queue as mq

QUICK = 'load failed: HTTP Error 500: {"error":"... {{ messages | selectattr(\'x\') }} ..."'
TRIAGE = "triage: model file or Ollama (did not load)\nsecond   line | with a pipe"


def _pipes(line: str) -> int:
    return len(re.findall(r"(?<!\\)\|", line))


def _write(tmp_path: Path, row: dict):
    mq.write_summary(tmp_path, [row], {"ollama_version": "v", "commit": "c"})
    md = (tmp_path / "summary.md").read_text(encoding="utf-8")
    with (tmp_path / "summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return md, rows


def _table_lines(md: str) -> list[str]:
    return [ln for ln in md.splitlines() if ln.startswith("|")]


def _bad_row() -> dict:
    return {"model": "m:1", "kind": "base", "status": "done", "digest": "d", "quick_check": QUICK, "triage": TRIAGE,
            "packs": {"scores": {}, "first_try": {}, "hard": {}}, "ladder": {}}


def test_1_every_table_line_has_the_same_number_of_unescaped_pipes_as_the_header(tmp_path):
    md, _rows = _write(tmp_path, _bad_row())
    lines = _table_lines(md)
    header = next(ln for ln in lines if ln.startswith("| model "))
    assert len(lines) == 3  # header, ---, one data row: the line break did not split the row
    for ln in lines:
        assert _pipes(ln) == _pipes(header), ln
    assert not any(ln.startswith("second") for ln in md.splitlines())  # nothing leaked onto a line of its own


def test_1_the_data_row_holds_the_escaped_form(tmp_path):
    md, _rows = _write(tmp_path, _bad_row())
    data = _table_lines(md)[-1]
    assert "messages \\| selectattr" in data
    assert "second line \\| with a pipe" in data  # the line break and the run of spaces became one space


def test_1_the_csv_keeps_the_exact_text(tmp_path):
    _md, rows = _write(tmp_path, _bad_row())
    assert rows[0]["quick_check"] == QUICK
    assert rows[0]["triage"] == TRIAGE


def test_2_a_normal_row_reads_the_same_in_markdown_and_csv(tmp_path):
    row = {"model": "m:1", "kind": "base", "status": "done", "digest": "d", "quick_check": "native tool call",
           "triage": "",
           "packs": {"scores": {"work": "3/3"}, "first_try": {"work": "1/1"}, "hard": {"outside_box": 2}},
           "ladder": {}}
    md, rows = _write(tmp_path, row)
    data = _table_lines(md)[-1]
    cells = [c.strip() for c in data.strip("|").split(" | ")]
    assert len(cells) == len(mq.SUMMARY_COLUMNS)
    for col, cell in zip(mq.SUMMARY_COLUMNS, cells):
        assert cell == rows[0][col], col


def test_2_existing_columns_are_built_as_before(tmp_path):
    packs = {"scores": {"work": "3/3"}, "first_try": {"work": "1/1"}, "hard": {"outside_box": 2}}
    row = {"model": "m:1", "kind": "base", "status": "done", "digest": "d", "packs": packs, "ladder": {}}
    _md, rows = _write(tmp_path, row)
    assert rows[0]["full_work"] == "3/3" and rows[0]["first_try_work"] == "1/1" and rows[0]["hard_outside_box"] == "2"
