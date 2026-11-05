"""H1 E-1: scripts/suspicious_tries.py takes many paths / globs and refuses to pass on nothing.

Before H1 ``main()`` read only ``argv[0]``, as one glob. An unquoted glob that the shell expanded
into a file list scanned only the first file, and an empty glob scanned 0 files; both printed
``VERDICT: no false success`` and exited 0. Now every argument is a path or a glob, the header names
each argument with its match count (an argument that matched nothing gets its own WARNING line),
duplicates count once, and 0 result-shaped files scanned is ``VERDICT: NOTHING SCANNED``, exit 2.

Synthetic result files go in tmp_path (values are made up; nothing is secret). The last test reads
the committed results/stage1 read-only. No model, no network.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "suspicious_tries.py"
RES = REPO / "results" / "stage1"
NEMOTRON_WORK = "work_parity_lab6_nemotron-3-nano_4b_20261030T"


@pytest.fixture(scope="module")
def st():
    spec = importlib.util.spec_from_file_location("suspicious_tries_h1", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _result(path: Path, *, false_success: bool = False) -> Path:
    """One result-shaped file: one task, one passing try, one call. ``false_success`` makes the
    call a ``write`` with a wrong-case key whose reply still says "Wrote" (the S8f.1a lab bug)."""
    if false_success:
        call = {"name": "write", "args": {"Path": "a.txt", "content": "x"}, "result": "Wrote 1 bytes"}
    else:
        call = {"name": "read", "args": {"path": "a.txt"}, "result": "x"}
    d = {"model": "m-synthetic", "profile": "core4",
         "tasks": {"t1": {"trials": [{"k": 1, "pass": True, "end_reason": "done", "calls": [call]}]}}}
    path.write_text(json.dumps(d), encoding="utf-8")
    return path


def _main(st, capsys, *argv) -> tuple[int, str]:
    code = st.main([str(a) for a in argv])
    return code, capsys.readouterr().out


def _cli(cwd: Path, *argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(SCRIPT), *argv], capture_output=True, text=True,
                          encoding="utf-8", cwd=cwd, timeout=120)


def _scanned(out: str) -> tuple[int, int]:
    m = re.search(r"^suspicious_tries: scanned (\d+) files, (\d+) tries", out, re.M)
    assert m, out
    return int(m.group(1)), int(m.group(2))


def _verdict(out: str) -> str:
    lines = [ln for ln in out.splitlines() if ln.startswith("VERDICT: ")]
    assert len(lines) == 1, out
    return lines[0]


def _matched(out: str) -> dict[str, int]:
    """Header lines ``    <n> file(s)  '<argument>'`` -> {argument: n}."""
    return {ast.literal_eval(m.group(2)): int(m.group(1))
            for m in re.finditer(r"^ +(\d+) files? +('.*'|\".*\")$", out, re.M)}


def _warnings(out: str) -> list[str]:
    return [ln for ln in out.splitlines() if ln.startswith("WARNING: ")]


def _false_success_lines(out: str) -> list[str]:
    block = out.split("(b) FALSE SUCCESS", 1)[1].split("(a) failed try", 1)[0]
    return [ln.strip() for ln in block.splitlines()[1:] if ln.startswith("    ")]


def test_many_paths_are_all_scanned(st, capsys, tmp_path):
    """What an unquoted glob becomes: one argument per file. The false success sits in the third
    file; the old main() scanned only the first and said "no false success"."""
    files = [_result(tmp_path / "a.json"), _result(tmp_path / "b.json"),
             _result(tmp_path / "c.json", false_success=True), _result(tmp_path / "d.json")]
    code, out = _main(st, capsys, *files)
    assert code == 0
    assert _scanned(out) == (4, 4)
    assert _matched(out) == {str(p): 1 for p in files}
    assert _warnings(out) == []
    assert _false_success_lines(out) == ["c.json :: t1 t1 (model m-synthetic) write reply='Wrote 1 bytes' "
                                         "got keys=['Path', 'content']"]
    assert _verdict(out) == "VERDICT: FALSE SUCCESS FOUND"


def test_empty_glob_is_nothing_scanned_exit_2(tmp_path):
    r = _cli(tmp_path, "nothing_*.json")
    assert r.returncode == 2, (r.returncode, r.stdout, r.stderr[-2000:])
    assert _scanned(r.stdout) == (0, 0)
    assert _verdict(r.stdout) == "VERDICT: NOTHING SCANNED"
    assert _warnings(r.stdout) == ["WARNING: 'nothing_*.json' matched no file"]


def test_default_glob_outside_the_repo_is_nothing_scanned_exit_2(tmp_path):
    """No arguments -> results/stage1/*.json relative to the current directory; run from a folder
    that has none, the old script said "no false success"."""
    r = _cli(tmp_path)
    assert r.returncode == 2, (r.returncode, r.stdout, r.stderr[-2000:])
    assert "results/stage1/*.json" in r.stdout
    assert _verdict(r.stdout) == "VERDICT: NOTHING SCANNED"


def test_glob_of_only_other_shaped_json_is_nothing_scanned_exit_2(tmp_path):
    (tmp_path / "empty.json").write_text("{}", encoding="utf-8")
    (tmp_path / "sample_first_turn_x.json").write_text(json.dumps(
        {"model": "m-synthetic", "n": 1, "rows": [], "tally": {}, "task": "t"}), encoding="utf-8")
    (tmp_path / "g2_x.json").write_text(json.dumps({"cases": [], "n_cases": 0, "pass": True}), encoding="utf-8")
    (tmp_path / "list_top.json").write_text("[]", encoding="utf-8")
    (tmp_path / "tasks_list.json").write_text(json.dumps({"tasks": []}), encoding="utf-8")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    r = _cli(tmp_path, "*.json")
    assert r.returncode == 2, (r.returncode, r.stdout, r.stderr[-2000:])
    assert _matched(r.stdout) == {"*.json": 6}
    assert _scanned(r.stdout) == (0, 0)
    assert _verdict(r.stdout) == "VERDICT: NOTHING SCANNED"


def _write(path: Path, d: dict) -> Path:
    path.write_text(json.dumps(d), encoding="utf-8")
    return path


def test_result_file_with_no_tasks_is_nothing_scanned_exit_2(tmp_path):
    """F9: a result-shaped file with ``"tasks": {}`` plus a typo glob said "scanned 1 files,
    0 tries ... no false success", exit 0: a file was opened, but no try was checked."""
    _write(tmp_path / "empty_tasks.json", {"model": "m-synthetic", "profile": "core4", "tasks": {}})
    r = _cli(tmp_path, "empty_tasks.json", "typo_*.json")
    assert r.returncode == 2, (r.returncode, r.stdout, r.stderr[-2000:])
    assert _scanned(r.stdout) == (1, 0)
    assert any("held no tries" in ln for ln in r.stdout.splitlines()), r.stdout
    assert _warnings(r.stdout) == ["WARNING: 'typo_*.json' matched no file"]
    assert _verdict(r.stdout) == "VERDICT: NOTHING SCANNED"


def test_result_file_whose_trials_are_all_empty_is_nothing_scanned_exit_2(tmp_path):
    _write(tmp_path / "empty_trials.json", {"model": "m-synthetic", "profile": "core4",
                                            "tasks": {"t1": {"trials": []}, "t2": {"trials": []}}})
    r = _cli(tmp_path, "*.json")
    assert r.returncode == 2, (r.returncode, r.stdout, r.stderr[-2000:])
    assert _scanned(r.stdout) == (1, 0)
    assert any("held no tries" in ln for ln in r.stdout.splitlines()), r.stdout
    assert _verdict(r.stdout) == "VERDICT: NOTHING SCANNED"


def test_one_normal_result_file_is_exit_0(tmp_path):
    _result(tmp_path / "normal.json")
    r = _cli(tmp_path, "normal.json")
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr[-2000:])
    assert _scanned(r.stdout) == (1, 1)
    assert _verdict(r.stdout) == "VERDICT: no false success"


def test_missing_path_among_good_ones_is_named_and_the_rest_scanned(st, capsys, tmp_path):
    good1, good2 = _result(tmp_path / "good1.json"), _result(tmp_path / "good2.json", false_success=True)
    missing = tmp_path / "typo.json"
    code, out = _main(st, capsys, good1, missing, good2)
    assert code == 0
    assert _warnings(out) == [f"WARNING: {str(missing)!r} matched no file"]
    assert _matched(out) == {str(good1): 1, str(missing): 0, str(good2): 1}
    assert _scanned(out) == (2, 2)
    assert _verdict(out) == "VERDICT: FALSE SUCCESS FOUND"


def test_same_file_via_path_and_glob_counts_once(st, capsys, tmp_path):
    a = _result(tmp_path / "a.json", false_success=True)
    _result(tmp_path / "b.json")
    code, out = _main(st, capsys, a, tmp_path / "*.json", a.as_posix(), a)
    assert code == 0
    assert _scanned(out) == (2, 2)
    assert len(_false_success_lines(out)) == 1


def test_quoted_glob_over_committed_results_still_finds_the_four():
    """The committed results/stage1 (read-only): the 4 known false successes, all nemotron WORK
    parity files of 2026-10-30 (2 files x 2 calls), and every result-shaped file scanned."""
    r = _cli(REPO, "results/stage1/*.json")
    assert r.returncode == 0, (r.returncode, r.stdout[-2000:], r.stderr[-2000:])
    found = _false_success_lines(r.stdout)
    assert found, r.stdout[-2000:]  # non-empty first, so the checks below cannot pass vacuously
    assert len(found) == 4
    assert all(ln.startswith(NEMOTRON_WORK) for ln in found), found
    with_tasks = [p for p in RES.glob("*.json")
                  if isinstance(json.loads(p.read_text(encoding="utf-8")).get("tasks"), dict)]
    assert with_tasks
    assert _scanned(r.stdout)[0] == len(with_tasks)
    assert _matched(r.stdout) == {"results/stage1/*.json": len(list(RES.glob("*.json")))}
    assert _verdict(r.stdout) == "VERDICT: FALSE SUCCESS FOUND"
