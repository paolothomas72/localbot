"""X4b: gate_run plans, resumes, and records a batch. No GPU and no real model."""
from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NAME = "pao" + "lo"


def _load(name: str):
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def gate(monkeypatch, tmp_path):
    mod = _load("gate_run")
    monkeypatch.setenv("LOCALBOT_RUNLOCK_PATH", str(tmp_path / "run.lock"))
    monkeypatch.setenv("LOCALBOT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(mod, "git_porcelain", lambda: "")
    monkeypatch.setattr(mod, "git_head", lambda: "abc123def456")
    monkeypatch.setattr(mod, "gpu_visible", lambda: True)
    monkeypatch.setattr(mod, "ollama_tags", lambda: {"m-a", "m-b", "m-c"})
    return mod


def _plan(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8", newline="\n")
    return path


def _row(**kw):
    base = {"gate": "G4", "model": "m-a", "pack": "privacy", "profile": "core4", "n": 1}
    base.update(kw)
    return base


def test_list_prints_grouped_commands_and_writes_nothing(gate, tmp_path, monkeypatch):
    plan = _plan(tmp_path / "plan.json", [
        _row(model="m-b", pack="work"),
        _row(model="m-a", pack="work"),
        _row(model="m-b", pack="trust"),
    ])
    runs = tmp_path / "runs"
    monkeypatch.setenv("LOCALBOT_RUNS_DIR", str(runs))
    code = gate.main(["--plan", str(plan), "--commit", "abc123def456", "--list", "--python", "python"])
    assert code == 0
    text = gate.LAST_STDOUT
    lines = [ln for ln in text.splitlines() if "localbot.exam.runner" in ln]
    assert lines == [
        "python -m localbot.exam.runner --pack work --profile core4 --n 1 m-b",
        "python -m localbot.exam.runner --pack trust --profile core4 --n 1 m-b",
        "python -m localbot.exam.runner --pack work --profile core4 --n 1 m-a",
    ]
    assert not runs.exists()
    assert not (tmp_path / "run.lock").exists()


def test_bad_plans_are_refused_in_plain_words(gate, tmp_path):
    cases = [
        (_row(pack="nope"), "unknown pack"),
        (_row(profile="nope"), "unknown profile"),
        (_row(n=0), "n must be at least 1"),
        ("dup", "duplicate row"),
        (_row(num_ctx=8129), "multiple of 1024"),
        (_row(num_ctx=8192), None),
        (_row(max_turns=-1), "max_turns"),
    ]
    for row, reason in cases:
        rows = [_row(), _row()] if row == "dup" else [row]
        path = _plan(tmp_path / "p.json", rows)
        code = gate.main(["--plan", str(path), "--commit", "abc", "--list"])
        if reason is None:
            assert code == 0
        else:
            assert code != 0
            assert reason in gate.LAST_STDOUT + gate.LAST_STDERR


def test_filters_and_an_empty_filter(gate, tmp_path):
    path = _plan(tmp_path / "plan.json", [_row(gate="G4"), _row(gate="G5", model="m-b", pack="trust")])
    code = gate.main(["--plan", str(path), "--commit", "abc", "--list", "--gate", "G5", "--models", "m-b"])
    assert code == 0
    assert "m-b" in gate.LAST_STDOUT and "--pack trust" in gate.LAST_STDOUT
    assert "--pack privacy" not in gate.LAST_STDOUT
    code = gate.main(["--plan", str(path), "--commit", "abc", "--list", "--gate", "NOPE"])
    assert code != 0
    assert "no rows match" in (gate.LAST_STDOUT + gate.LAST_STDERR).lower()


def test_omitted_defaults_add_no_extra_flags(gate, tmp_path):
    path = _plan(tmp_path / "plan.json", [_row()])
    assert gate.main(["--plan", str(path), "--commit", "abc", "--list", "--python", "py"]) == 0
    line = next(ln for ln in gate.LAST_STDOUT.splitlines() if "exam.runner" in ln)
    assert "--num-ctx" not in line and "--max-turns" not in line and "--num-predict" not in line
    assert "--profile core4" in line


def test_shipped_plan_long_rows_use_core4_product():
    gate = _load("gate_run")
    rows = gate.load_plan(REPO / "docs" / "gate_plan.json")
    long_rows = [r for r in rows if r["pack"] == "long"]
    assert long_rows and all(r["profile"] == "core4_product" and r["gate"] == "LONG" for r in long_rows)
    assert not any(r["gate"] in {"G1", "G2"} for r in rows)


def test_dirty_tree_stops_before_a_run(gate, tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "git_porcelain", lambda: " M localbot/loop.py")
    calls = []
    monkeypatch.setattr(gate, "launch", lambda *a, **k: calls.append(a) or 0)
    path = _plan(tmp_path / "plan.json", [_row()])
    code = gate.main(["--plan", str(path), "--commit", "abc123def456", "--tag", "t"])
    assert code != 0 and not calls
    assert "clean" in (gate.LAST_STDOUT + gate.LAST_STDERR).lower() or "uncommitted" in (
        gate.LAST_STDOUT + gate.LAST_STDERR).lower()


def test_wrong_head_stops(gate, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(gate, "launch", lambda *a, **k: calls.append(a) or 0)
    path = _plan(tmp_path / "plan.json", [_row()])
    code = gate.main(["--plan", str(path), "--commit", "deadbeef", "--tag", "t"])
    assert code != 0 and not calls
    assert "HEAD" in gate.LAST_STDOUT + gate.LAST_STDERR


def test_lock_busy_names_the_holder(gate, tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO / "scripts"))
    import runlock
    held = runlock.acquire("gate-holder", cmd="held-by-test")
    try:
        calls = []
        monkeypatch.setattr(gate, "launch", lambda *a, **k: calls.append(a) or 0)
        path = _plan(tmp_path / "plan.json", [_row()])
        code = gate.main(["--plan", str(path), "--commit", "abc123def456", "--tag", "t"])
        assert code != 0 and not calls
        assert "gate-holder" in gate.LAST_STDOUT + gate.LAST_STDERR
    finally:
        held.release()


PIN = "6cc467f05439"
OTHER = "aaaaaaaaaaaa"


def _summary_rows(tmp_path: Path, tag: str = "t") -> list[dict]:
    files = list((tmp_path / "runs" / tag).glob("summary_*.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text(encoding="utf-8"))["rows"]


def test_pinned_row_with_matching_digest_runs(gate, tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ollama_tags", lambda: {"m-a": PIN})
    seen = []

    def launch(argv, log_path, cwd):
        seen.append(argv[-1])
        _finish(log_path, tmp_path / "ran.json", "abc123def456")
        return 0

    monkeypatch.setattr(gate, "launch", launch)
    path = _plan(tmp_path / "plan.json", [_row(digest=PIN)])
    assert gate.main(["--plan", str(path), "--commit", "abc123def456", "--tag", "t"]) == 0
    assert seen == ["m-a"]
    rows = _summary_rows(tmp_path)
    assert rows[0]["status"] == "ok" and rows[0]["digest_seen"] == PIN


def test_pinned_row_with_a_different_digest_is_skipped(gate, tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ollama_tags", lambda: {"m-a": OTHER, "m-b": PIN})
    seen = []

    def launch(argv, log_path, cwd):
        seen.append(argv[-1])
        _finish(log_path, tmp_path / "ran.json", "abc123def456")
        return 0

    monkeypatch.setattr(gate, "launch", launch)
    path = _plan(tmp_path / "plan.json", [
        _row(model="m-a", digest=PIN),
        _row(model="m-b", pack="trust"),
    ])
    assert gate.main(["--plan", str(path), "--commit", "abc123def456", "--tag", "t"]) == 0
    assert seen == ["m-b"]
    text = gate.LAST_STDOUT + gate.LAST_STDERR
    assert f"digest differs: expected {PIN}, found {OTHER}" in text
    assert "SKIPPED" in text
    rows = {r["model"]: r for r in _summary_rows(tmp_path)}
    assert rows["m-a"]["status"] == "SKIPPED" and rows["m-a"]["digest_seen"] == OTHER
    assert rows["m-b"]["status"] == "ok" and rows["m-b"]["digest_seen"] == PIN


def test_unpinned_row_runs_on_its_tag_alone(gate, tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ollama_tags", lambda: {"m-a": OTHER})
    seen = []

    def launch(argv, log_path, cwd):
        seen.append(argv[-1])
        _finish(log_path, tmp_path / "ran.json", "abc123def456")
        return 0

    monkeypatch.setattr(gate, "launch", launch)
    path = _plan(tmp_path / "plan.json", [_row()])
    assert gate.main(["--plan", str(path), "--commit", "abc123def456", "--tag", "t"]) == 0
    assert seen == ["m-a"]
    assert _summary_rows(tmp_path)[0]["digest_seen"] == OTHER


def test_digest_seen_is_recorded_for_pinned_and_unpinned_rows(gate, tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ollama_tags", lambda: {"m-a": PIN, "m-b": OTHER})
    monkeypatch.setattr(gate, "launch", lambda argv, log_path, cwd: (
        _finish(log_path, tmp_path / f"{argv[-1]}.json", "abc123def456") or 0))
    path = _plan(tmp_path / "plan.json", [_row(digest=PIN), _row(model="m-b", pack="trust")])
    assert gate.main(["--plan", str(path), "--commit", "abc123def456", "--tag", "t"]) == 0
    rows = _summary_rows(tmp_path)
    assert {r["model"]: r["digest_seen"] for r in rows} == {"m-a": PIN, "m-b": OTHER}


def test_plan_pins_match_the_batch_script():
    import re
    gate = _load("gate_run")
    text = (REPO / "scripts" / "stage1_gpu_batch.ps1").read_text(encoding="utf-8")
    pins: dict[str, str] = {}
    for tag, digest in re.findall(r'Tag\s*=\s*"([^"]+)"\s*;\s*Id\s*=\s*"([^"]*)"', text):
        pins[tag] = digest
    rows = gate.load_plan(REPO / "docs" / "gate_plan.json")
    by_model: dict[str, set[str]] = {}
    for row in rows:
        by_model.setdefault(row["model"], set()).add(row.get("digest") or "")
    for tag, digest in pins.items():
        assert tag in by_model, tag
        if digest:
            assert by_model[tag] == {digest}, (tag, by_model[tag], digest)
        else:
            assert by_model[tag] == {""}, (tag, by_model[tag])


def test_missing_tag_is_skipped_and_the_other_row_runs(gate, tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ollama_tags", lambda: {"m-a"})
    seen = []

    def launch(argv, log_path, cwd):
        seen.append(argv[-1])
        _finish(log_path, tmp_path / f"{argv[-1]}.json", "abc123def456")
        return 0

    monkeypatch.setattr(gate, "launch", launch)
    path = _plan(tmp_path / "plan.json", [_row(model="m-a"), _row(model="m-missing", pack="trust")])
    assert gate.main(["--plan", str(path), "--commit", "abc123def456", "--tag", "t"]) == 0
    assert seen == ["m-a"]
    text = gate.LAST_STDOUT + gate.LAST_STDERR
    assert "SKIPPED" in text and "m-missing" in text


def _finish(log_path: Path, result: Path, commit: str, wrote: bool = True) -> None:
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps({
        "run_env": {"git": {"commit": commit}},
        "wall_s": 3.0, "passed": 1, "total": 1, "passed_lab_judge": 1,
        "host_changes": 0, "g5_secret_hits": 0, "secret_seen": 0,
        "gpu_fit": {"lane": "gpu", "timing_comparable": True},
        "timing_comparable": True,
        "tasks": {"only": {"pass": True, "trials": [{
            "end_reason": "done",
            "telemetry": {"wall_s": 3.0, "tok_s": 10.0, "eval_count": 4, "vram_used_mib_peak": 100,
                          "calls": [{"wall_s": 2.0, "prompt_eval_count": 50, "eval_count": 4}]},
        }]}},
    }), encoding="utf-8", newline="\n")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = f"wrote {result}  privacy m disk 1/1\n" if wrote else "started\n"
    log_path.write_text(line, encoding="utf-8", newline="\n")


def test_three_rows_run_in_order_and_write_a_summary(gate, tmp_path, monkeypatch):
    order = []

    def launch(argv, log_path, cwd):
        order.append((argv[argv.index("--pack") + 1], argv[-1]))
        _finish(log_path, tmp_path / f"r{len(order)}.json", "abc123def456")
        return 0

    monkeypatch.setattr(gate, "launch", launch)
    path = _plan(tmp_path / "plan.json", [
        _row(model="m-a", pack="privacy"),
        _row(model="m-a", pack="trust"),
        _row(model="m-b", pack="privacy"),
    ])
    assert gate.main(["--plan", str(path), "--commit", "abc123def456", "--tag", "batch"]) == 0
    assert order == [("privacy", "m-a"), ("trust", "m-a"), ("privacy", "m-b")]
    summaries = list((tmp_path / "runs" / "batch").glob("summary_*.json"))
    assert len(summaries) == 1
    data = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert len(data["rows"]) == 3
    assert all(r["status"] == "ok" for r in data["rows"])


def test_a_crash_on_row_2_reruns_only_the_half_row(gate, tmp_path, monkeypatch):
    calls = []

    raised = {"n": 0}

    def launch(argv, log_path, cwd):
        calls.append(argv[-1] + ":" + argv[argv.index("--pack") + 1])
        raised["n"] += 1
        if raised["n"] == 2:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("started, no result\n", encoding="utf-8", newline="\n")
            raise RuntimeError("killed mid row")
        _finish(log_path, tmp_path / f"r{raised['n']}.json", "abc123def456")
        return 0

    monkeypatch.setattr(gate, "launch", launch)
    path = _plan(tmp_path / "plan.json", [
        _row(pack="privacy"), _row(pack="trust"), _row(pack="work"),
    ])
    with pytest.raises(RuntimeError):
        gate.main(["--plan", str(path), "--commit", "abc123def456", "--tag", "batch"])
    calls.clear()
    assert gate.main(["--plan", str(path), "--commit", "abc123def456", "--tag", "batch"]) == 0
    assert calls == ["m-a:trust", "m-a:work"]


def test_other_commit_is_not_done(gate, tmp_path, monkeypatch):
    calls = []

    def launch(argv, log_path, cwd):
        calls.append(1)
        _finish(log_path, tmp_path / "fresh.json", "abc123def456")
        return 0

    monkeypatch.setattr(gate, "launch", launch)
    path = _plan(tmp_path / "plan.json", [_row()])
    # Plant a finished log from another commit before the run.
    log = tmp_path / "runs" / "batch" / "old.log"
    _finish(log, tmp_path / "planted.json", "ffffffffffffffff")
    (tmp_path / "runs" / "batch").mkdir(parents=True, exist_ok=True)
    (tmp_path / "runs" / "batch" / "summary_old.json").write_text(json.dumps({"rows": [{
        "status": "ok", "gate": "G4", "model": "m-a", "pack": "privacy", "profile": "core4",
        "result": f"wrote {tmp_path / 'planted.json'}  privacy m",
        "log": str(log),
    }]}), encoding="utf-8", newline="\n")
    assert gate.main(["--plan", str(path), "--commit", "abc123def456", "--tag", "batch"]) == 0
    assert calls == [1]


def test_missing_wrote_line_is_not_done(gate, tmp_path, monkeypatch):
    calls = []

    def launch(argv, log_path, cwd):
        calls.append(1)
        _finish(log_path, tmp_path / "r.json", "abc123def456", wrote=True)
        return 0

    monkeypatch.setattr(gate, "launch", launch)
    path = _plan(tmp_path / "plan.json", [_row()])
    batch = tmp_path / "runs" / "batch"
    batch.mkdir(parents=True)
    log = batch / "half.log"
    log.write_text("runner started\n", encoding="utf-8", newline="\n")
    result = tmp_path / "half.json"
    result.write_text(json.dumps({"run_env": {"git": {"commit": "abc123def456"}}}), encoding="utf-8")
    (batch / "summary_old.json").write_text(json.dumps({"rows": [{
        "status": "ok", "gate": "G4", "model": "m-a", "pack": "privacy", "profile": "core4",
        "result": "runner started", "log": str(log),
    }]}), encoding="utf-8", newline="\n")
    assert gate.main(["--plan", str(path), "--commit", "abc123def456", "--tag", "batch"]) == 0
    assert calls == [1]


def test_summaries_and_logs_are_stamped_not_overwritten(gate, tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "stamp", lambda: "20261103T010000Z")

    def launch(argv, log_path, cwd):
        _finish(log_path, tmp_path / "r.json", "abc123def456")
        return 0

    monkeypatch.setattr(gate, "launch", launch)
    path = _plan(tmp_path / "plan.json", [_row()])
    assert gate.main(["--plan", str(path), "--commit", "abc123def456", "--tag", "batch"]) == 0
    first = (tmp_path / "runs" / "batch" / "summary_20261103T010000Z.json").read_bytes()
    logs = list((tmp_path / "runs" / "batch").glob("*.log"))
    assert len(logs) == 1
    log_bytes = logs[0].read_bytes()
    assert gate.main(["--plan", str(path), "--commit", "abc123def456", "--tag", "batch"]) == 0
    assert (tmp_path / "runs" / "batch" / "summary_20261103T010000Z.json").read_bytes() == first
    assert (tmp_path / "runs" / "batch" / "summary_20261103T010000Z_2.json").is_file()
    assert logs[0].read_bytes() == log_bytes


def test_space_and_apostrophe_in_repo_and_python_path(gate, tmp_path, monkeypatch):
    repo = tmp_path / "repo o'brien"
    repo.mkdir()
    py_dir = tmp_path / "py o'brien"
    py_dir.mkdir()
    script = py_dir / "runner.py"
    script.write_text("print('ran')\n", encoding="utf-8", newline="\n")
    seen = {}

    def launch(argv, log_path, cwd):
        seen["argv0"] = argv[0]
        seen["cwd"] = str(cwd)
        proc = subprocess.run([sys.executable, str(script)], cwd=repo, capture_output=True, text=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(proc.stdout, encoding="utf-8", newline="\n")
        _finish(log_path, tmp_path / "r.json", "abc123def456")
        return 0

    monkeypatch.setattr(gate, "launch", launch)
    monkeypatch.setattr(gate, "REPO", repo)
    path = _plan(tmp_path / "plan.json", [_row()])
    py = py_dir / "python.exe"
    assert gate.main(["--plan", str(path), "--commit", "abc123def456", "--tag", "batch",
                      "--python", str(py)]) == 0
    assert seen["argv0"] == str(py)
    assert "o'brien" in seen["cwd"] and "repo o" in seen["cwd"]


def test_efficiency_matches_the_hand_count(gate):
    sample = REPO / "tests" / "fixtures" / "privacy_core4_sample.json"
    data = json.loads(sample.read_text(encoding="utf-8"))
    import hashlib
    assert hashlib.sha256(sample.read_bytes()).hexdigest() == (
        "0fbe96c624c9ba981042a6029a4e8b27f72bf5c04695aa86a01ffa2e9e5ce6fc")
    eff = gate.efficiency(data)
    assert eff["wall_s"] == 412.8
    assert eff["tok_s"] == 102.5
    assert eff["context_peak"] == 965
    assert eff["output_tokens"] == 13827
    assert eff["turns"] == 30
    assert eff["end_reasons"] == {"done": 24}
    assert eff["gpu_lane"] == "gpu"
    assert eff["vram_peak_mib"] == 9903
    assert eff["time_per_task_s"] == 51.6
    assert eff["model_load_s"] == 7.659
    assert eff["timing_comparable"] is True


def test_a_spilled_run_is_not_in_the_median(gate, tmp_path):
    history = tmp_path / "summary.json"
    history.write_text(json.dumps({"rows": [
        {"model": "m-a", "pack": "privacy", "wall_s": 10, "timing_comparable": True, "status": "ok"},
        {"model": "m-a", "pack": "privacy", "wall_s": 30, "timing_comparable": True, "status": "ok"},
        {"model": "m-a", "pack": "privacy", "wall_s": 999, "timing_comparable": False, "status": "ok"},
        {"model": "m-a", "pack": "trust", "wall_s": 5, "timing_comparable": False, "status": "ok"},
    ]}), encoding="utf-8", newline="\n")
    text = gate.estimate_text([_row(pack="privacy"), _row(pack="trust"), _row(model="m-b")], [history])
    assert "privacy" in text and "20" in text  # median of 10 and 30
    assert "unknown" in text


def test_estimate_without_history_is_unknown(gate):
    text = gate.estimate_text([_row()], [])
    assert "unknown" in text


def test_new_scripts_have_no_owner_name_or_personal_path():
    needle = NAME
    home = "Users" + "/" 
    files = [
        REPO / "scripts" / "gate_run.py",
        REPO / "scripts" / "weekly_report.py",
        REPO / "docs" / "gate_plan.json",
        REPO / "docs" / "reports" / "WEEKLY_TEMPLATE.md",
        REPO / "tests" / "test_gate_run.py",
        REPO / "tests" / "test_weekly_report.py",
    ]
    private = REPO / "docs" / "reports" / "WEEKLY_TEMPLATE.md"  # not in the public copy
    hits = []
    for path in files:
        if path == private and not path.exists():
            continue  # skipped below, after every other file has been checked
        text = path.read_text(encoding="utf-8")
        if needle in text.lower() or home + needle in text.replace("\\", "/"):
            hits.append(path.name)
    assert hits == []
    if not private.exists():
        pytest.skip("needs a private file that is not in the public copy: docs/reports/WEEKLY_TEMPLATE.md")
