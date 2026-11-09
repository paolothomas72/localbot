"""C18b: the queue skips a broken model by itself, between packs only (red tests first).

Nothing here starts a process, loads a model or touches the network: ``gate_run.launch`` is replaced,
the queue runs on ``FakeEnv`` (tests/test_c9_model_queue.py), and the fake run folders live in tmp_path.
"""
from __future__ import annotations

import json
import socket
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import scripts.model_queue as mq
import scripts.queue_status as qs
from tests import test_c18a_queue_status as c18a
from tests.test_c9_model_queue import FakeEnv, write_batch

gate = mq.gate          # the same gate_run module the queue calls
runlock = mq.runlock

COMMIT = "abc123def456"
MODEL = c18a.MODEL      # "org/m:1"
TAG = c18a.TAG          # "t"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Any socket connect in these tests is a bug (the check must not reach 127.0.0.1:11434)."""
    def refuse(self, *a, **k):
        raise AssertionError("network used: a C18b test must not connect anywhere")
    monkeypatch.setattr(socket.socket, "connect", refuse)


# --- gate_run.main with the new before_pack hook ---------------------------------------------------------------------

def _gate_setup(tmp_path: Path, monkeypatch, models: list[str]) -> tuple[list, Path, Path]:
    """Patch gate_run so main() runs offline; returns (launch calls, plan path, runs dir)."""
    runs = tmp_path / "runs"
    monkeypatch.setenv("LOCALBOT_RUNS_DIR", str(runs))
    monkeypatch.setenv("LOCALBOT_RUNLOCK_PATH", str(tmp_path / "run.lock"))
    monkeypatch.delenv(runlock.HELD_ENV, raising=False)
    monkeypatch.setattr(gate, "git_porcelain", lambda: "")
    monkeypatch.setattr(gate, "git_head", lambda: COMMIT)
    monkeypatch.setattr(gate, "gpu_visible", lambda: True)
    monkeypatch.setattr(gate, "ollama_tags", lambda: {m: "d" for m in models})
    calls: list[tuple[str, str]] = []

    def launch(argv, log_path, cwd):
        model = argv[-1]
        pack = argv[argv.index("--pack") + 1]
        calls.append((model, pack))
        result = tmp_path / "results" / f"r{len(calls)}.json"
        result.parent.mkdir(exist_ok=True)
        result.write_text(json.dumps({"run_env": {"git": {"commit": COMMIT}}, "tasks": {}, "total": 0, "wall_s": 1}),
                          encoding="utf-8", newline="\n")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"wrote {result}\n", encoding="utf-8", newline="\n")
        return 0

    monkeypatch.setattr(gate, "launch", launch)
    plan = tmp_path / "plan.json"
    rows = [{"gate": g, "model": m, "pack": p, "profile": "core4", "n": 3}
            for g, p in (("Q-work", "work"), ("Q-trust", "trust")) for m in models]
    plan.write_text(json.dumps({"rows": rows}), encoding="utf-8", newline="\n")
    return calls, plan, runs


def _journal(runs: Path, tag: str) -> list[dict]:
    items = []
    for path in sorted((runs / tag).glob("journal_*.jsonl")):
        items += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return items


def _gate_argv(plan: Path) -> list[str]:
    return ["--plan", str(plan), "--commit", COMMIT, "--tag", "g"]


def test_1_before_pack_none_launches_every_row(tmp_path, monkeypatch):
    calls, plan, runs = _gate_setup(tmp_path, monkeypatch, ["x:1b", "y:1b"])
    code = gate.main(_gate_argv(plan))
    assert code == 0
    assert sorted(calls) == [("x:1b", "trust"), ("x:1b", "work"), ("y:1b", "trust"), ("y:1b", "work")]
    assert all(r["status"] == "ok" for r in _journal(runs, "g"))


def test_1b_before_pack_none_is_the_default_keyword(tmp_path, monkeypatch):
    calls, plan, runs = _gate_setup(tmp_path, monkeypatch, ["x:1b"])
    assert gate.main(_gate_argv(plan), before_pack=None) == 0
    assert len(calls) == 2


def test_2_before_pack_reason_skips_that_models_rows_only(tmp_path, monkeypatch):
    calls, plan, runs = _gate_setup(tmp_path, monkeypatch, ["x:1b", "y:1b"])
    asked: list[tuple[str, str]] = []

    def before_pack(row):
        asked.append((row["model"], row["pack"]))
        return "broken" if row["model"] == "x:1b" else None

    code = gate.main(_gate_argv(plan), before_pack=before_pack)
    assert code == 0
    assert [c for c in calls if c[0] == "x:1b"] == []                    # launch never called for X
    assert sorted(calls) == [("y:1b", "trust"), ("y:1b", "work")]       # Y still launches
    assert sorted(asked) == [("x:1b", "trust"), ("x:1b", "work"), ("y:1b", "trust"), ("y:1b", "work")]
    skipped = [r for r in _journal(runs, "g") if r["model"] == "x:1b"]
    assert sorted(r["pack"] for r in skipped) == ["trust", "work"]       # each skipped pack is on record
    assert all(r["status"] == "SKIPPED" and r["result"] == "skipped: broken" for r in skipped)
    ok = [r for r in _journal(runs, "g") if r["model"] == "y:1b"]
    assert len(ok) == 2 and all(r["status"] == "ok" for r in ok)


def test_2b_the_skip_line_is_ascii_and_names_the_model(tmp_path, monkeypatch, capsys):
    name = "caf\u00e9/m\u00f6del:1b"
    calls, plan, runs = _gate_setup(tmp_path, monkeypatch, [name])
    assert gate.main(_gate_argv(plan), before_pack=lambda row: "broken") == 0
    out = capsys.readouterr().out
    assert calls == []
    out.encode("ascii")                                                  # raises if any character is not ASCII
    assert "skipped: broken" in out and "caf" in out


# --- the default check_model on a fake batch ---------------------------------------------------------------------------

def _check(tmp_path: Path, model: str = MODEL):
    return mq.Env().check_model(model, tmp_path / f"queue_{TAG}")


def test_3_five_infra_tries_in_a_running_pack_is_broken(tmp_path):
    c18a._build(tmp_path, lines=[c18a._try(f"e_{i}", infra=True) for i in range(5)])
    assert _check(tmp_path) == "broken"


def test_3a_five_infra_tries_in_a_finished_pack_is_broken(tmp_path):
    c18a._build(tmp_path, lines=[c18a._try(f"e_{i}", infra=True) for i in range(5)] + [c18a._wrote(no_result=0)])
    assert _check(tmp_path) == "broken"


def test_3b_four_infra_then_one_normal_try_is_none(tmp_path):
    lines = [c18a._try(f"e_{i}", infra=True) for i in range(4)] + [c18a._try("e_4", infra=False)]
    c18a._build(tmp_path, lines=lines)
    assert _check(tmp_path) is None


def _real_pack(tasks: int, no_result: int, per_task: int = 3, disk: bool = True) -> list[str]:
    """The real shape of a finished pack log (round 3): ``per_task`` try lines for each of ``tasks`` tasks, none of
    them infra=True, and a wrote line whose no_result counts TASKS and whose 'disk P/T' gives the task total."""
    lines = [c18a._try(f"e_{t}", k=k, infra=False) for t in range(tasks) for k in range(1, per_task + 1)]
    wrote = c18a._wrote(no_result=no_result)
    if disk:
        wrote = wrote.replace("disk 9/12 lab-judge 9/12", f"disk 0/{tasks} lab-judge 0/{tasks}")
    else:
        wrote = wrote.replace("disk 9/12 ", "")
        assert "disk" not in wrote
    return lines + [wrote]


def test_3c_every_task_of_a_finished_pack_without_a_result_is_broken_no_result(tmp_path):
    lines = _real_pack(tasks=6, no_result=6)
    assert len([ln for ln in lines if " t1" in ln or " t2" in ln or " t3" in ln]) == 18   # 3 try lines per task
    c18a._build(tmp_path, lines=lines)
    assert _check(tmp_path) == "broken (no result)"


def test_3d_no_result_below_the_task_total_is_none(tmp_path):
    c18a._build(tmp_path, lines=_real_pack(tasks=6, no_result=5))
    assert _check(tmp_path) is None


def test_3d2_a_wrote_line_without_disk_has_no_total_and_is_none(tmp_path):
    c18a._build(tmp_path, lines=_real_pack(tasks=6, no_result=6, disk=False))
    assert _check(tmp_path) is None


def test_3d3_real_numbers_thirty_try_lines_over_ten_tasks(tmp_path):
    c18a._build(tmp_path, lines=_real_pack(tasks=10, no_result=10))
    assert _check(tmp_path) == "broken (no result)"


def test_3d4_real_numbers_thirty_try_lines_over_ten_tasks_none_missing(tmp_path):
    c18a._build(tmp_path, lines=_real_pack(tasks=10, no_result=0))
    assert _check(tmp_path) is None


def test_3d5_a_zero_task_total_is_none(tmp_path):
    c18a._build(tmp_path, lines=_real_pack(tasks=0, no_result=0))
    assert _check(tmp_path) is None


# --- round 3, item 2: a capped model resumed with --finish-capped is checked too ---------------------------------------

def test_3p_capped_row_with_five_infra_tries_at_the_end_is_broken(tmp_path):
    c18a._build(tmp_path, lines=[c18a._try(f"e_{i}", infra=True) for i in range(5)], status="capped")
    assert c18a._one(tmp_path)["step"] == "capped"                       # the BROKEN flag cannot fire for this step
    assert _check(tmp_path) == "broken"


def test_3q_capped_row_with_four_infra_tries_at_the_end_is_none(tmp_path):
    c18a._build(tmp_path, lines=[c18a._try(f"e_{i}", infra=True) for i in range(4)], status="capped")
    assert _check(tmp_path) is None


def test_3e_a_pack_finished_before_c17_without_no_result_is_none(tmp_path):
    old_wrote = c18a._wrote().replace("no_result=2 ", "")
    assert "no_result" not in old_wrote
    lines = [c18a._try(f"e_{i}", infra=False) for i in range(6)] + [old_wrote]
    c18a._build(tmp_path, lines=lines)
    assert _check(tmp_path) is None


def test_3f_a_running_pack_never_counts_as_broken_no_result(tmp_path):
    c18a._build(tmp_path, lines=[c18a._try(f"e_{i}", infra=False) for i in range(6)])
    assert _check(tmp_path) is None


def test_3g_only_the_newest_finished_pack_counts(tmp_path):
    older = "\n".join([c18a._try(f"e_{i}", infra=False) for i in range(6)] + [c18a._wrote(no_result=6)]) + "\n"
    healthy = [c18a._try(f"e_{i}", infra=False) for i in range(6)] + [c18a._wrote(no_result=0)]
    c18a._build(tmp_path, lines=healthy, extra_logs={"packs_Q-trust_trust_core4_org_m_1.log": older})
    assert _check(tmp_path) is None


def test_3h_no_pack_logs_is_none(tmp_path):
    c18a._build(tmp_path, lines=None)
    assert _check(tmp_path) is None


def test_3i_model_not_in_the_records_is_none(tmp_path):
    c18a._build(tmp_path, lines=[c18a._try(f"e_{i}", infra=True) for i in range(5)])
    assert _check(tmp_path, model="other/model:9") is None


def test_3j_missing_batch_folder_is_none_not_an_exception(tmp_path):
    assert mq.Env().check_model(MODEL, tmp_path / "queue_nothing") is None


def test_3k_check_writes_nothing_in_the_run_folder(tmp_path):
    c18a._build(tmp_path, lines=[c18a._try(f"e_{i}", infra=True) for i in range(5)])
    before = c18a._hash_tree(tmp_path)
    assert _check(tmp_path) == "broken"
    assert c18a._hash_tree(tmp_path) == before


def test_3l_queue_status_exposes_the_newest_finished_pack_no_result_and_total(tmp_path, capsys):
    c18a._build(tmp_path, lines=_real_pack(tasks=6, no_result=4))
    rec = c18a._one(tmp_path)
    assert rec["newest_pack_no_result"] == 4 and rec["newest_pack_total"] == 6 and rec["tries"] == 18
    printed = qs.format_line(rec)
    assert "newest_pack" not in printed                                  # what it prints does not change
    printed.encode("ascii")


def test_3l2_printed_status_line_is_the_same_with_or_without_a_disk_field(tmp_path):
    c18a._build(tmp_path, lines=_real_pack(tasks=6, no_result=4))
    with_disk = qs.format_line(c18a._one(tmp_path))
    (tmp_path / "queue_t").rename(tmp_path / "old")
    other = tmp_path / "other"
    other.mkdir()
    c18a._build(other, lines=_real_pack(tasks=6, no_result=4, disk=False))
    assert qs.format_line(c18a._one(other)) == with_disk


def test_3m_newest_pack_no_result_and_total_are_none_for_a_running_pack(tmp_path):
    c18a._build(tmp_path, lines=[c18a._try("e_0")])
    rec = c18a._one(tmp_path)
    assert rec["newest_pack_no_result"] is None and rec["newest_pack_total"] is None


def test_3m2_total_is_none_when_the_wrote_line_has_no_disk_field(tmp_path):
    c18a._build(tmp_path, lines=_real_pack(tasks=6, no_result=6, disk=False))
    rec = c18a._one(tmp_path)
    assert rec["newest_pack_no_result"] == 6 and rec["newest_pack_total"] is None


# --- wiring: the default run_packs passes the hook to gate_run.main -------------------------------------------------

def _wired(tmp_path, monkeypatch, check):
    """Env with the default run_packs and a stand-in check; gate.main is replaced by a recorder."""
    seen: dict = {}

    def fake_main(argv, **kw):
        seen["argv"] = argv
        hook = kw.get("before_pack")
        seen["hook"] = hook
        seen["reason"] = hook({"gate": "Q-privacy", "model": MODEL, "pack": "privacy", "profile": "core4",
                               "n": 3}) if callable(hook) else "NO HOOK"
        return 0

    monkeypatch.setattr(gate, "main", fake_main)
    monkeypatch.setattr(gate, "git_head", lambda: COMMIT)
    env = mq.Env()
    env.check_model = check
    batch = tmp_path / "runs" / f"queue_{TAG}"
    out = batch / "models" / mq._safe(MODEL) / "packs"
    result = env.run_packs(MODEL, batch / "plan.json", out)
    return seen, result, batch


def test_3n_default_run_packs_passes_before_pack_that_calls_check_model(tmp_path, monkeypatch):
    asked: list = []
    seen, result, batch = _wired(tmp_path, monkeypatch, lambda m, b: asked.append((m, b)) or "broken")
    assert callable(seen["hook"])
    assert seen["reason"] == "broken"
    assert asked == [(MODEL, batch)]                                     # (model, batch folder), as in the brief
    assert result["skipped"] == {"reason": "broken", "before": "privacy"}


def test_3o_a_check_that_raises_never_skips_and_the_text_is_kept(tmp_path, monkeypatch):
    def boom(m, b):
        raise RuntimeError("check exploded")

    seen, result, batch = _wired(tmp_path, monkeypatch, boom)
    assert callable(seen["hook"]) and seen["reason"] is None
    assert "skipped" not in result
    assert "check exploded" in result["check_error"]


# --- round 3, item 3: end to end with the REAL gate.main, run_packs and check_model ---------------------------------------

def test_3r_end_to_end_real_gate_real_run_packs_real_check_skips_trust_and_privacy(tmp_path, monkeypatch):
    c18a._build(tmp_path, lines=None)                                    # queue_t/ with the model's row, no pack logs yet
    batch = tmp_path / f"queue_{TAG}"
    out_dir = batch / "models" / mq._safe(MODEL) / "packs"
    monkeypatch.setenv("LOCALBOT_RUNLOCK_PATH", str(tmp_path / "run.lock"))
    monkeypatch.delenv(runlock.HELD_ENV, raising=False)
    monkeypatch.setattr(gate, "git_porcelain", lambda: "")
    monkeypatch.setattr(gate, "git_head", lambda: COMMIT)
    monkeypatch.setattr(gate, "gpu_visible", lambda: True)
    monkeypatch.setattr(gate, "ollama_tags", lambda: {MODEL: "d"})
    launched: list[str] = []

    def launch(argv, log_path, cwd):                                     # the only stand-in: no subprocess
        pack = argv[argv.index("--pack") + 1]
        launched.append(pack)
        result = tmp_path / "results" / f"{pack}.json"
        result.parent.mkdir(exist_ok=True)
        result.write_text(json.dumps({"run_env": {"git": {"commit": COMMIT}}, "tasks": {}, "total": 0, "wall_s": 1,
                                      "pack": pack, "model": MODEL}), encoding="utf-8", newline="\n")
        wrote = c18a._wrote(no_result=0).replace("C:\\x\\results\\stage1\\work_core4_m.json", str(result))
        body = [c18a._try(f"e_{i}", infra=True) for i in range(5)] + [wrote]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(body) + "\n", encoding="utf-8", newline="\n")
        return 0

    monkeypatch.setattr(gate, "launch", launch)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"rows": [
        {"gate": f"Q-{p}", "model": MODEL, "pack": p, "profile": "core4", "n": 3}
        for p in ("work", "trust", "privacy")]}), encoding="utf-8", newline="\n")

    env = mq.Env()                                                       # default run_packs, default check_model
    result = env.run_packs(MODEL, plan, out_dir)

    assert launched == ["work"]                                          # trust and privacy never started
    assert result["skipped"] == {"reason": "broken", "before": "trust"}
    assert "check_error" not in result
    journal = _journal(out_dir.parent, out_dir.name)
    by_pack = {r["pack"]: r for r in journal}
    assert by_pack["work"]["status"] == "ok"
    for pack in ("trust", "privacy"):
        assert by_pack[pack]["status"] == "SKIPPED" and by_pack[pack]["result"] == "skipped: broken"


# --- phase2_one -------------------------------------------------------------------------------------------------------

def _phase2(tmp_path: Path, fe: FakeEnv, check, model: str = "m:1b", parts=("pull", "show", "fit", "quick_check", "packs")):
    env = fe.as_env()
    env.check_model = check
    batch = tmp_path / "runs" / f"queue_{TAG}"
    row = {"model": model, "kind": "base", "status": "phase1_done", "quick_check": "native tool call",
           "parts_done": list(parts)}
    if "packs" in parts:
        row["packs"] = {"exit": 0, "scores": {"work": "3/3", "trust": "2/3"}, "first_try": {}, "hard": {}}
    state = {"meta": {}, "rows": {model: row}}
    out = mq.phase2_one(row, batch, env, fe.clock_value + 3600, state)
    return out, batch, state


@pytest.mark.parametrize("reason", ["broken", "broken (no result)"])
def test_4_broken_before_the_ladder_skips_the_ladder(tmp_path, reason):
    fe = FakeEnv()
    asked: list = []
    row, batch, state = _phase2(tmp_path, fe, lambda m, b: asked.append((m, b)) or reason)
    assert fe.ladder_log == []                                           # the ladder did not run
    assert row["status"] == "skipped: " + reason
    assert row["skipped"]["reason"] == reason and row["skipped"]["before"] == "ladder"
    when = datetime.fromisoformat(row["skipped"]["at"])
    assert when.utcoffset() == timedelta(0)                              # ISO time, UTC
    assert row["retest_next_batch"] is True
    assert row["packs"]["scores"]["work"] == "3/3" and "ladder" not in row["parts_done"]   # finished packs are kept
    assert asked and asked[-1] == ("m:1b", batch)
    on_disk = json.loads((batch / "models" / mq._safe("m:1b") / "row.json").read_text(encoding="utf-8"))
    assert on_disk["status"] == "skipped: " + reason and on_disk["retest_next_batch"] is True
    in_state = json.loads(mq.state_path(batch).read_text(encoding="utf-8"))["rows"]["m:1b"]
    assert in_state["status"] == "skipped: " + reason
    assert "m:1b" in fe.unload_log                                       # ensure_card_clear(env, model) ran


def test_4b_a_skip_inside_the_packs_result_stops_before_the_ladder(tmp_path):
    fe = FakeEnv()
    fe.packs_hook = lambda m, plan, out: {
        "exit": 0, "scores": {"work": "3/3"}, "first_try": {}, "hard": {},
        "skipped": {"reason": "broken", "before": "privacy"}}
    row, batch, state = _phase2(tmp_path, fe, lambda m, b: None, parts=("pull", "show", "fit", "quick_check"))
    assert fe.packs_log == ["m:1b"] and fe.ladder_log == ["m:1b"]        # C19: the ladder runs before the packs now
    assert row["status"] == "skipped: broken"
    assert row["skipped"]["before"] == "privacy" and row["retest_next_batch"] is True
    assert row["packs"]["scores"] == {"work": "3/3"}
    assert json.loads(mq.state_path(batch).read_text(encoding="utf-8"))["rows"]["m:1b"]["status"] == "skipped: broken"


def test_4c_check_returning_none_runs_the_ladder_as_before(tmp_path):
    fe = FakeEnv()
    row, batch, state = _phase2(tmp_path, fe, lambda m, b: None)
    assert fe.ladder_log == ["m:1b"] and row["status"] == "done"
    assert "skipped" not in row and "retest_next_batch" not in row


def test_4d_a_check_that_raises_does_not_skip_a_healthy_model(tmp_path):
    fe = FakeEnv()

    def boom(m, b):
        raise RuntimeError("check exploded")

    row, batch, state = _phase2(tmp_path, fe, boom)
    assert fe.ladder_log == ["m:1b"] and row["status"] == "done"
    assert "check exploded" in row["check_error"]


def test_4e_the_skip_line_is_one_ascii_line_naming_model_and_reason(tmp_path):
    fe = FakeEnv()
    name = "caf\u00e9/m\u00f6del:1b"
    _phase2(tmp_path, fe, lambda m, b: "broken", model=name)
    lines = [s for s in fe.said if "skipped" in s.lower()]
    assert len(lines) == 1
    lines[0].encode("ascii")
    assert "broken" in lines[0] and "ladder" in lines[0] and "caf" in lines[0]
    assert all(s.isascii() for s in fe.said)


# --- resume, --finish-capped, delete_lines -------------------------------------------------------------------------------

def _seed(tmp_path: Path, fe: FakeEnv, rows: dict) -> Path:
    batch = tmp_path / "runs" / "queue_res"
    mq.write_json_atomic(mq.state_path(batch), {
        "meta": {"commit": fe.head, "ollama_version": fe.version, "preexisting": [], "preexisting_set": [],
                 "models_dir": "x", "free_disk_gb": 100.0},
        "rows": rows, "history": []})
    return batch


def _skipped_row(model: str) -> dict:
    return {"model": model, "kind": "base", "status": "skipped: broken", "quick_check": "native tool call",
            "triage": "", "fits": "yes", "digest": "", "template_hash": "", "size_bytes": 0, "tools_listed": True,
            "parts_done": ["pull", "show", "fit", "quick_check", "packs"],
            "packs": {"exit": 0, "scores": {"work": "3/3"}, "first_try": {}, "hard": {}},
            "skipped": {"reason": "broken", "before": "ladder", "at": "2026-11-08T03:00:00+00:00"},
            "retest_next_batch": True}


def _phase1_row(model: str, status: str = "phase1_done") -> dict:
    return {"model": model, "kind": "base", "status": status, "quick_check": "native tool call", "triage": "",
            "fits": "yes", "digest": "", "template_hash": "", "size_bytes": 0, "tools_listed": True,
            "parts_done": ["pull", "show", "fit", "quick_check"]}


def test_5_resume_runs_only_the_row_that_is_not_skipped(tmp_path):
    b = write_batch(tmp_path / "b.txt", ["a:1b  base", "b:1b  base"])
    fe = FakeEnv()
    _seed(tmp_path, fe, {"a:1b": _skipped_row("a:1b"), "b:1b": _phase1_row("b:1b")})
    env = fe.as_env()
    env.check_model = lambda m, bt: None
    code = mq.main(["--batch", str(b), "--tag", "res"], env=env, runs_root=tmp_path / "runs")
    assert code == 0
    assert fe.packs_log == ["b:1b"] and fe.ladder_log == ["b:1b"]
    assert "a:1b" not in fe.pull_log                                     # not even downloaded / quick-checked again
    rows = json.loads(mq.state_path(tmp_path / "runs" / "queue_res").read_text(encoding="utf-8"))["rows"]
    assert rows["a:1b"]["status"] == "skipped: broken" and rows["a:1b"]["retest_next_batch"] is True
    assert rows["b:1b"]["status"] == "done"


def test_5b_finish_capped_does_not_pick_up_a_skipped_row(tmp_path):
    b = write_batch(tmp_path / "b.txt", ["a:1b  base", "c:1b  base"])
    fe = FakeEnv()
    capped = _phase1_row("c:1b", "capped")
    capped["capped_at"] = "before_packs"
    _seed(tmp_path, fe, {"a:1b": _skipped_row("a:1b"), "c:1b": capped})
    env = fe.as_env()
    env.check_model = lambda m, bt: None
    code = mq.main(["--batch", str(b), "--tag", "res", "--finish-capped"], env=env, runs_root=tmp_path / "runs")
    assert code == 0
    assert fe.packs_log == ["c:1b"] and "a:1b" not in fe.pull_log
    rows = json.loads(mq.state_path(tmp_path / "runs" / "queue_res").read_text(encoding="utf-8"))["rows"]
    assert rows["a:1b"]["status"] == "skipped: broken"


def test_5c_delete_lines_never_lists_a_skipped_model(tmp_path):
    rows = [_skipped_row("a:1b"), {"model": "b:1b", "status": "done"}]
    assert mq.delete_lines(rows, set()) == ["ollama rm b:1b"]


# --- two in a row ---------------------------------------------------------------------------------------------------------

def _three(tmp_path: Path, extra: list[str], broken, tag: str = "skip3"):
    b = write_batch(tmp_path / "b.txt", ["a:1b  base", "b:1b  base", "c:1b  base"])
    fe = FakeEnv()
    env = fe.as_env()
    env.check_model = lambda m, bt: "broken" if m in broken else None
    code = mq.main(["--batch", str(b), "--tag", tag, *extra], env=env, runs_root=tmp_path / "runs")
    rows = json.loads(mq.state_path(tmp_path / "runs" / f"queue_{tag}").read_text(encoding="utf-8"))["rows"]
    return code, fe, rows, b


def test_6_two_skips_in_a_row_stop_the_batch_by_default(tmp_path):
    code, fe, rows, b = _three(tmp_path, ["--stop-after-skips", "2"], {"a:1b", "b:1b", "c:1b"})   # C19: default is off
    assert rows["a:1b"]["status"] == "skipped: broken" and rows["b:1b"]["status"] == "skipped: broken"
    assert rows["c:1b"]["status"] == "phase1_done"                       # untouched, a later run resumes it
    assert "c:1b" not in fe.packs_log
    said = "\n".join(fe.said)
    assert "stopped: 2 models in a row were skipped as broken; check Ollama and the PC before resuming" in said
    assert (tmp_path / "runs" / "queue_skip3" / "summary.md").is_file()  # the summary is still written
    assert code == mq.EXIT_STOPPED_AFTER_SKIPS
    assert code not in (0, 2, 3)                                         # 0 ok, 2 STOP/usage, 3 busy are taken


def test_6b_stop_after_skips_zero_turns_it_off(tmp_path):
    code, fe, rows, b = _three(tmp_path, ["--stop-after-skips", "0"], {"a:1b", "b:1b", "c:1b"})
    assert [rows[m]["status"] for m in ("a:1b", "b:1b", "c:1b")] == ["skipped: broken"] * 3
    assert not any("stopped:" in s for s in fe.said)
    assert code == 0


def test_6c_skip_done_skip_does_not_stop(tmp_path):
    code, fe, rows, b = _three(tmp_path, [], {"a:1b", "c:1b"})
    assert [rows[m]["status"] for m in ("a:1b", "b:1b", "c:1b")] == ["skipped: broken", "done", "skipped: broken"]
    assert not any("stopped:" in s for s in fe.said)
    assert code == 0


def test_6d_the_limit_is_a_parameter(tmp_path):
    code, fe, rows, b = _three(tmp_path, ["--stop-after-skips", "1"], {"a:1b", "b:1b", "c:1b"})
    assert rows["a:1b"]["status"] == "skipped: broken"
    assert rows["b:1b"]["status"] == "phase1_done" and rows["c:1b"]["status"] == "phase1_done"
    assert code == mq.EXIT_STOPPED_AFTER_SKIPS


def test_6e_a_later_run_resumes_the_rows_the_stop_left_alone(tmp_path):
    code, fe, rows, b = _three(tmp_path, ["--stop-after-skips", "2"], {"a:1b", "b:1b", "c:1b"})   # C19: default is off
    assert code == mq.EXIT_STOPPED_AFTER_SKIPS
    fe2 = FakeEnv(head=fe.head, version=fe.version)
    env2 = fe2.as_env()
    env2.check_model = lambda m, bt: None
    code2 = mq.main(["--batch", str(b), "--tag", "skip3"], env=env2, runs_root=tmp_path / "runs")
    assert code2 == 0
    assert fe2.packs_log == ["c:1b"]                                     # a and b stay skipped, c runs
    rows2 = json.loads(mq.state_path(tmp_path / "runs" / "queue_skip3").read_text(encoding="utf-8"))["rows"]
    assert rows2["a:1b"]["status"] == "skipped: broken" and rows2["b:1b"]["status"] == "skipped: broken"
    assert rows2["c:1b"]["status"] == "done"


def test_6f_every_printed_line_of_a_stopped_batch_is_ascii(tmp_path, capsys):
    code, fe, rows, b = _three(tmp_path, [], {"a:1b", "b:1b", "c:1b"})
    assert all(s.isascii() for s in fe.said)
    capsys.readouterr().out.encode("ascii")


# --- summary and queue_status ---------------------------------------------------------------------------------------------

def test_7_summary_lists_a_skipped_model_under_its_own_heading(tmp_path):
    batch = tmp_path / "queue_s"
    batch.mkdir()
    done = {"model": "ok:1b", "kind": "base", "status": "done", "packs": {"scores": {"work": "3/3"}}, "ladder": {}}
    rows = [done, _skipped_row("bad:1b")]
    mq.write_summary(batch, rows, {"ollama_version": "v", "commit": "c"})
    text = (batch / "summary.md").read_text(encoding="utf-8")
    head = "Skipped: retest in next batch"
    assert head in text
    section = text[text.index(head):]
    assert "bad:1b" in section and "broken" in section and "ladder" in section
    assert "partial" in section.lower() and "3/3" in section
    assert "ok:1b" not in section                                        # only skipped models are listed there
    assert "skipped: broken" in (batch / "summary.csv").read_text(encoding="utf-8")


def test_7b_summary_without_a_skipped_model_has_no_such_heading(tmp_path):
    batch = tmp_path / "queue_s"
    batch.mkdir()
    mq.write_summary(batch, [{"model": "ok:1b", "kind": "base", "status": "done", "packs": {}, "ladder": {}}], {})
    assert "Skipped: retest in next batch" not in (batch / "summary.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("status", ["skipped: broken", "skipped: broken (no result)"])
def test_7c_queue_status_shows_step_skipped_and_no_flags(tmp_path, status):
    lines = [c18a._try(f"e_{i}", infra=True) for i in range(5)]          # would be BROKEN if the step were 'full tests'
    c18a._build(tmp_path, lines=lines, status=status)
    rec = c18a._one(tmp_path)
    assert rec["step"] == "skipped" and rec["flags"] == []


def test_7d_a_skipped_model_does_not_keep_the_status_exit_code_at_1(tmp_path, capsys):
    lines = [c18a._try(f"e_{i}", infra=True) for i in range(5)]
    c18a._build(tmp_path, lines=lines, status="skipped: broken")
    code = qs.main(["--tag", TAG, "--runs-dir", str(tmp_path)], ps_fn=lambda: [], now=c18a.NOW)
    assert code == 0
    assert "step: skipped" in capsys.readouterr().out
