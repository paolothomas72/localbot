# tests/test_c9_model_queue.py — C9 overnight model queue (failing-first groups 1-9 + edges)
from __future__ import annotations

import os

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import scripts.model_queue as mq
from scripts.chat_ladder import FAKE_KEY


class FakeLock:
    def release(self) -> None:
        return None


@dataclass
class FakeEnv:
    """Injectable Env stand-in: no real Ollama / GPU / git / subprocesses."""
    tags: dict[str, str] = field(default_factory=dict)
    porcelain: str = ""
    head: str = "abc1234deadbeef"
    gpu_ok: bool = True
    free_gb: float = 100.0
    version: str = "0.9.9-fake"
    pulls: dict[str, tuple[bool, str]] = field(default_factory=dict)
    shows: dict[str, dict] = field(default_factory=dict)
    qc: dict[str, dict] = field(default_factory=dict)
    fit_override: dict[str, dict] = field(default_factory=dict)
    card: list[str] = field(default_factory=list)  # models currently "on the card"
    deleted: list[str] = field(default_factory=list)
    pull_log: list[str] = field(default_factory=list)
    unload_log: list[str] = field(default_factory=list)
    ps_log: list[list[str]] = field(default_factory=list)
    packs_log: list[str] = field(default_factory=list)
    ladder_log: list[str] = field(default_factory=list)
    said: list[str] = field(default_factory=list)
    clock_value: float = 1_000_000.0
    packs_hook: Any = None  # optional callable(model)->dict; may mutate clock / raise stop
    ladder_hook: Any = None
    max_card: int = 0  # max concurrent models observed on card during tests/unload

    def as_env(self) -> mq.Env:
        fe = self

        def ollama_tags():
            return dict(fe.tags)

        def pull(m: str):
            fe.pull_log.append(m)
            return fe.pulls.get(m, (True, "ok"))

        def show(m: str):
            return dict(fe.shows.get(m) or {
                "template": "{{ .Tools }} tools here",
                "modelfile": "TEMPLATE {{ .Tools }}",
                "capabilities": ["tools"],
                "size": 2 * 1024**3,
                "details": {"family": "fake", "parameter_size": "3B", "quantization_level": "Q4"},
            })

        def ps():
            snap = list(fe.card)
            fe.ps_log.append(snap)
            fe.max_card = max(fe.max_card, len(snap))
            return [{"name": n, "size_vram": 1} for n in snap]

        def unload(m: str):
            fe.unload_log.append(m)
            if m in fe.card:
                fe.card.remove(m)
            # clear all if ensure_card_clear loops
            fe.card.clear()

        def quick_check(m: str):
            # one model on card during check
            if m not in fe.card:
                fe.card.append(m)
            fe.max_card = max(fe.max_card, len(fe.card))
            return dict(fe.qc.get(m) or {"result": "native tool call", "native": True, "text_tool": False})

        def run_packs(m: str, plan: Path, out: Path):
            fe.packs_log.append(m)
            if m not in fe.card:
                fe.card.append(m)
            fe.max_card = max(fe.max_card, len(fe.card))
            out.mkdir(parents=True, exist_ok=True)
            if fe.packs_hook:
                return fe.packs_hook(m, plan, out)
            return {
                "exit": 0,
                "scores": {"work": "3/3", "trust": "3/3", "privacy": "3/3", "injection": "3/3", "long": "3/3"},
                "first_try": {"work": "1/1", "trust": "1/1", "privacy": "1/1", "injection": "1/1", "long": "1/1"},
                "hard": {"outside_box": 0, "secret_file": 0, "git_folder_changed": 0, "injected_executed": 0, "planted_leaks": 0},
            }

        def run_ladder(m: str, out: Path):
            fe.ladder_log.append(m)
            if m not in fe.card:
                fe.card.append(m)
            fe.max_card = max(fe.max_card, len(fe.card))
            out.mkdir(parents=True, exist_ok=True)
            if fe.ladder_hook:
                return fe.ladder_hook(m, out)
            return {"exit": 0, "best": "ctx=8192", "card_limit": "16384"}

        def delete_model(m: str):
            fe.deleted.append(m)
            fe.tags.pop(m, None)

        def say(s: str):
            fe.said.append(str(s))

        return mq.Env(
            ollama_tags=ollama_tags,
            git_porcelain=lambda: fe.porcelain,
            git_head=lambda: fe.head,
            gpu_visible=lambda: fe.gpu_ok,
            acquire_lock=lambda: FakeLock(),
            free_disk_gb=lambda: fe.free_gb,
            ollama_version=lambda: fe.version,
            pull=pull,
            show=show,
            ps=ps,
            unload=unload,
            read_vram=lambda: None,
            quick_check=quick_check,
            run_packs=run_packs,
            run_ladder=run_ladder,
            delete_model=delete_model,
            clock=lambda: fe.clock_value,
            sleep=lambda _s: None,
            say=say,
        )


def write_batch(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path


# --- 1. --list -----------------------------------------------------------------------


def test_1_list_changes_nothing_and_prints_plan(tmp_path: Path):
    batch = write_batch(tmp_path / "b.txt", [
        "# popular",
        "llama3.2:3b  base",
        "hf.co/openbmb/MiniCPM5-2B-GGUF:Q4_K_M  hf.co",
    ])
    fe = FakeEnv(tags={"keep:me": "d1"})
    runs = tmp_path / "runs"
    code = mq.main(
        ["--batch", str(batch), "--tag", "list1", "--list"],
        env=fe.as_env(),
        runs_root=runs,
    )
    assert code == 0
    assert not (runs / "queue_list1").exists()
    assert fe.pull_log == []
    assert fe.deleted == []
    joined = "\n".join(fe.said)
    assert "1. llama3.2:3b  (base)" in joined
    assert "2. hf.co/openbmb/MiniCPM5-2B-GGUF:Q4_K_M  (hf.co)" in joined
    assert "sizes unknown" in joined


def test_1b_batch1_file_has_25_models():
    path = Path("docs/batches/batch1.txt")
    if not path.is_file():
        # Public export allowlist does not yet include docs/batches/ (ask to add it).
        path = mq.REPO / "docs/batches/batch1.txt"
    if not path.is_file():
        pytest.skip("docs/batches/batch1.txt not in this tree (export allowlist)")
    models = mq.parse_batch(path)
    assert len(models) == 25
    assert models[0].model == "llama3.2:3b" and models[0].kind == "base"
    assert models[-1].kind == "hf.co"


# --- 2. Phase order ------------------------------------------------------------------


def test_2_passers_full_tests_before_failures_and_triage(tmp_path: Path):
    batch = write_batch(tmp_path / "b.txt", [
        "fail-a:1b  base",
        "pass-b:1b  base",
        "fail-c:1b  base",
    ])
    fe = FakeEnv(tags={"pre:keep": "x"})
    fe.qc = {
        "fail-a:1b": {"result": "no tool call"},
        "pass-b:1b": {"result": "native tool call"},
        "fail-c:1b": {"result": "tool call as text"},
    }
    code = mq.main(
        ["--batch", str(batch), "--tag", "ord2"],
        env=fe.as_env(),
        runs_root=tmp_path / "runs",
    )
    assert code == 0
    # full tests order: passer first, then failures (still run)
    assert fe.packs_log == ["pass-b:1b", "fail-a:1b", "fail-c:1b"]
    state = json.loads((tmp_path / "runs" / "queue_ord2" / "state.json").read_text(encoding="utf-8"))
    assert state["rows"]["fail-a:1b"]["triage"].startswith("triage:")
    assert state["rows"]["fail-c:1b"]["status"] == "done"
    assert state["rows"]["pass-b:1b"]["status"] == "done"


# --- 3. Quick check classification ---------------------------------------------------


@pytest.mark.parametrize("result,kind,tool_sec,expect_qc,expect_triage_part", [
    ("native tool call", "base", True, "native tool call", ""),
    ("tool call as text", "base", True, "tool call as text", "triage:"),
    ("no tool call", "base", True, "no tool call", "triage:"),
    ("error/timeout", "base", True, "error/timeout", "triage:"),
    ("tool call as text", "hf.co", True, "tool call as text", "harness or pull problem"),
    ("native tool call", "base", False, "native tool call", "no tool section"),
])
def test_3_quick_check_labels(tmp_path, result, kind, tool_sec, expect_qc, expect_triage_part):
    model = "hf.co/x/y:Q4" if kind == "hf.co" else "m:1b"
    batch = write_batch(tmp_path / "b.txt", [f"{model}  {kind}"])
    fe = FakeEnv()
    fe.qc[model] = {"result": result}
    tmpl = "{{ .Tools }}" if tool_sec else "no tools here"
    fe.shows[model] = {
        "template": tmpl, "modelfile": tmpl, "capabilities": ["tools"] if tool_sec else [],
        "size": 1024**3, "details": {},
    }
    code = mq.main(["--batch", str(batch), "--tag", "qc3"], env=fe.as_env(), runs_root=tmp_path / "runs")
    assert code == 0
    row = json.loads((tmp_path / "runs" / "queue_qc3" / "state.json").read_text(encoding="utf-8"))["rows"][model]
    assert row["quick_check"] == expect_qc
    if expect_triage_part:
        assert expect_triage_part in (row["triage"] or "")
    else:
        assert row["triage"] == ""


# --- 4. Pull fail / not fit / low disk -----------------------------------------------


def test_4_pull_failure_continues_next(tmp_path: Path):
    batch = write_batch(tmp_path / "b.txt", ["bad:1b  base", "good:1b  base"])
    fe = FakeEnv()
    fe.pulls["bad:1b"] = (False, "needs newer Ollama")
    fe.qc["good:1b"] = {"result": "native tool call"}
    code = mq.main(["--batch", str(batch), "--tag", "pull4"], env=fe.as_env(), runs_root=tmp_path / "runs")
    assert code == 0
    rows = json.loads((tmp_path / "runs" / "queue_pull4" / "state.json").read_text(encoding="utf-8"))["rows"]
    assert rows["bad:1b"]["status"] == "could not download"
    assert rows["good:1b"]["status"] == "done"
    assert "bad:1b" not in fe.packs_log


def test_4_not_fitting_stage2_no_tests(tmp_path: Path, monkeypatch):
    batch = write_batch(tmp_path / "b.txt", ["huge:1b  base"])
    fe = FakeEnv()

    def boom_fit(model, show, env):
        return {"fits": False, "lane": "spill", "reason": "too big", "size_bytes": 20 * 1024**3}

    monkeypatch.setattr(mq, "fit_at_8192", boom_fit)
    code = mq.main(["--batch", str(batch), "--tag", "fit4"], env=fe.as_env(), runs_root=tmp_path / "runs")
    assert code == 0
    row = json.loads((tmp_path / "runs" / "queue_fit4" / "state.json").read_text(encoding="utf-8"))["rows"]["huge:1b"]
    assert row["status"] == "over the card: Stage 2"
    assert fe.packs_log == []
    assert fe.ladder_log == []


def test_4_low_disk_plain_stop(tmp_path: Path):
    batch = write_batch(tmp_path / "b.txt", ["m:1b  base"])
    fe = FakeEnv(free_gb=10.0)
    code = mq.main(["--batch", str(batch), "--tag", "disk4"], env=fe.as_env(), runs_root=tmp_path / "runs")
    assert code != 0
    assert fe.pull_log == []


# --- 5. Resume -----------------------------------------------------------------------


def test_5_resume_after_phase1_model3(tmp_path: Path):
    batch = write_batch(tmp_path / "b.txt", ["a:1b  base", "b:1b  base", "c:1b  base", "d:1b  base"])
    fe = FakeEnv()
    for m in ("a:1b", "b:1b", "c:1b", "d:1b"):
        fe.qc[m] = {"result": "native tool call"}

    # Kill after phase1 of model 3: packs_hook raises after 0 packs once 3 phase1 done
    stop = {"n": 0}

    def packs_hook(m, plan, out):
        stop["n"] += 1
        if stop["n"] == 1:
            raise RuntimeError("killed mid-batch")
        return {
            "exit": 0,
            "scores": {"work": "3/3", "trust": "3/3", "privacy": "3/3", "injection": "3/3", "long": "3/3"},
            "first_try": {}, "hard": {"outside_box": 0, "secret_file": 0, "git_folder_changed": 0, "injected_executed": 0, "planted_leaks": 0},
        }

    fe.packs_hook = packs_hook
    with pytest.raises(RuntimeError):
        mq.main(["--batch", str(batch), "--tag", "res5"], env=fe.as_env(), runs_root=tmp_path / "runs")

    # Resume with fresh FakeEnv sharing tag state on disk
    fe2 = FakeEnv()
    for m in ("a:1b", "b:1b", "c:1b", "d:1b"):
        fe2.qc[m] = {"result": "native tool call"}
    fe2.head = fe.head
    fe2.version = fe.version
    code = mq.main(["--batch", str(batch), "--tag", "res5"], env=fe2.as_env(), runs_root=tmp_path / "runs")
    assert code == 0
    # Phase1 should not re-pull all four from scratch if state saved — at least packs finish
    rows = json.loads((tmp_path / "runs" / "queue_res5" / "state.json").read_text(encoding="utf-8"))["rows"]
    assert all(rows[m]["status"] == "done" for m in ("a:1b", "b:1b", "c:1b", "d:1b"))
    # On resume, phase1 finished models should not re-run quick_check pulls for a,b,c if saved
    # (d may still need phase1 if kill was at first packs of ordered passers)


def test_5_resume_mid_packs_does_not_rerun_finished_part(tmp_path: Path):
    batch = write_batch(tmp_path / "b.txt", ["solo:1b  base"])
    fe = FakeEnv()
    fe.qc["solo:1b"] = {"result": "native tool call"}
    calls = {"packs": 0, "ladder": 0}

    def packs_hook(m, plan, out):
        calls["packs"] += 1
        if calls["packs"] == 1:
            # C19: the ladder runs first, so the kill comes in the packs, after the ladder was saved
            raise RuntimeError("killed mid-packs")
        return {"exit": 0, "scores": {}, "first_try": {}, "hard": {}}

    def ladder_hook(m, out):
        calls["ladder"] += 1
        return {"exit": 0, "best": "ok", "card_limit": "8k"}

    fe.packs_hook = packs_hook
    fe.ladder_hook = ladder_hook
    with pytest.raises(RuntimeError):
        mq.main(["--batch", str(batch), "--tag", "mid5"], env=fe.as_env(), runs_root=tmp_path / "runs")

    # After kill, the ladder should be in parts_done
    row = json.loads((tmp_path / "runs" / "queue_mid5" / "models" / "solo_1b" / "row.json").read_text(encoding="utf-8"))
    assert "ladder" in row["parts_done"]
    assert "packs" not in row["parts_done"]

    fe2 = FakeEnv(head=fe.head, version=fe.version)
    fe2.qc["solo:1b"] = {"result": "native tool call"}
    packs2 = {"n": 0}

    def ladder2_hook(m, out):
        packs2["n"] += 1
        return {"exit": 0, "best": "ok", "card_limit": "8k"}

    fe2.ladder_hook = ladder2_hook
    code = mq.main(["--batch", str(batch), "--tag", "mid5"], env=fe2.as_env(), runs_root=tmp_path / "runs")
    assert code == 0
    assert packs2["n"] == 0  # the finished ladder is not re-run
    row2 = json.loads((tmp_path / "runs" / "queue_mid5" / "state.json").read_text(encoding="utf-8"))["rows"]["solo:1b"]
    assert row2["status"] == "done"
    assert "ladder" in row2["parts_done"]


def test_5_commit_change_refused_without_allow(tmp_path: Path):
    batch = write_batch(tmp_path / "b.txt", ["m:1b  base"])
    fe = FakeEnv()
    fe.qc["m:1b"] = {"result": "native tool call"}
    assert mq.main(["--batch", str(batch), "--tag", "chg5"], env=fe.as_env(), runs_root=tmp_path / "runs") == 0
    fe2 = FakeEnv(head="ffffff_different", version=fe.version)
    fe2.qc["m:1b"] = {"result": "native tool call"}
    code = mq.main(["--batch", str(batch), "--tag", "chg5"], env=fe2.as_env(), runs_root=tmp_path / "runs")
    assert code != 0


def test_5_ollama_change_refused_without_allow(tmp_path: Path):
    batch = write_batch(tmp_path / "b.txt", ["m:1b  base"])
    fe = FakeEnv()
    fe.qc["m:1b"] = {"result": "native tool call"}
    assert mq.main(["--batch", str(batch), "--tag", "olv5"], env=fe.as_env(), runs_root=tmp_path / "runs") == 0
    fe2 = FakeEnv(head=fe.head, version="1.0.0-new")
    code = mq.main(["--batch", str(batch), "--tag", "olv5"], env=fe2.as_env(), runs_root=tmp_path / "runs")
    assert code != 0


def test_5_allow_change_continues(tmp_path: Path):
    batch = write_batch(tmp_path / "b.txt", ["m:1b  base"])
    fe = FakeEnv()
    fe.qc["m:1b"] = {"result": "native tool call"}
    # Cap first run so resume has work
    fe.clock_value = 1_000_000.0
    fe.packs_hook = lambda m, p, o: (_ for _ in ()).throw(RuntimeError("stop"))
    with pytest.raises(RuntimeError):
        mq.main(["--batch", str(batch), "--tag", "all5", "--model-hours", "8"],
                env=fe.as_env(), runs_root=tmp_path / "runs")
    fe2 = FakeEnv(head="newcommit999", version="new-ollama")
    fe2.qc["m:1b"] = {"result": "native tool call"}
    code = mq.main(
        ["--batch", str(batch), "--tag", "all5", "--allow-change"],
        env=fe2.as_env(), runs_root=tmp_path / "runs",
    )
    assert code == 0


# --- 6. Never two on card ------------------------------------------------------------


def test_6_never_two_models_on_card_and_unload_before_tests(tmp_path: Path):
    batch = write_batch(tmp_path / "b.txt", ["a:1b  base", "b:1b  base"])
    fe = FakeEnv()
    for m in ("a:1b", "b:1b"):
        fe.qc[m] = {"result": "native tool call"}
    code = mq.main(["--batch", str(batch), "--tag", "card6"], env=fe.as_env(), runs_root=tmp_path / "runs")
    assert code == 0
    assert fe.max_card <= 1
    assert fe.unload_log  # unload happened
    assert fe.ps_log  # /api/ps checked


# --- 7. Delete rules -----------------------------------------------------------------


def test_7_delete_print_only_without_flag(tmp_path: Path):
    gate = next(iter(mq.GATE_MODELS))
    batch = write_batch(tmp_path / "b.txt", [
        "newdone:1b  base",
        f"{gate}  base",
        "preexist:1b  base",
    ])
    fe = FakeEnv(tags={"preexist:1b": "digest-pre"})
    for m in ("newdone:1b", gate, "preexist:1b"):
        fe.qc[m] = {"result": "native tool call"}
    code = mq.main(["--batch", str(batch), "--tag", "del7"], env=fe.as_env(), runs_root=tmp_path / "runs")
    assert code == 0
    assert fe.deleted == []
    joined = "\n".join(fe.said)
    assert "ollama rm newdone:1b" in joined
    assert f"ollama rm {gate}" not in joined
    assert "ollama rm preexist:1b" not in joined


def test_7_delete_after_only_done_batch_not_protected(tmp_path: Path):
    gate = "llama3.2:1b"
    assert gate in mq.GATE_MODELS
    batch = write_batch(tmp_path / "b.txt", [
        "newdone:1b  base",
        f"{gate}  base",
        "preexist:1b  base",
        "bad:1b  base",
    ])
    fe = FakeEnv(tags={"preexist:1b": "digest-pre", gate: "g"})
    fe.pulls["bad:1b"] = (False, "nope")
    for m in ("newdone:1b", gate, "preexist:1b"):
        fe.qc[m] = {"result": "native tool call"}
    code = mq.main(
        ["--batch", str(batch), "--tag", "del7b", "--delete-after"],
        env=fe.as_env(), runs_root=tmp_path / "runs",
    )
    assert code == 0
    assert fe.deleted == ["newdone:1b"]
    assert gate not in fe.deleted
    assert "preexist:1b" not in fe.deleted
    assert "bad:1b" not in fe.deleted


# --- 8. Time cap / finish-capped -----------------------------------------------------


def test_8_time_cap_and_finish_capped(tmp_path: Path):
    batch = write_batch(tmp_path / "b.txt", ["slow:1b  base", "ok:1b  base"])
    fe = FakeEnv()
    for m in ("slow:1b", "ok:1b"):
        fe.qc[m] = {"result": "native tool call"}
    # First model hits cap before packs: clock jumps past deadline inside packs_hook for slow
    def packs_hook(m, plan, out):
        if m == "slow:1b":
            # phase2_one checks clock before packs; set clock past deadline for slow by using hook that
            # never runs because we bump clock before phase2 for slow via monkey... Instead: set
            # clock high so first model caps before packs.
            return {"exit": 0, "scores": {}, "first_try": {}, "hard": {}}
        return {"exit": 0, "scores": {"work": "1/1"}, "first_try": {}, "hard": {}}

    # Cap: clock already past any deadline when phase2 starts for first ordered model.
    # Use model-hours very small and bump clock inside phase1 by wrapping quick_check.
    base_clock = [1_000_000.0]

    def qc(m):
        if m not in fe.card:
            fe.card.append(m)
        # After both phase1, advance clock hugely before phase2 by setting fe.clock_value in last qc
        if m == "ok:1b":
            fe.clock_value = base_clock[0] + 100_000  # way past 0.001 hours
        return {"result": "native tool call"}

    fe2 = FakeEnv()
    fe2.quick_check = None  # use as_env's; override via fe.qc + clock after phase1
    # Simpler approach: model-hours=0.0001 and clock jumps at start of phase2 via packs for first
    fe = FakeEnv()
    fe.qc["slow:1b"] = {"result": "native tool call"}
    fe.qc["ok:1b"] = {"result": "native tool call"}
    # Make clock return a value that exceeds deadline immediately in phase2_one
    # deadline = clock() + hours*3600; if clock is monotonic and we freeze it, never caps.
    # Trick: clock returns rising value; first call in phase2_one for packs check is already over.
    # phase2_one: if env.clock() > model_deadline where deadline = env.clock() + hours*3600 at call site.
    # So at call site deadline is always clock+hours. Cap only if clock moves during work.
    # Cap before packs: set clock to jump between deadline assignment and packs check.
    ticks = {"n": 0}
    real_clock = [1_000_000.0]

    def clock():
        ticks["n"] += 1
        # After deadline assignment (one clock call) jump far for the packs check
        if ticks["n"] > 1:
            return real_clock[0] + 10_000_000
        return real_clock[0]

    env = fe.as_env()
    env.clock = clock
    # Only first phase2 model should cap; but clock stays high so ALL cap. Use finish-capped later.
    # Better: only jump once for first model.
    state = {"phase2": 0}
    def clock2():
        # run_batch: deadline = env.clock() + hours*3600  (call A)
        # phase2_one: if env.clock() > model_deadline (call B)
        # For first model, make call B larger; for later models keep stable after reset
        state["phase2"] += 1
        # calls: many from phase1? phase1 doesn't use clock. Only phase2.
        # First deadline assign: state=1 -> t0; first packs check state=2 -> t0+big (cap)
        # Second deadline assign state=3 -> t0; second packs check state=4 -> t0 (ok) if we only add once
        if state["phase2"] == 2:
            return real_clock[0] + 9_999_999
        return real_clock[0]

    env.clock = clock2
    code = mq.main(
        ["--batch", str(batch), "--tag", "cap8", "--model-hours", "1"],
        env=env, runs_root=tmp_path / "runs",
    )
    assert code == 0
    rows = json.loads((tmp_path / "runs" / "queue_cap8" / "state.json").read_text(encoding="utf-8"))["rows"]
    # At least one capped
    statuses = {k: v["status"] for k, v in rows.items()}
    assert "capped" in statuses.values() or any(v.get("capped_at") for v in rows.values())

    # finish-capped finishes only capped
    fe3 = FakeEnv(head=fe.head, version=fe.version)
    fe3.qc["slow:1b"] = {"result": "native tool call"}
    fe3.qc["ok:1b"] = {"result": "native tool call"}
    code = mq.main(
        ["--batch", str(batch), "--tag", "cap8", "--finish-capped"],
        env=fe3.as_env(), runs_root=tmp_path / "runs",
    )
    assert code == 0
    rows2 = json.loads((tmp_path / "runs" / "queue_cap8" / "state.json").read_text(encoding="utf-8"))["rows"]
    assert all(r["status"] == "done" for r in rows2.values())


# --- 9. Fake key scrub ---------------------------------------------------------------


def test_9_fake_key_nowhere_in_queue_outputs(tmp_path: Path):
    batch = write_batch(tmp_path / "b.txt", ["m:1b  base"])
    fe = FakeEnv()
    fe.qc["m:1b"] = {"result": "native tool call", "detail": f"saw {FAKE_KEY}"}
    fe.packs_hook = lambda m, p, o: {
        "exit": 0,
        "scores": {"work": f"leak {FAKE_KEY}"},
        "first_try": {},
        "hard": {"outside_box": 0, "secret_file": 0, "git_folder_changed": 0, "injected_executed": 0, "planted_leaks": 0},
        "note": FAKE_KEY,
    }
    fe.ladder_hook = lambda m, o: {"exit": 0, "best": FAKE_KEY, "card_limit": "x"}
    code = mq.main(["--batch", str(batch), "--tag", "key9"], env=fe.as_env(), runs_root=tmp_path / "runs")
    assert code == 0
    root = tmp_path / "runs" / "queue_key9"
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".csv", ".md", ".json", ".txt", ".log"}:
            # row.json may hold packs note — brief: summary / screen / records the queue writes itself
            # summary.csv/md must be clean; screen said must be clean
            if path.name in {"summary.csv", "summary.md"}:
                text = path.read_text(encoding="utf-8")
                assert FAKE_KEY not in text
                assert "ladder-fake-key" not in text
    for line in fe.said:
        assert FAKE_KEY not in line


# --- Edges ---------------------------------------------------------------------------


def test_edge_gate_model_in_batch_never_deleted(tmp_path: Path):
    gate = "nemotron-3-nano:4b"
    batch = write_batch(tmp_path / "b.txt", [f"{gate}  base", "other:1b  base"])
    fe = FakeEnv(tags={})  # gate not preexisting, but still protected
    fe.qc[gate] = {"result": "native tool call"}
    fe.qc["other:1b"] = {"result": "native tool call"}
    mq.main(["--batch", str(batch), "--tag", "eg", "--delete-after"], env=fe.as_env(), runs_root=tmp_path / "runs")
    assert gate not in fe.deleted
    assert "other:1b" in fe.deleted


def test_edge_last_model_of_batch(tmp_path: Path):
    batch = write_batch(tmp_path / "b.txt", ["a:1b  base", "b:1b  base", "last:1b  base"])
    fe = FakeEnv()
    for m in ("a:1b", "b:1b", "last:1b"):
        fe.qc[m] = {"result": "native tool call"}
    code = mq.main(["--batch", str(batch), "--tag", "last"], env=fe.as_env(), runs_root=tmp_path / "runs")
    assert code == 0
    rows = json.loads((tmp_path / "runs" / "queue_last" / "state.json").read_text(encoding="utf-8"))["rows"]
    assert rows["last:1b"]["status"] == "done"
    assert "last:1b" in fe.packs_log


def test_edge_resume_at_very_last_part(tmp_path: Path):
    batch = write_batch(tmp_path / "b.txt", ["solo:1b  base"])
    fe = FakeEnv()
    fe.qc["solo:1b"] = {"result": "native tool call"}
    fe.packs_hook = lambda m, p, o: (_ for _ in ()).throw(RuntimeError("kill at last part"))   # C19: packs follow the ladder
    with pytest.raises(RuntimeError):
        mq.main(["--batch", str(batch), "--tag", "lastpart"], env=fe.as_env(), runs_root=tmp_path / "runs")
    fe2 = FakeEnv(head=fe.head, version=fe.version)
    fe2.qc["solo:1b"] = {"result": "native tool call"}
    packs_n = {"n": 0}
    fe2.ladder_hook = lambda m, o: packs_n.__setitem__("n", packs_n["n"] + 1) or {"exit": 0, "best": "ok", "card_limit": "8k"}
    code = mq.main(["--batch", str(batch), "--tag", "lastpart"], env=fe2.as_env(), runs_root=tmp_path / "runs")
    assert code == 0
    assert packs_n["n"] == 0                                             # the finished ladder is not re-run
    row = json.loads((tmp_path / "runs" / "queue_lastpart" / "state.json").read_text(encoding="utf-8"))["rows"]["solo:1b"]
    assert row["status"] == "done"


def test_gate_models_match_gate_plan_names():
    # Brief's 6 gate models must be exact Ollama names from gate_plan.json
    text = Path("docs/gate_plan.json").read_text(encoding="utf-8")
    for name in mq.GATE_MODELS:
        assert name in text, name


def test_stage1_sign_off_long_is_core4_product():
    long_rows = [r for r in mq.STAGE1_SIGN_OFF if r["pack"] == "long"]
    assert long_rows and long_rows[0]["profile"] == "core4_product"
    assert all(r["n"] == 3 for r in mq.STAGE1_SIGN_OFF)


# --- Audit edges / production boundaries (C9 hard-critic) ---------------------------------



def test_edge_delete_skips_capped_error_over_card(tmp_path: Path):
    """Delete list is only status==done; capped/error/over-card stay installed."""
    rows = [
        {"model": "done:1b", "status": "done"},
        {"model": "capped:1b", "status": "capped"},
        {"model": "err:1b", "status": "error"},
        {"model": "over:1b", "status": "over the card: Stage 2"},
        {"model": "failpull:1b", "status": "could not download"},
    ]
    lines = mq.delete_lines(rows, protected=set())
    assert lines == ["ollama rm done:1b"]


def test_edge_preexisting_snapshot_persists_across_reboot(tmp_path: Path):
    """never-delete snapshot is the start-of-batch list, not a live re-read after reboot."""
    batch = write_batch(tmp_path / "b.txt", ["new:1b  base", "pre:1b  base"])
    fe = FakeEnv(tags={"pre:1b": "d0"})
    fe.qc["new:1b"] = {"result": "native tool call"}
    fe.qc["pre:1b"] = {"result": "native tool call"}
    assert mq.main(["--batch", str(batch), "--tag", "snap"], env=fe.as_env(), runs_root=tmp_path / "runs") == 0
    state = json.loads((tmp_path / "runs" / "queue_snap" / "state.json").read_text(encoding="utf-8"))
    snap = set(state["meta"]["preexisting_set"])
    assert "pre:1b" in snap
    # Reboot: live tags no longer include pre:1b, but delete must still protect it
    fe2 = FakeEnv(tags={"new:1b": "d1"}, head=fe.head, version=fe.version)
    fe2.qc["new:1b"] = {"result": "native tool call"}
    fe2.qc["pre:1b"] = {"result": "native tool call"}
    mq.main(["--batch", str(batch), "--tag", "snap", "--delete-after"], env=fe2.as_env(), runs_root=tmp_path / "runs")
    assert "pre:1b" not in fe2.deleted
    assert "new:1b" in fe2.deleted


def test_edge_gate_canonical_names_not_aliases(tmp_path: Path):
    """Gate protection uses exact Ollama ids from gate_plan, not short aliases."""
    alias = "llama3.1:8b"  # brief nickname — NOT the pinned id
    canonical = "llama3.1:8b-instruct-q8_0"
    assert canonical in mq.GATE_MODELS
    assert alias not in mq.GATE_MODELS
    batch = write_batch(tmp_path / "b.txt", [f"{canonical}  base", f"{alias}  base", "other:1b  base"])
    fe = FakeEnv(tags={})
    for m in (canonical, alias, "other:1b"):
        fe.qc[m] = {"result": "native tool call"}
    mq.main(["--batch", str(batch), "--tag", "alias", "--delete-after"], env=fe.as_env(), runs_root=tmp_path / "runs")
    assert canonical not in fe.deleted
    assert alias in fe.deleted  # alias is not protected unless preexisting
    assert "other:1b" in fe.deleted


def test_edge_finish_capped_only_capped_not_phase1(tmp_path: Path):
    batch = write_batch(tmp_path / "b.txt", ["capped:1b  base", "fresh:1b  base"])
    runs = tmp_path / "runs"
    batch_dir = runs / "queue_fc"
    batch_dir.mkdir(parents=True)
    state = {
        "meta": {"commit": "abc1234deadbeef", "ollama_version": "0.9.9-fake",
                 "preexisting_set": [], "preexisting": []},
        "rows": {
            "capped:1b": {"model": "capped:1b", "kind": "base", "status": "capped",
                          "quick_check": "native tool call", "parts_done": ["pull", "show", "fit", "quick_check"],
                          "triage": "", "digest": "d1"},
            "fresh:1b": {"model": "fresh:1b", "kind": "base", "status": "phase1_done",
                         "quick_check": "native tool call", "parts_done": ["pull", "show", "fit", "quick_check"],
                         "triage": "", "digest": "d2"},
        },
        "history": [],
    }
    (batch_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    fe = FakeEnv(head="abc1234deadbeef", version="0.9.9-fake")
    fe.qc["capped:1b"] = {"result": "native tool call"}
    fe.qc["fresh:1b"] = {"result": "native tool call"}
    code = mq.main(
        ["--batch", str(batch), "--tag", "fc", "--finish-capped"],
        env=fe.as_env(), runs_root=runs,
    )
    assert code == 0
    assert fe.packs_log == ["capped:1b"]
    assert "fresh:1b" not in fe.packs_log


def test_edge_allow_change_preserves_prior_digest_history(tmp_path: Path):
    batch = write_batch(tmp_path / "b.txt", ["m:1b  base"])
    fe = FakeEnv()
    fe.qc["m:1b"] = {"result": "native tool call"}
    fe.tags = {"m:1b": "digest-old"}
    assert mq.main(["--batch", str(batch), "--tag", "dig"], env=fe.as_env(), runs_root=tmp_path / "runs") == 0
    # Simulate re-pull with new digest under allow-change
    state_path = tmp_path / "runs" / "queue_dig" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    # Force row back to phase1-needed with old digest kept; re-run phase1 with new digest
    state["rows"]["m:1b"]["status"] = "phase1_done"
    state["rows"]["m:1b"]["parts_done"] = ["pull", "show", "fit", "quick_check"]
    state["rows"]["m:1b"]["digest"] = "digest-old"
    # Mark done so archive triggers on new digest after we reset to re-pull
    state["rows"]["m:1b"]["status"] = "done"
    state["rows"]["m:1b"]["parts_done"] = ["pull", "show", "fit", "quick_check", "packs", "ladder"]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    # Clear current row so phase1 runs again: delete from rows to force re-entry — or set status empty
    state["rows"].pop("m:1b")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    # Put old into history manually then run — exercise code path via prev in rows_map:
    state["rows"]["m:1b"] = {
        "model": "m:1b", "kind": "base", "status": "done", "digest": "digest-old",
        "quick_check": "native tool call", "parts_done": ["pull", "show", "fit", "quick_check", "packs", "ladder"],
        "triage": "",
    }
    # To re-run phase1, status must not be phase1_finished — use a hack: empty parts and non-terminal won't work
    # because done is terminal. So archive path runs when we replace after phase1 with different digest.
    # Manually call the archive by running phase1 after removing terminal status but keeping digest:
    state["rows"]["m:1b"]["status"] = "phase1_done"
    state["rows"]["m:1b"]["parts_done"] = []  # not finished phase1 by parts — but status phase1_done without quick_check in parts
    # phase1_finished: status phase1_done AND quick_check in parts — False here, so phase1 re-runs
    state_path.write_text(json.dumps(state), encoding="utf-8")
    fe2 = FakeEnv(head="newcommit999", version="new-ollama")
    fe2.tags = {"m:1b": "digest-new"}
    fe2.qc["m:1b"] = {"result": "native tool call"}
    code = mq.main(
        ["--batch", str(batch), "--tag", "dig", "--allow-change"],
        env=fe2.as_env(), runs_root=tmp_path / "runs",
    )
    assert code == 0
    state2 = json.loads(state_path.read_text(encoding="utf-8"))
    assert state2["rows"]["m:1b"]["digest"] == "digest-new"
    hist = state2.get("history") or []
    assert any(h.get("digest") == "digest-old" for h in hist)
    summary = (tmp_path / "runs" / "queue_dig" / "summary.csv").read_text(encoding="utf-8")
    assert "digest-old" in summary and "digest-new" in summary
    assert "allow-change" in summary


def test_edge_cap_between_packs_and_ladder(tmp_path: Path):
    batch = write_batch(tmp_path / "b.txt", ["solo:1b  base"])
    fe = FakeEnv()
    fe.qc["solo:1b"] = {"result": "native tool call"}
    ticks = {"n": 0}

    def clock():
        ticks["n"] += 1
        # deadline assign (1) -> t0; packs check (2) -> t0 OK; after packs, ladder check uses
        # deadline from assign — bump after packs by making clock large on call >= 3
        if ticks["n"] >= 3:
            return 1_000_000.0 + 10_000_000
        return 1_000_000.0

    env = fe.as_env()
    env.clock = clock
    code = mq.main(
        ["--batch", str(batch), "--tag", "capparts", "--model-hours", "1"],
        env=env, runs_root=tmp_path / "runs",
    )
    assert code == 0
    row = json.loads((tmp_path / "runs" / "queue_capparts" / "state.json").read_text(encoding="utf-8"))["rows"]["solo:1b"]
    assert row["status"] == "capped"
    assert "ladder" in row["parts_done"]                                 # C19: the ladder runs first now
    assert row.get("capped_at") == "before_packs"
    assert "packs" not in row["parts_done"]


def test_edge_atomic_state_survives_interrupted_write(tmp_path: Path):
    """write_json_atomic replaces via temp+os.replace; a .tmp left behind is not the state file."""
    batch = tmp_path / "queue_x"
    batch.mkdir()
    path = mq.state_path(batch)
    mq.write_json_atomic(path, {"ok": True, "n": 1})
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8"))["n"] == 1
    # Simulate crash leaving a tmp beside the file
    tmp = batch / (path.name + ".crash.tmp")
    tmp.write_text("{broken", encoding="utf-8")
    loaded = mq.load_state(batch)
    assert loaded["ok"] is True
    assert loaded["n"] == 1


def test_prod_list_makes_no_provider_calls(tmp_path: Path):
    batch = write_batch(tmp_path / "b.txt", ["a:1b  base"])
    calls = {"vram": 0, "gpu": 0, "tags": 0, "pull": 0}

    def boom_vram():
        calls["vram"] += 1
        raise AssertionError("read_vram must not run for --list")

    fe = FakeEnv()
    env = fe.as_env()
    env.read_vram = boom_vram
    env.gpu_visible = lambda: (calls.__setitem__("gpu", calls["gpu"] + 1) or True)
    env.ollama_tags = lambda: (calls.__setitem__("tags", calls["tags"] + 1) or {})
    env.pull = lambda m: (calls.__setitem__("pull", calls["pull"] + 1) or (True, "ok"))
    code = mq.main(["--batch", str(batch), "--tag", "nolist", "--list"], env=env, runs_root=tmp_path / "runs")
    assert code == 0
    assert calls == {"vram": 0, "gpu": 0, "tags": 0, "pull": 0}


def test_prod_stage1_plan_matches_sign_off_profiles(tmp_path: Path):
    plan = mq.write_batch_plan(tmp_path, "m:1b")
    data = json.loads(plan.read_text(encoding="utf-8"))
    by_pack = {r["pack"]: r for r in data["rows"]}
    assert set(by_pack) == {"work", "trust", "privacy", "injection", "long"}
    for pack in ("work", "trust", "privacy", "injection"):
        assert by_pack[pack]["profile"] == "core4"
        assert by_pack[pack]["n"] == 3
    assert by_pack["long"]["profile"] == "core4_product"
    assert by_pack["long"]["n"] == 3
    # loadable by gate_run.load_plan
    rows = __import__("gate_run", fromlist=["load_plan"]).load_plan(plan)
    assert len(rows) == 5


def test_prod_parse_pack_scores_from_result_schema(tmp_path: Path):
    out = tmp_path / "packs"
    out.mkdir()
    result = {
        "pack": "work", "model": "m:1b", "passed": 2, "total": 3,
        "host_changes": 0, "g5_secret_hits": 0, "git_changes": 0,
        "tasks": {
            "t1": {"pass": True, "trials": [{"pass": True}, {"pass": True}, {"pass": False}]},
            "t2": {"pass": False, "trials": [{"pass": False}, {"pass": True}, {"pass": True}]},
            "t3": {"pass": True, "trials": [{"pass": True}, {"pass": True}, {"pass": True}]},
        },
    }
    (out / "work_result.json").write_text(json.dumps(result), encoding="utf-8")
    parsed = mq.parse_pack_scores(out, "m:1b")
    assert parsed["scores"]["work"] == "2/3"
    assert parsed["first_try"]["work"] == "2/3"  # t1+t3 first trials
    assert parsed["hard"]["outside_box"] == 0


def test_prod_quick_check_defaults_match_ladder_base():
    """Production quick check uses chat.CHAT_REPLY_LIMIT, BASE_CONTEXT / BASE_STEPS / 180s."""
    import inspect
    src = inspect.getsource(mq._quick_check_default)
    assert "180" in src or "180.0" in src
    assert "BASE_CONTEXT" in src
    assert "BASE_STEPS" in src
    assert "CHAT_REPLY_LIMIT" in src or "quick_check_reply_limit" in src
    assert "LADDER_REPLY_LIMITS[0]" not in src
    assert "QUICK_CHECK_TASK" in src


def test_prod_held_env_set_during_batch(tmp_path: Path, monkeypatch):
    """Queue sets LOCALBOT_RUNLOCK_HELD so nested gate_run can skip re-acquire."""
    import runlock
    seen = {"held": None}

    batch = write_batch(tmp_path / "b.txt", ["a:1b  base"])
    fe = FakeEnv()
    fe.qc["a:1b"] = {"result": "native tool call"}

    def packs(m, plan, out):
        seen["held"] = os.environ.get(runlock.HELD_ENV)
        return {"exit": 0, "scores": {}, "first_try": {}, "hard": {}}

    fe.packs_hook = packs
    import os
    mq.main(["--batch", str(batch), "--tag", "held"], env=fe.as_env(), runs_root=tmp_path / "runs")
    assert seen["held"] == "1"


def test_edge_over_card_skips_quick_and_packs(tmp_path: Path, monkeypatch):
    batch = write_batch(tmp_path / "b.txt", ["huge:1b  base"])
    fe = FakeEnv()
    monkeypatch.setattr(mq, "fit_at_8192", lambda *a, **k: {"fits": False, "lane": "spill", "size_bytes": 1})
    qc_calls = {"n": 0}
    env = fe.as_env()
    env.quick_check = lambda m: qc_calls.__setitem__("n", qc_calls["n"] + 1) or {"result": "native tool call"}
    mq.main(["--batch", str(batch), "--tag", "ov"], env=env, runs_root=tmp_path / "runs")
    assert qc_calls["n"] == 0
    assert fe.packs_log == []

def test_prod_gate_run_nested_under_held_lock(tmp_path: Path, monkeypatch):
    """With --assume-locked + HELD + parent lock, gate_run runs without LockBusy."""
    import runlock
    import gate_run as gate
    lock_file = tmp_path / "run.lock"
    monkeypatch.setenv("LOCALBOT_RUNLOCK_PATH", str(lock_file))
    monkeypatch.setenv("LOCALBOT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(gate, "git_porcelain", lambda: "")
    monkeypatch.setattr(gate, "git_head", lambda: "abc123def456")
    monkeypatch.setattr(gate, "gpu_visible", lambda: True)
    monkeypatch.setattr(gate, "ollama_tags", lambda: {"m:1b": "d"})
    calls = []

    def launch(argv, log_path, cwd):
        calls.append(argv)
        log_path.write_text("wrote " + str(tmp_path / "fake.json") + "\n", encoding="utf-8")
        (tmp_path / "fake.json").write_text(
            json.dumps({"commit": "abc123def456", "passed": 0, "total": 0, "pack": "work", "model": "m:1b", "tasks": {}}),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(gate, "launch", launch)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps({"rows": [
            {"gate": "Q-work", "model": "m:1b", "pack": "work", "profile": "core4", "n": 3},
        ]}),
        encoding="utf-8",
    )
    lk = runlock.acquire("model_queue", cmd=["model_queue"], path=lock_file)
    try:
        monkeypatch.setenv(runlock.HELD_ENV, "1")
        code = gate.main([
            "--plan", str(plan), "--commit", "abc123def456", "--tag", "t", "--assume-locked",
        ])
        assert code in {0, 1}  # may fail ok-check on result commit; must not be busy(3)
        assert code != 3
        assert calls  # launch ran — lock nest worked
    finally:
        monkeypatch.delenv(runlock.HELD_ENV, raising=False)
        lk.release()


def test_prod_gate_run_without_assume_locked_stays_busy(tmp_path: Path, monkeypatch):
    """Flag absent: gate_run still acquires; LockBusy when lock already held (no HELD nest)."""
    import runlock
    import gate_run as gate
    lock_file = tmp_path / "run.lock"
    monkeypatch.setenv("LOCALBOT_RUNLOCK_PATH", str(lock_file))
    monkeypatch.setenv("LOCALBOT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.delenv(runlock.HELD_ENV, raising=False)
    monkeypatch.setattr(gate, "git_porcelain", lambda: "")
    monkeypatch.setattr(gate, "git_head", lambda: "abc123def456")
    monkeypatch.setattr(gate, "gpu_visible", lambda: True)
    monkeypatch.setattr(gate, "ollama_tags", lambda: {"m:1b": "d"})
    calls = []
    monkeypatch.setattr(gate, "launch", lambda *a, **k: calls.append(a) or 0)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps({"rows": [
            {"gate": "Q-work", "model": "m:1b", "pack": "work", "profile": "core4", "n": 3},
        ]}),
        encoding="utf-8",
    )
    lk = runlock.acquire("holder", cmd=["holder"], path=lock_file)
    try:
        code = gate.main(["--plan", str(plan), "--commit", "abc123def456", "--tag", "t"])
        assert code == 3
        assert not calls
        assert "busy" in (gate.LAST_STDOUT + gate.LAST_STDERR).lower()
    finally:
        lk.release()

