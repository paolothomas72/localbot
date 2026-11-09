"""C19: the long pack runs twice (standard + best this card allows); ladder first; note per model; two-skips stop
off by default; smart 12-hour cap (red tests first).

Nothing here starts a process, loads a model or touches the network: the queue runs on ``FakeEnv``
(tests/test_c9_model_queue.py); the end-to-end test replaces only ``gate_run.launch``, ``chat_ladder.main`` and the
host probes. Fake run folders live in tmp_path.
"""
from __future__ import annotations

import csv
import json
import socket
from pathlib import Path

import pytest

import scripts.model_queue as mq
from tests import test_c18a_queue_status as c18a
from tests.test_c18b_queue_skip import _three
from tests.test_c9_model_queue import FakeEnv, write_batch
from tests.test_c9b_review_fixes import _try_row

gate = mq.gate
runlock = mq.runlock
ladder = mq.ladder

MODEL = "m:1b"
TAG = "t"
COMMIT = "abc123def456"
HOUR = 3600.0


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def refuse(self, *a, **k):
        raise AssertionError("network used: a C19 test must not connect anywhere")
    monkeypatch.setattr(socket.socket, "connect", refuse)


# --- helpers -------------------------------------------------------------------------------------------------------

def _csv(path: Path, levels: list[tuple]) -> Path:
    """levels: (context, fits, slow) per level; 3 tries each; the first level is the speed reference."""
    rows = []
    for idx, (ctx, fits, slow) in enumerate(levels, start=1):
        for t in range(1, 4):
            rows.append(_try_row(level=idx, try_n=t, reply_limit=512, context=ctx, think="default", steps_limit=12,
                                 ended_why="done", worked=True, fits=fits, seconds=10.0,
                                 tps=10.0 if slow else 50.0, doubtful=slow))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=ladder.COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ladder.COLUMNS})
    return path


def _env(fe: FakeEnv, calls: list, ctx="default", b_hook=None, check=None):
    """FakeEnv -> Env whose ladder / packs calls are recorded as (kind, model, plan, out) in ``calls``."""
    if ctx == "default":
        ctx = 16384
    fe.ladder_hook = lambda m, out: {"exit": 0, "best": "ctx=8192", "card_limit": "16384 ctx; 10/12 GB",
                                     "card_limit_ctx": ctx}
    env = fe.as_env()
    env.check_model = check or (lambda m, b: None)
    real_packs, real_ladder = env.run_packs, env.run_ladder

    def packs(m, plan, out):
        calls.append((out.name, m, plan, out))
        if out.name == "long_best" and b_hook:
            out.mkdir(parents=True, exist_ok=True)
            return b_hook(m, plan, out)
        return real_packs(m, plan, out)

    def lad(m, out):
        calls.append(("ladder", m, None, out))
        return real_ladder(m, out)

    env.run_packs, env.run_ladder = packs, lad
    return env


def _row(parts=("pull", "show", "fit", "quick_check"), model=MODEL, **extra) -> dict:
    row = {"model": model, "kind": "base", "status": "phase1_done", "quick_check": "native tool call",
           "parts_done": list(parts)}
    if "packs" in parts:
        row["packs"] = {"exit": 0, "scores": {"long": "3/17"}, "first_try": {}, "hard": {}}
    if "ladder" in parts:
        row["ladder"] = {"exit": 0, "best": "ctx=8192", "card_limit": "16384 ctx; 10/12 GB", "card_limit_ctx": 16384}
    row.update(extra)
    return row


def _go(tmp_path: Path, fe: FakeEnv, env, row: dict, left: float = 100 * HOUR, finish_capped: bool = False,
        wall: "float | None" = None):
    batch = tmp_path / "runs" / f"queue_{TAG}"
    if wall is not None:
        packs_dir = batch / "models" / mq._safe(row["model"]) / "packs"
        packs_dir.mkdir(parents=True, exist_ok=True)
        (packs_dir / "summary_20261108T000000.json").write_text(
            json.dumps({"rows": [{"gate": "Q-long", "wall_s": wall}]}), encoding="utf-8", newline="\n")
    state = {"meta": {}, "rows": {row["model"]: row}}
    kw = {"finish_capped": True} if finish_capped else {}
    out = mq.phase2_one(row, batch, env, fe.clock_value + left, state, **kw)
    return out, batch, state


def _names(calls: list) -> list[str]:
    return [c[0] for c in calls]


# --- 1. order ------------------------------------------------------------------------------------------------------

def test_1_fresh_row_runs_ladder_then_packs_then_long_best(tmp_path):
    fe, calls = FakeEnv(), []
    row, batch, state = _go(tmp_path, fe, _env(fe, calls), _row())
    assert _names(calls) == ["ladder", "packs", "long_best"]
    assert row["status"] == "done" and row["parts_done"][-3:] == ["ladder", "packs", "long_best"]


# --- 2. card_limit_ctx ---------------------------------------------------------------------------------------------

def test_2_card_limit_ctx_is_the_largest_all_fit_context(tmp_path):
    p = _csv(tmp_path / "ladder_a.csv", [(8192, "yes", False), (16384, "yes", False), (32768, "no", False)])
    got = mq.ladder_best_from_csv(p)
    assert got["card_limit_ctx"] == 16384 and isinstance(got["card_limit_ctx"], int)
    assert got["card_limit"] == "16384 ctx; 10/12.8 GB"                 # unchanged strings
    assert got["best"] == "ctx=8192;reply=512;think=default;steps=12"
    assert mq.parse_ladder_summary(tmp_path)["card_limit_ctx"] == 16384


def test_2b_nothing_fits_is_none(tmp_path):
    p = _csv(tmp_path / "ladder_b.csv", [(8192, "no", False), (16384, "no", False)])
    got = mq.ladder_best_from_csv(p)
    assert got["card_limit_ctx"] is None and got["card_limit"] == ""


def test_2c_a_slow_level_that_fit_is_excluded(tmp_path):
    p = _csv(tmp_path / "ladder_c.csv", [(8192, "yes", False), (16384, "yes", True)])
    got = mq.ladder_best_from_csv(p)
    assert got["card_limit_ctx"] == 8192
    assert got["card_limit"] == "8192 ctx; 10/12.8 GB"


def test_2d_no_csv_and_empty_csv_are_none(tmp_path):
    assert mq.parse_ladder_summary(tmp_path)["card_limit_ctx"] is None
    empty = tmp_path / "e" / "ladder_e.csv"
    empty.parent.mkdir()
    empty.write_text(",".join(ladder.COLUMNS) + "\n", encoding="utf-8", newline="\n")
    assert mq.ladder_best_from_csv(empty)["card_limit_ctx"] is None


# --- 3. plans ------------------------------------------------------------------------------------------------------

def test_3_run_b_plan_has_exactly_one_row_and_runs_into_long_best(tmp_path):
    fe, calls = FakeEnv(), []
    row, batch, state = _go(tmp_path, fe, _env(fe, calls, ctx=32768), _row())
    model_dir = batch / "models" / mq._safe(MODEL)
    b = [c for c in calls if c[0] == "long_best"]
    assert len(b) == 1 and b[0][2] == model_dir / "stage1_plan_long_best.json" and b[0][3] == model_dir / "long_best"
    plan_b = json.loads((model_dir / "stage1_plan_long_best.json").read_text(encoding="utf-8"))
    assert plan_b == {"rows": [{"gate": "Q-long-best", "model": MODEL, "pack": "long", "profile": "core4_product",
                                "n": 3, "num_ctx": 32768, "max_turns": 25, "num_predict": 1024}]}
    assert (mq.LONG_BEST_STEPS, mq.LONG_BEST_REPLY, mq.LONG_BEST_GATE) == (25, 1024, "Q-long-best")
    assert row["long_best"]["settings"] == {"num_ctx": 32768, "max_turns": 25, "num_predict": 1024}
    plan_a = json.loads((model_dir / "stage1_plan.json").read_text(encoding="utf-8"))
    assert len(plan_a["rows"]) == 5
    for r in plan_a["rows"]:
        assert not ({"num_ctx", "max_turns", "num_predict"} & set(r))
    assert [r["gate"] for r in plan_a["rows"]] == ["Q-work", "Q-trust", "Q-privacy", "Q-injection", "Q-long"]


# --- 4. nothing measured -------------------------------------------------------------------------------------------

def test_4_no_card_limit_means_run_b_is_not_called(tmp_path):
    fe, calls = FakeEnv(), []
    row, batch, state = _go(tmp_path, fe, _env(fe, calls, ctx=None), _row())
    assert _names(calls) == ["ladder", "packs"]
    assert row["long_best"] == {"not_run": "no context was measured as fitting on the card"}
    assert row["status"] == "done"


# --- 5. resume -----------------------------------------------------------------------------------------------------

def test_5_resume_ladder_and_packs_done_runs_only_long_best(tmp_path):
    fe, calls = FakeEnv(), []
    row, _, _ = _go(tmp_path, fe, _env(fe, calls), _row(parts=("pull", "show", "fit", "quick_check", "ladder", "packs")))
    assert _names(calls) == ["long_best"] and row["status"] == "done"


def test_5b_all_three_done_runs_nothing(tmp_path):
    fe, calls = FakeEnv(), []
    parts = ("pull", "show", "fit", "quick_check", "ladder", "packs", "long_best")
    row, _, _ = _go(tmp_path, fe, _env(fe, calls), _row(parts=parts))
    assert calls == [] and row["status"] == "done"


def test_5c_old_order_row_with_only_packs_runs_ladder_and_long_best(tmp_path):
    fe, calls = FakeEnv(), []
    row, _, _ = _go(tmp_path, fe, _env(fe, calls), _row(parts=("pull", "show", "fit", "quick_check", "packs")))
    assert _names(calls) == ["ladder", "long_best"] and row["status"] == "done"


# --- 6. deadline ---------------------------------------------------------------------------------------------------

def test_6_deadline_passed_before_b_caps_and_finish_capped_runs_only_b(tmp_path):
    fe, calls = FakeEnv(), []
    parts = ("pull", "show", "fit", "quick_check", "ladder", "packs")
    row, batch, state = _go(tmp_path, fe, _env(fe, calls), _row(parts=parts), left=-1)
    assert calls == [] and row["status"] == "capped" and row["capped_at"] == "before_long_best"
    assert "long_best" not in row["parts_done"]
    row2, _, _ = _go(tmp_path, fe, _env(fe, calls), row, finish_capped=True)
    assert _names(calls) == ["long_best"] and row2["status"] == "done"


def test_6b_the_old_capped_values_are_kept(tmp_path):
    fe, calls = FakeEnv(), []
    row, _, _ = _go(tmp_path, fe, _env(fe, calls), _row(), left=-1)
    assert row["capped_at"] == "before_ladder" and calls == []
    fe, calls = FakeEnv(), []
    row, _, _ = _go(tmp_path, fe, _env(fe, calls), _row(parts=("pull", "show", "fit", "quick_check", "ladder")), left=-1)
    assert row["capped_at"] == "before_packs" and calls == []


# --- 7. the note ---------------------------------------------------------------------------------------------------

def _try_lines(n: int, at_limit: int) -> str:
    out = []
    for i in range(n):
        end = "max_steps" if i < at_limit else "done"
        out.append(f"{MODEL} e_read t{i} pass=True judge=True end={end} vetoes=0 infra=False")
    out.append("wrote C:\\x\\results\\stage1\\long_core4_product_m.json  long disk 3/17")
    return "\n".join(out) + "\n"


def _note_batch(tmp_path: Path, a_limit=35, b_limit=12) -> Path:
    batch = tmp_path / "queue_n"
    base = batch / "models" / mq._safe(MODEL)
    (base / "packs").mkdir(parents=True)
    (base / "long_best").mkdir(parents=True)
    (base / "packs" / f"packs_Q-long_long_core4_product_{MODEL.replace(':', '_')}.log").write_text(
        _try_lines(51, a_limit), encoding="utf-8", newline="\n")
    (base / "packs" / f"packs_Q-work_work_core4_{MODEL.replace(':', '_')}.log").write_text(
        _try_lines(9, 9), encoding="utf-8", newline="\n")              # another pack: must not be counted
    (base / "long_best" / f"long_best_Q-long-best_long_core4_product_{MODEL.replace(':', '_')}.log").write_text(
        _try_lines(51, b_limit), encoding="utf-8", newline="\n")
    return batch


def _done_row(**long_best) -> dict:
    return {"model": MODEL, "kind": "base", "status": "done", "packs": {"scores": {"long": "3/17"}},
            "ladder": {}, "long_best": long_best}


def _summary_cells(batch: Path, row: dict) -> tuple[dict, str]:
    mq.write_summary(batch, [row], {"ollama_version": "v", "commit": "c"})
    with (batch / "summary.csv").open(encoding="utf-8", newline="") as handle:
        cells = next(csv.DictReader(handle))
    return cells, (batch / "summary.md").read_text(encoding="utf-8")


def test_7_note_has_both_scores_both_settings_and_both_step_limit_counts(tmp_path):
    batch = _note_batch(tmp_path)
    row = _done_row(scores={"long": "9/17"}, settings={"num_ctx": 32768, "max_turns": 25, "num_predict": 1024})
    cells, md = _summary_cells(batch, row)
    expected = ("standard 8192 ctx / 8 steps: 3 of 17 - best on this card 32768 ctx / 25 steps: 9 of 17 - "
                "stopped at the step limit: A 35 of 51 tries, B 12 of 51")
    assert cells["long_note"] == expected
    assert expected in md and "long_note" in mq.SUMMARY_COLUMNS


def test_7b_note_when_b_did_not_run(tmp_path):
    batch = _note_batch(tmp_path)
    cells, md = _summary_cells(batch, _done_row(not_run="no context was measured as fitting on the card"))
    assert cells["long_note"] == ("standard 8192 ctx / 8 steps: 3 of 17 - best on this card: not run "
                                  "(no context was measured as fitting on the card)")


def test_7c_note_when_capped(tmp_path):
    batch = _note_batch(tmp_path)
    row = {"model": MODEL, "kind": "base", "status": "capped", "capped_at": "before_long_best",
           "packs": {"scores": {"long": "3/17"}}, "ladder": {}, "cap_estimate_s": 7200, "cap_left_s": 3600}
    cells, md = _summary_cells(batch, row)
    assert cells["long_note"] == ("standard 8192 ctx / 8 steps: 3 of 17 - best on this card: not run yet "
                                  "(capped: estimated 120 min, 60 min left)")


def test_7d_a_reason_with_a_pipe_is_escaped_in_markdown_only(tmp_path):
    batch = _note_batch(tmp_path)
    cells, md = _summary_cells(batch, _done_row(not_run="a|b"))
    assert "(a|b)" in cells["long_note"]
    assert "(a\\|b)" in md
    body = [ln for ln in md.splitlines() if ln.startswith("| " + MODEL)][0]
    assert len(body.replace("\\|", "").split("|")) == len(mq.SUMMARY_COLUMNS) + 2   # no extra cell from the pipe


def test_7e_a_row_without_long_data_has_an_empty_note(tmp_path):
    batch = tmp_path / "queue_e"
    batch.mkdir()
    cells, _ = _summary_cells(batch, {"model": MODEL, "kind": "base", "status": "done", "packs": {}, "ladder": {}})
    assert cells["long_note"] == ""


# --- 8. two-skips stop is off by default ---------------------------------------------------------------------------

def test_8_default_two_broken_models_do_not_stop_the_batch(tmp_path):
    code, fe, rows, b = _three(tmp_path, [], {"a:1b", "b:1b"})
    assert mq.DEFAULT_STOP_AFTER_SKIPS == 0
    assert rows["a:1b"]["status"] == "skipped: broken" and rows["b:1b"]["status"] == "skipped: broken"
    assert rows["c:1b"]["status"] == "done"
    assert code == 0


def test_8b_stop_after_skips_two_still_stops(tmp_path):
    code, fe, rows, b = _three(tmp_path, ["--stop-after-skips", "2"], {"a:1b", "b:1b"})
    assert rows["c:1b"]["status"] == "phase1_done"
    assert code == mq.EXIT_STOPPED_AFTER_SKIPS


def test_8c_help_text_says_off_by_default():
    help_text = mq.build_parser().format_help()
    assert "off by default" in help_text


# --- 9. a broken model is skipped before run B too -----------------------------------------------------------------

def test_9_skip_reported_by_run_b_marks_the_model_skipped(tmp_path):
    fe, calls = FakeEnv(), []
    hook = lambda m, plan, out: {"exit": 0, "scores": {}, "first_try": {}, "hard": {},
                                 "skipped": {"reason": "broken", "before": "long"}}
    row, batch, state = _go(tmp_path, fe, _env(fe, calls, b_hook=hook), _row())
    assert _names(calls) == ["ladder", "packs", "long_best"]
    assert row["status"] == "skipped: broken" and row["skipped"]["before"] == "long"
    assert row["retest_next_batch"] is True and "long_best" in row["parts_done"]


def test_9b_the_default_run_packs_asks_the_check_for_run_b(tmp_path, monkeypatch):
    seen: dict = {}

    def fake_main(argv, **kw):
        hook = kw.get("before_pack")
        seen["argv"] = argv
        seen["reason"] = hook({"gate": "Q-long-best", "model": MODEL, "pack": "long", "profile": "core4_product",
                               "n": 3}) if callable(hook) else "NO HOOK"
        return 0

    monkeypatch.setattr(gate, "main", fake_main)
    monkeypatch.setattr(gate, "git_head", lambda: COMMIT)
    env = mq.Env()
    asked: list = []
    env.check_model = lambda m, b: asked.append((m, b)) or "broken"
    batch = tmp_path / "runs" / f"queue_{TAG}"
    result = env.run_packs(MODEL, batch / "plan.json", batch / "models" / mq._safe(MODEL) / "long_best")
    assert seen["reason"] == "broken" and asked == [(MODEL, batch)]
    assert result["skipped"] == {"reason": "broken", "before": "long"}
    assert seen["argv"][seen["argv"].index("--tag") + 1] == "long_best"


# --- 10. end to end ------------------------------------------------------------------------------------------------

def test_10_end_to_end_real_phase2_real_run_packs_real_run_ladder(tmp_path, monkeypatch):
    """Only gate_run.launch, chat_ladder.main (it would start real tries) and the host probes are replaced.
    It cannot pass on the old code: there is no run B there."""
    runs = tmp_path / "runs"
    batch = runs / f"queue_{TAG}"
    model = "org/m:1"
    monkeypatch.setenv("LOCALBOT_RUNLOCK_PATH", str(tmp_path / "run.lock"))
    monkeypatch.delenv(runlock.HELD_ENV, raising=False)
    monkeypatch.setattr(gate, "git_porcelain", lambda: "")
    monkeypatch.setattr(gate, "git_head", lambda: COMMIT)
    monkeypatch.setattr(gate, "gpu_visible", lambda: True)
    monkeypatch.setattr(gate, "ollama_tags", lambda: {model: "d"})
    launched: list[dict] = []

    def launch(argv, log_path, cwd):
        pack = argv[argv.index("--pack") + 1]
        launched.append({"pack": pack, "argv": list(argv), "log": log_path.name})
        result = tmp_path / "results" / f"{len(launched)}_{pack}.json"
        result.parent.mkdir(exist_ok=True)
        result.write_text(json.dumps({"run_env": {"git": {"commit": COMMIT}}, "tasks": {}, "total": 0, "wall_s": 1,
                                      "pack": pack, "model": model}), encoding="utf-8", newline="\n")
        body = [c18a._try("e_read", k) .replace("infra=False", "infra=False") for k in (1, 2)]
        body.append(c18a._wrote(no_result=0).replace("C:\\x\\results\\stage1\\work_core4_m.json", str(result)))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(body) + "\n", encoding="utf-8", newline="\n")
        return 0

    monkeypatch.setattr(gate, "launch", launch)
    ladder_calls: list[list[str]] = []

    def fake_ladder_main(argv):
        ladder_calls.append(list(argv))
        _csv(Path(argv[argv.index("--out-dir") + 1]) / "ladder_x.csv",
             [(8192, "yes", False), (16384, "yes", False), (32768, "no", False)])
        return 0

    monkeypatch.setattr(ladder, "main", fake_ladder_main)
    fe = FakeEnv()
    env = fe.as_env()
    env.check_model = lambda m, b: mq._check_model_default(m, b, now=fe.clock_value)
    env.run_packs = lambda m, plan, out: mq._run_packs_default(m, plan, out, check=env.check_model)
    env.run_ladder = lambda m, out: mq._run_ladder_default(m, out)
    monkeypatch.setenv("LOCALBOT_RUNS_DIR", str(runs))
    row = _row(model=model)
    state = {"meta": {}, "rows": {model: row}}
    out = mq.phase2_one(row, batch, env, fe.clock_value + 100 * HOUR, state)

    assert out["status"] == "done", out
    assert "check_error" not in out
    assert len(ladder_calls) == 1 and out["ladder"]["card_limit_ctx"] == 16384
    packs_launched = [x["pack"] for x in launched]
    assert packs_launched == ["work", "trust", "privacy", "injection", "long", "long"]
    a_long, b_long = launched[4], launched[5]
    for flag in ("--num-ctx", "--max-turns", "--num-predict"):
        assert flag not in a_long["argv"]
    assert b_long["argv"][b_long["argv"].index("--num-ctx") + 1] == "16384"
    assert b_long["argv"][b_long["argv"].index("--max-turns") + 1] == "25"
    assert b_long["argv"][b_long["argv"].index("--num-predict") + 1] == "1024"
    assert "Q-long-best" in b_long["log"] and a_long["log"].startswith("packs_")
    assert out["long_best"]["settings"] == {"num_ctx": 16384, "max_turns": 25, "num_predict": 1024}
    assert out["long_best"]["out_dir"].endswith("long_best")
    assert mq.long_a_wall_s(batch / "models" / mq._safe(model) / "packs") is not None


# --- 11. smart cap -------------------------------------------------------------------------------------------------

PARTS_AB = ("pull", "show", "fit", "quick_check", "ladder", "packs")


def test_11_one_hour_left_run_b_is_not_started(tmp_path):
    fe, calls = FakeEnv(), []
    row, _, state = _go(tmp_path, fe, _env(fe, calls), _row(parts=PARTS_AB), left=1 * HOUR, wall=3600)
    assert calls == []
    assert row["status"] == "capped" and row["capped_at"] == "before_long_best"
    assert row["cap_estimate_s"] == 7200 and row["cap_left_s"] == 3600
    assert state["rows"][MODEL]["capped_at"] == "before_long_best"


def test_11b_three_hours_left_run_b_runs(tmp_path):
    fe, calls = FakeEnv(), []
    row, _, _ = _go(tmp_path, fe, _env(fe, calls), _row(parts=PARTS_AB), left=3 * HOUR, wall=3600)
    assert _names(calls) == ["long_best"] and row["status"] == "done"


def test_11c_exactly_enough_time_run_b_runs(tmp_path):
    fe, calls = FakeEnv(), []
    row, _, _ = _go(tmp_path, fe, _env(fe, calls), _row(parts=PARTS_AB), left=2 * HOUR, wall=3600)
    assert _names(calls) == ["long_best"] and row["status"] == "done"


def test_11d_long_a_wall_s_is_none_for_anything_unusable_and_b_runs(tmp_path):
    packs = tmp_path / "p"
    assert mq.long_a_wall_s(packs) is None                              # no folder
    packs.mkdir()
    assert mq.long_a_wall_s(packs) is None                              # no summary file
    (packs / "summary_1.json").write_text("{not json", encoding="utf-8", newline="\n")
    assert mq.long_a_wall_s(packs) is None                              # bad JSON
    (packs / "summary_2.json").write_text(json.dumps({"rows": [{"gate": "Q-work", "wall_s": 5}]}),
                                          encoding="utf-8", newline="\n")
    assert mq.long_a_wall_s(packs) is None                              # no Q-long entry
    (packs / "summary_3.json").write_text(json.dumps({"rows": [{"gate": "Q-long", "wall_s": "slow"}]}),
                                          encoding="utf-8", newline="\n")
    assert mq.long_a_wall_s(packs) is None                              # not a number
    (packs / "summary_4.json").write_text(json.dumps({"rows": [{"gate": "Q-long", "wall_s": 3600}]}),
                                          encoding="utf-8", newline="\n")
    assert mq.long_a_wall_s(packs) == 3600.0
    for fe_wall in (None,):
        fe, calls = FakeEnv(), []
        row, _, _ = _go(tmp_path, fe, _env(fe, calls), _row(parts=PARTS_AB), left=1 * HOUR, wall=fe_wall)
        assert _names(calls) == ["long_best"]                           # estimate None -> only the plain check


def test_11e_finish_capped_does_not_use_the_estimate(tmp_path):
    fe, calls = FakeEnv(), []
    row, batch, _ = _go(tmp_path, fe, _env(fe, calls), _row(parts=PARTS_AB), left=1 * HOUR, wall=3600)
    assert calls == [] and row["status"] == "capped"
    row2, _, _ = _go(tmp_path, fe, _env(fe, calls), row, left=1 * HOUR, finish_capped=True)
    assert _names(calls) == ["long_best"] and row2["status"] == "done"


def test_11f_constants_and_help():
    assert mq.DEFAULT_MODEL_HOURS == 12 and mq.LONG_BEST_TIME_FACTOR == 2.0
    args = mq.build_parser().parse_args(["--batch", "b", "--tag", "t"])
    assert args.model_hours == 12 and args.stop_after_skips == 0
    help_text = " ".join(mq.build_parser().format_help().split())
    assert "hours per model (default 12); a part is not started if it would not finish in time; " \
           "a started part is never stopped" in help_text


def test_11g_no_kill_or_timeout_code_was_added():
    text = Path(mq.__file__).read_text(encoding="utf-8")
    for word in ("Popen", ".kill(", ".terminate(", "threading.Timer", "signal.alarm"):
        assert word not in text
