"""BRIEF C18a: scripts/queue_status.py, a read-only view of a running model queue.

Every folder here is built in tmp_path. Nothing reads main's live folders, calls Ollama or loads a model: ``ps_fn`` is
a stand-in and ``now`` is fixed. Try lines have the real shape written by localbot/exam/runner.py."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import scripts.queue_status as qs

NOW = 1_800_000_000.0
MODEL = "org/m:1"
FOLDER = "org_m_1"
TAG = "t"


def _try(task: str = "e_read", k: int = 1, infra: bool = False, ok: bool = True) -> str:
    return (f"{MODEL} {task} t{k} pass={ok} judge={ok} end=max_steps vetoes=0 host_change=0 neg_ok=True "
            f"secret_seen=0 g5=0 git_changed=False arg_err=0 false_success=0 infra={infra}")


def _wrote(no_result: int = 2, infra_reruns: int = 1) -> str:
    return (f"wrote C:\\x\\results\\stage1\\work_core4_m.json  work {MODEL} disk 9/12 lab-judge 9/12 "
            f"no_result={no_result} passed_without_acting=0 host_changes=0 git_changes=0 g5_secret_hits=0 "
            f"secret_seen=0 infra_reruns={infra_reruns} gpu=gpu")


def _age(path: Path, minutes: float) -> None:
    stamp = NOW - minutes * 60
    os.utime(path, (stamp, stamp))


def _build(tmp_path: Path, *, lines: list[str] | None = None, pack: str = "work", status: str = "phase1_done",
           raw_tail: str = "", age: float = 10.0, state: bool = True, extra_logs: dict | None = None) -> Path:
    """tmp_path/queue_t/ with one model. ``lines`` go into the newest pack log, ``raw_tail`` is appended as is."""
    batch = tmp_path / f"queue_{TAG}"
    packs = batch / "models" / FOLDER / "packs"
    packs.mkdir(parents=True)
    row = {"model": MODEL, "status": status, "quick_check": "native tool call", "parts_done": ["pull", "quick_check"]}
    (batch / "models" / FOLDER / "row.json").write_text(json.dumps(row), encoding="utf-8", newline="\n")
    if state:
        (batch / "state.json").write_text(json.dumps({"meta": {"commit": "c"}, "rows": {MODEL: row}}),
                                          encoding="utf-8", newline="\n")
    for name, body in (extra_logs or {}).items():
        path = packs / name
        path.write_text(body, encoding="utf-8", newline="\n")
        _age(path, age + 100)
    if lines is not None:
        log = packs / f"packs_Q-{pack}_{pack}_core4_{FOLDER}.log"
        log.write_text("\n".join(lines) + ("\n" if lines else "") + raw_tail, encoding="utf-8", newline="\n")
        _age(log, age)
    for path in batch.rglob("*"):
        if path.is_file() and path.name in ("row.json", "state.json"):
            _age(path, age + 200)
    return tmp_path


def _one(tmp_path: Path, ps_fn=lambda: [], **kw) -> dict:
    recs = qs.collect_status(tmp_path, TAG, NOW, ps_fn, **kw)
    assert len(recs) == 1
    return recs[0]


def _hash_tree(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


# 1 ---------------------------------------------------------------------------------------------------------------

def test_1_running_work_pack_with_five_tries(tmp_path):
    _build(tmp_path, lines=[_try(f"e_{i}") for i in range(5)])
    rec = _one(tmp_path)
    assert rec["model"] == MODEL
    assert rec["step"] == "full tests"
    assert rec["pack"] == "work"
    assert rec["tries"] == 5
    assert rec["flags"] == []


def test_1b_non_try_lines_are_ignored(tmp_path):
    _build(tmp_path, lines=["starting pack", _try(), "", "Traceback noise t9 pass", _try("e_calc")])
    assert _one(tmp_path)["tries"] == 2


# 2 ---------------------------------------------------------------------------------------------------------------

def test_2_last_activity_minutes_and_stalled(tmp_path):
    _build(tmp_path, lines=[_try()], age=10)
    rec = _one(tmp_path)
    assert rec["last_activity_min"] == 10
    assert "STALLED" not in rec["flags"]


def test_2b_eighty_minutes_is_stalled_at_default_75(tmp_path):
    _build(tmp_path, lines=[_try()], age=80)
    rec = _one(tmp_path)
    assert rec["last_activity_min"] == 80
    assert rec["flags"] == ["STALLED"]


def test_2c_stall_minutes_is_a_parameter(tmp_path):
    _build(tmp_path, lines=[_try()], age=80)
    assert _one(tmp_path, stall_minutes=120)["flags"] == []


def test_2d_newest_file_in_the_model_folder_counts_not_just_the_log(tmp_path):
    _build(tmp_path, lines=[_try()], age=80, extra_logs={"journal_x.jsonl": "{}\n"})
    newest = tmp_path / f"queue_{TAG}" / "models" / FOLDER / "packs" / "journal_x.jsonl"
    _age(newest, 3)
    assert _one(tmp_path)["last_activity_min"] == 3


# 3 ---------------------------------------------------------------------------------------------------------------

def test_3_newest_five_all_infra_is_broken(tmp_path):
    _build(tmp_path, lines=[_try(ok=False, infra=True) for _ in range(5)])
    rec = _one(tmp_path)
    assert "BROKEN" in rec["flags"]
    assert rec["errors"] == 5
    assert rec["error_streak"] == 5


def test_3b_four_errors_then_a_normal_try_is_not_broken(tmp_path):
    _build(tmp_path, lines=[_try(ok=False, infra=True) for _ in range(4)] + [_try()])
    rec = _one(tmp_path)
    assert "BROKEN" not in rec["flags"]
    assert rec["errors"] == 4
    assert rec["error_streak"] == 0


def test_3c_normal_then_four_errors_is_not_broken_yet(tmp_path):
    _build(tmp_path, lines=[_try()] + [_try(ok=False, infra=True) for _ in range(4)])
    rec = _one(tmp_path)
    assert "BROKEN" not in rec["flags"]
    assert rec["error_streak"] == 4


def test_3d_broken_after_is_a_parameter(tmp_path):
    _build(tmp_path, lines=[_try(ok=False, infra=True) for _ in range(3)])
    assert "BROKEN" in _one(tmp_path, broken_after=3)["flags"]


# 4 ---------------------------------------------------------------------------------------------------------------

GIB = 1024 ** 3


def _ps(size: int, vram: int) -> list[dict]:
    return [{"name": MODEL, "size": size, "size_vram": vram}]


def test_4_partly_on_cpu_is_spill(tmp_path):
    _build(tmp_path, lines=[_try()])
    rec = _one(tmp_path, ps_fn=lambda: _ps(10 * GIB, 6 * GIB))
    assert rec["gpu"] == "spill"
    assert "SPILL" in rec["flags"]


def test_4b_fully_on_gpu_has_no_flag(tmp_path):
    _build(tmp_path, lines=[_try()])
    rec = _one(tmp_path, ps_fn=lambda: _ps(10 * GIB, 10 * GIB))
    assert rec["gpu"] == "gpu"
    assert rec["flags"] == []


def test_4c_empty_ps_is_not_loaded(tmp_path):
    _build(tmp_path, lines=[_try()])
    rec = _one(tmp_path, ps_fn=lambda: [])
    assert rec["gpu"] == "not loaded"
    assert rec["flags"] == []


def test_4d_ps_raising_is_unknown(tmp_path):
    _build(tmp_path, lines=[_try()])

    def boom():
        raise OSError("connection refused")

    rec = _one(tmp_path, ps_fn=boom)
    assert rec["gpu"].startswith("unknown")
    assert "Ollama not reachable" in rec["gpu_text"]
    assert rec["flags"] == []


def test_4e_another_model_loaded_is_not_loaded_for_this_one(tmp_path):
    _build(tmp_path, lines=[_try()])
    rec = _one(tmp_path, ps_fn=lambda: [{"name": "other:1b", "size": 5 * GIB, "size_vram": 1 * GIB}])
    assert rec["gpu"] == "not loaded"
    assert rec["flags"] == []


# 5 ---------------------------------------------------------------------------------------------------------------

def test_5_cut_off_last_line_is_skipped_and_earlier_lines_counted(tmp_path):
    cut = _try("e_cut", 2)[: _try("e_cut", 2).index("judge=")]  # ends right after "pass=True "
    _build(tmp_path, lines=[_try(), _try("e_calc")], raw_tail=cut)
    rec = _one(tmp_path)
    assert rec["tries"] == 2


def test_5b_cut_off_in_the_middle_of_infra_value_is_not_an_error(tmp_path):
    cut = _try(ok=False, infra=True)[:-3]  # "... infra=Tr"
    _build(tmp_path, lines=[_try()], raw_tail=cut)
    rec = _one(tmp_path)
    assert rec["tries"] == 1
    assert rec["errors"] == 0


def test_5c_garbage_bytes_in_state_json_fall_back_to_row_json(tmp_path):
    _build(tmp_path, lines=[_try()])
    (tmp_path / f"queue_{TAG}" / "state.json").write_text('{"meta": {', encoding="utf-8")
    rec = _one(tmp_path)
    assert rec["model"] == MODEL
    assert rec["tries"] == 1


def test_5d_missing_state_json_scans_row_json(tmp_path):
    _build(tmp_path, lines=[_try()], state=False)
    rec = _one(tmp_path)
    assert rec["model"] == MODEL
    assert rec["step"] == "full tests"


# finished pack / step ----------------------------------------------------------------------------------------------

def test_6_finished_pack_reports_no_result_from_the_wrote_line(tmp_path):
    _build(tmp_path, lines=[_try(), _try("e_calc"), _wrote(no_result=2, infra_reruns=1)])
    rec = _one(tmp_path)
    assert rec["no_result"] == 2
    assert rec["infra_reruns"] == 1
    assert rec["pack_running"] is False


def test_6b_a_log_without_a_wrote_line_is_a_running_pack(tmp_path):
    _build(tmp_path, lines=[_try(), _try("e_calc")])
    rec = _one(tmp_path)
    assert rec["pack_running"] is True
    assert rec["no_result"] == 0


def test_6c_running_pack_is_the_newest_log_and_older_finished_packs_add_no_result(tmp_path):
    older = "\n".join([_try(), _wrote(no_result=2, infra_reruns=0)]) + "\n"
    _build(tmp_path, lines=[_try("p_a"), _try("p_b"), _try("p_c")], pack="trust", age=5,
           extra_logs={f"packs_Q-work_work_core4_{FOLDER}.log": older})
    rec = _one(tmp_path)
    assert rec["pack"] == "trust"
    assert rec["tries"] == 3
    assert rec["pack_running"] is True
    assert rec["no_result"] == 2


@pytest.mark.parametrize("status,step", [
    ("done", "done"), ("capped", "capped"), ("could not download", "could not download"),
    ("over the card: Stage 2", "over the card"),
])
def test_7_steps_from_the_row_status(tmp_path, status, step):
    _build(tmp_path, lines=[_try()], status=status)
    assert _one(tmp_path)["step"] == step


def test_7b_phase1_done_without_any_pack_log_is_waiting(tmp_path):
    _build(tmp_path, lines=None, status="phase1_done")
    assert _one(tmp_path)["step"] == "waiting"


def test_7c_model_in_state_without_a_folder_is_waiting(tmp_path):
    batch = tmp_path / f"queue_{TAG}"
    batch.mkdir()
    (batch / "state.json").write_text(json.dumps({"meta": {}, "rows": {MODEL: {"model": MODEL, "status": ""}}}),
                                      encoding="utf-8")
    rec = _one(tmp_path)
    assert rec["step"] == "waiting"
    assert rec["last_activity_min"] is None


# 8 ---------------------------------------------------------------------------------------------------------------

def test_8_read_only_nothing_changes_and_nothing_is_added(tmp_path, capsys):
    _build(tmp_path, lines=[_try(), _try("e_calc"), _wrote()], raw_tail="half line", extra_logs={"journal_x.jsonl": "{}\n"})
    before = _hash_tree(tmp_path)
    folders = {str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_dir()}
    mtimes = {p: p.stat().st_mtime for p in tmp_path.rglob("*") if p.is_file()}
    code = qs.main(["--tag", TAG, "--runs-dir", str(tmp_path)], ps_fn=lambda: [], now=NOW)
    assert code in (0, 1)
    assert _hash_tree(tmp_path) == before
    assert {str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_dir()} == folders
    assert {p: p.stat().st_mtime for p in tmp_path.rglob("*") if p.is_file()} == mtimes
    assert capsys.readouterr().out.strip()


# 9 ---------------------------------------------------------------------------------------------------------------

def test_9_missing_run_folder_exits_2_with_a_plain_message(tmp_path, capsys):
    code = qs.main(["--tag", "nope", "--runs-dir", str(tmp_path)], ps_fn=lambda: [], now=NOW)
    out = capsys.readouterr()
    assert code == 2
    assert "nope" in out.out + out.err
    assert "Traceback" not in out.out + out.err


def test_9b_collect_status_raises_for_a_missing_folder_and_does_not_exit(tmp_path):
    with pytest.raises(FileNotFoundError):
        qs.collect_status(tmp_path, "nope", NOW, lambda: [])


def test_9c_output_is_ascii_even_with_odd_model_names(tmp_path, capsys):
    _build(tmp_path, lines=[_try()])
    state = tmp_path / f"queue_{TAG}" / "state.json"
    data = json.loads(state.read_text(encoding="utf-8"))
    data["rows"]["caf\u00e9/\u2713:1"] = {"model": "caf\u00e9/\u2713:1", "status": ""}
    state.write_text(json.dumps(data), encoding="utf-8")
    qs.main(["--tag", TAG, "--runs-dir", str(tmp_path)], ps_fn=lambda: [], now=NOW)
    out = capsys.readouterr().out
    out.encode("ascii")  # raises if any non-ASCII character was printed
    assert MODEL in out


def test_9d_exit_codes_0_and_1(tmp_path, capsys):
    _build(tmp_path, lines=[_try()], age=10)
    assert qs.main(["--tag", TAG, "--runs-dir", str(tmp_path)], ps_fn=lambda: [], now=NOW) == 0
    capsys.readouterr()
    for p in (tmp_path / f"queue_{TAG}").rglob("*"):
        if p.is_file():
            _age(p, 300)
    assert qs.main(["--tag", TAG, "--runs-dir", str(tmp_path)], ps_fn=lambda: [], now=NOW) == 1
    assert "STALLED" in capsys.readouterr().out


def test_9e_runs_dir_defaults_to_the_environment_variable(tmp_path, monkeypatch, capsys):
    _build(tmp_path, lines=[_try()])
    monkeypatch.setenv("LOCALBOT_RUNS_DIR", str(tmp_path))
    assert qs.main(["--tag", TAG], ps_fn=lambda: [], now=NOW) == 0


def test_9f_help_states_the_infra_only_limit(capsys):
    with pytest.raises(SystemExit):
        qs.main(["--help"])
    text = capsys.readouterr().out
    assert "infra" in text
    assert "no_result" in text


# A: the row status wins over "no pack log" -------------------------------------------------------------------------

@pytest.mark.parametrize("status,step", [
    ("done", "done"), ("capped", "capped"), ("could not download", "could not download"),
    ("over the card: Stage 2", "over the card"), ("error", "error"),
])
def test_10_status_wins_over_no_pack_log(tmp_path, status, step):
    _build(tmp_path, lines=None, status=status)
    assert _one(tmp_path)["step"] == step


# B: the default ps_fn lets a failure through (OllamaProvider.ps() swallows every error) -----------------------------

def test_11_default_ps_fn_with_a_raising_transport_says_ollama_not_reachable(tmp_path):
    seen = []

    def transport(url, body, timeout):
        seen.append(url)
        raise OSError("connection refused")

    _build(tmp_path, lines=[_try()])
    rec = _one(tmp_path, ps_fn=qs.default_ps_fn(transport=transport))
    assert seen and seen[0].endswith("/api/ps")
    assert rec["gpu"] == "unknown"
    assert rec["gpu_text"] == "unknown (Ollama not reachable)"
    assert rec["flags"] == []


def test_11b_default_ps_fn_with_a_working_transport_returns_the_models(tmp_path):
    def transport(url, body, timeout):
        return {"models": _ps(10 * GIB, 6 * GIB)}

    _build(tmp_path, lines=[_try()])
    rec = _one(tmp_path, ps_fn=qs.default_ps_fn(transport=transport))
    assert rec["gpu"] == "spill"
    assert "SPILL" in rec["flags"]


def test_11c_default_ps_fn_with_nothing_loaded_is_not_loaded(tmp_path):
    _build(tmp_path, lines=[_try()])
    rec = _one(tmp_path, ps_fn=qs.default_ps_fn(transport=lambda url, body, timeout: {"models": []}))
    assert rec["gpu"] == "not loaded"


# C: size > 0 and size_vram == 0 is SPILL (gpu_fit's rule, reused) --------------------------------------------------

def test_12_all_on_cpu_is_spill(tmp_path):
    _build(tmp_path, lines=[_try()])
    rec = _one(tmp_path, ps_fn=lambda: _ps(10 * GIB, 0))
    assert rec["gpu"] == "spill"
    assert "SPILL" in rec["flags"]


# E: the printed labels --------------------------------------------------------------------------------------------

def test_13_printed_output_labels_the_two_error_columns(tmp_path, capsys):
    _build(tmp_path, lines=[_try(), _try("e_calc"), _wrote(no_result=2)])
    qs.main(["--tag", TAG, "--runs-dir", str(tmp_path)], ps_fn=lambda: [], now=NOW)
    out = capsys.readouterr().out
    assert "infra only while a pack runs" in out
    assert "finished packs only" in out
    assert "no_result" in out


# no_result missing from a pre-C17 "wrote" line is "not recorded", never 0 ---------------------------------------------

_OLD_WROTE = _wrote().replace("no_result=2 ", "")


def test_14_wrote_line_without_no_result_gives_none_and_not_recorded(tmp_path, capsys):
    assert "no_result=" not in _OLD_WROTE
    _build(tmp_path, lines=[_try(), _OLD_WROTE])
    assert _one(tmp_path)["no_result"] is None
    qs.main(["--tag", TAG, "--runs-dir", str(tmp_path)], ps_fn=lambda: [], now=NOW)
    out = capsys.readouterr().out
    assert "missing results (no_result, finished packs only): not recorded" in out


def test_14b_one_old_finished_pack_makes_the_whole_total_not_recorded(tmp_path):
    older = "\n".join([_try(), _wrote(no_result=2)]) + "\n"
    _build(tmp_path, lines=[_try(), _OLD_WROTE], pack="trust", age=5,
           extra_logs={f"packs_Q-work_work_core4_{FOLDER}.log": older})
    assert _one(tmp_path)["no_result"] is None


def test_14c_every_finished_pack_with_the_field_still_sums(tmp_path, capsys):
    older = "\n".join([_try(), _wrote(no_result=2)]) + "\n"
    _build(tmp_path, lines=[_try(), _wrote(no_result=3)], pack="trust", age=5,
           extra_logs={f"packs_Q-work_work_core4_{FOLDER}.log": older})
    assert _one(tmp_path)["no_result"] == 5
    qs.main(["--tag", TAG, "--runs-dir", str(tmp_path)], ps_fn=lambda: [], now=NOW)
    assert "finished packs only): 5" in capsys.readouterr().out


def test_14d_no_finished_pack_at_all_is_zero(tmp_path):
    _build(tmp_path, lines=[_try()])
    assert _one(tmp_path)["no_result"] == 0


def test_14e_infra_reruns_still_sums_when_no_result_is_missing(tmp_path):
    _build(tmp_path, lines=[_try(), _OLD_WROTE])
    assert _one(tmp_path)["infra_reruns"] == 1
