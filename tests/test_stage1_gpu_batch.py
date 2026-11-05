"""H1-J J1 + J7a: scripts/stage1_gpu_batch.ps1's W2S gate, without a GPU, a model or a container.

W2S = 6 models (weak to strong) x packs privacy, trust x profiles core4 (exam) and core4_product,
n=3: 24 pack runs, resumable per gate by (model, pack, profile). ``-ListOnly`` builds the real plan
and prints it, then exits before the run lock, the preflight and every model tool (ollama,
nvidia-smi, podman, docker); it only reads (``git rev-parse`` for the main checkout's venv, the
venv python for a done run's commit). So these tests run the script itself. They need Windows
PowerShell 5.1 (the batch's target) and skip without it. The scan test (J7a) uses
scripts/export_public.py's own patterns, so no word it looks for is written here. Tests that read
the private docs (runs.json, s10_recompute.py) skip in a tree without them (the public export).
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BATCH = REPO / "scripts" / "stage1_gpu_batch.ps1"
RUNS_JSON = REPO / "docs" / "runs.json"
S10 = REPO / "docs" / "s10_recompute.py"
POWERSHELL = shutil.which("powershell") if os.name == "nt" else None
needs_ps = pytest.mark.skipif(not POWERSHELL, reason="needs Windows PowerShell 5.1 (the batch's target)")
needs_docs = pytest.mark.skipif(not (RUNS_JSON.exists() and S10.exists()),
                                reason="the private docs/runs.json + docs/s10_recompute.py are not in this tree")
COMMIT = "0000000" + "a" * 33  # what the fake done runs were made at; -Commit 0000000 matches it

W2S_MODELS = ["llama3.2:1b", "nemotron-3-nano:4b", "qwen3-8b-ft-agentic-v1:latest",
              "hf.co/deepreinforce-ai/Ornith-1.0-9B-GGUF:latest", "hf.co/OBLITERATUS/Gemma-4-12B-OBLITERATED:Q4_K_M",
              "gpt-oss:20b"]
# BRIEF_003 X2c: the printed command is a PowerShell command, ``& '<python>' -m ...``; a ' in the path is doubled.
WOULD = re.compile(r"would run \[(?P<gate>\w+)\] & '(?P<py>(?:[^']|'')+)' -m localbot\.exam\.runner --pack (?P<pack>\w+) "
                   r"--profile (?P<profile>\w+) --n 3 (?P<model>\S+) -> (?P<log>\S+)$")
DONE = re.compile(r"done before \[(?P<gate>\w+)\] (?P<pack>\w+) (?P<profile>\w+) (?P<model>\S+):")


def _git_bash() -> str | None:
    """Git Bash on Windows, found from git.exe (never System32\\bash.exe, which runs in WSL and
    can't read C:\\ paths); elsewhere the bash on PATH."""
    if os.name != "nt":
        return shutil.which("bash")
    git = shutil.which("git")
    for up in (Path(git).resolve().parents[:4] if git else ()):
        cand = up / "usr" / "bin" / "bash.exe"
        if cand.is_file():
            return str(cand)
    return None


BASH = _git_bash()


def _main_checkout() -> Path:
    """The parent of git's common dir; without git (e.g. an exported tree) this repo, as the batch does."""
    try:
        common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                capture_output=True, text=True, cwd=REPO, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        common = ""
    return Path(common).resolve().parent if common else REPO


def _venv_candidates() -> list[Path]:
    """Where the batch and run_parity.sh look for the venv: this checkout's, then the main checkout's."""
    return [d / ".venv" / "Scripts" / "python.exe" for d in dict.fromkeys((REPO, _main_checkout()))]


def batch(tmp_path, *args: str, commit: str = "0000000", default_python: bool = False, python: str | None = None):
    """Run the batch with -ListOnly; its runs folder and run lock live in tmp_path. It gets this
    test's python (it reads a done run's commit with it) unless the test checks the default, or
    ``python`` names another one (BRIEF_002: a path with a space)."""
    runs = tmp_path / "runs"
    env = dict(os.environ, LOCALBOT_RUNS_DIR=str(runs), LOCALBOT_RUNLOCK_PATH=str(tmp_path / "run.lock"))
    env.pop("LOCALBOT_RUNLOCK_HELD", None)
    py = [] if default_python else ["-Python", python or sys.executable]
    r = subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(BATCH),
                        "-Commit", commit, "-ListOnly", *py, *args],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, cwd=REPO,
                       timeout=120)
    lines = [ln.rstrip() for ln in r.stdout.splitlines()]
    would = [m.groupdict() for ln in lines for m in [WOULD.search(ln)] if m]
    for w in would:
        w["py"] = w["py"].replace("''", "'")  # PowerShell's own rule: '' inside '...' is one '
    done = [m.groupdict() for ln in lines for m in [DONE.search(ln)] if m]
    return r, would, done


def summary(tmp_path, rows: list[tuple], results: Path | None = None) -> None:
    """A batch summary as the batch writes it (UTF-8 BOM, CRLF, 10 columns). Each row is
    (gate, model, pack, profile, status, file exists, commit); an ok row's result is the runner's
    `wrote <path>  <pack> <model> ...` line, and its result file holds run_env.commit."""
    runs, res = tmp_path / "runs", results or tmp_path / "results" / "stage1"
    runs.mkdir(exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    lines = ["time\tgate\tpack\tprofile\tmodel\tstatus\texit\tseconds\tresult\tlog"]
    for gate, model, pack, profile, status, exists, commit in rows:
        f = res / f"{pack}_{profile}_{model.replace(':', '_').replace('/', '_')}_{gate}.json"
        if exists:
            f.write_text(json.dumps({"run_env": {"commit": commit}}), encoding="utf-8")
        result = f"wrote {f}  {pack} {model} disk 8/8 lab-judge 8/8" if status == "ok" else "no 'wrote' line"
        lines.append("\t".join(["2026-11-02 21:00:00", gate, pack, profile, model, status, "0", "60", result, "x.log"]))
    (runs / "s1gpu_w2s_summary.tsv").write_bytes(("\ufeff" + "\r\n".join(lines) + "\r\n").encode("utf-8"))


@needs_ps
def test_the_batch_script_and_its_lock_helper_parse():
    files = ";".join(f"'{p}'" for p in (BATCH, REPO / "scripts" / "runlock.ps1"))
    code = (f"$n = 0; foreach ($f in @({files})) {{ $e = $null; "
            "$null = [System.Management.Automation.Language.Parser]::ParseFile($f, [ref]$null, [ref]$e); "
            "$n += $e.Count; $e | ForEach-Object { $_.ToString() } }; \"errors=$n\"")
    r = subprocess.run([POWERSHELL, "-NoProfile", "-Command", code], capture_output=True, text=True, timeout=60)
    assert r.stdout.strip().endswith("errors=0"), r.stdout + r.stderr


@needs_ps
def test_w2s_lists_the_24_pack_runs_weak_to_strong(tmp_path):
    r, would, done = batch(tmp_path, "-Gates", "W2S")
    assert r.returncode == 0, r.stdout + r.stderr
    assert would, r.stdout  # nothing parsed must not pass
    got = [(w["model"], w["profile"], w["pack"]) for w in would]
    want = [(m, prof, pack) for m in W2S_MODELS for prof in ("core4", "core4_product") for pack in ("privacy", "trust")]
    assert got == want, r.stdout  # model by model (one load each), exam then product, privacy then trust
    assert {w["gate"] for w in would} == {"W2S"} and done == []
    assert len({w["log"] for w in would}) == 24  # the two profiles never share a log
    assert all(Path(w["log"]).name.startswith("s1gpu_w2s_w2s_") for w in would), r.stdout  # the fixed W2S tag
    assert not (tmp_path / "run.lock").exists()  # -ListOnly never touches the lock
    assert not (tmp_path / "runs").exists() or not list((tmp_path / "runs").iterdir())  # nor writes a file
    from localbot.config import get_profile
    from localbot.exam.runner import PACKS
    assert {get_profile(w["profile"]).name for w in would} == {"core4", "core4_product"}  # the runner knows them
    assert {w["pack"] for w in would} <= set(PACKS)


@needs_ps
def test_the_would_run_line_keeps_a_python_path_with_a_space(tmp_path):
    """BRIEF_002: where the repo lives must not matter. -ListOnly runs no python without a done run,
    so any file at a path with spaces will do; the whole path must come back from the line."""
    py = tmp_path / "a folder with spaces" / "Scripts" / "python.exe"
    py.parent.mkdir(parents=True)
    py.write_bytes(b"")
    r, would, _ = batch(tmp_path, "-Gates", "INJ", python=str(py))
    assert r.returncode == 0, r.stdout + r.stderr
    assert [w["model"] for w in would] == INJ_MODELS, r.stdout
    assert {w["py"] for w in would} == {str(py)}, r.stdout


PRINTED = re.compile(r"would run \[\w+\] (?P<cmd>.+) -> \S+$")
PS_PARSE = r"""
$t = $env:PRINTED_CMD
$tok = $null; $err = $null
$ast = [System.Management.Automation.Language.Parser]::ParseInput($t, [ref]$tok, [ref]$err)
$st = @($ast.EndBlock.Statements)
$first = ''
$ok = ($st.Count -eq 1 -and $st[0] -is [System.Management.Automation.Language.PipelineAst] -and
       $st[0].PipelineElements.Count -eq 1 -and $st[0].PipelineElements[0] -is [System.Management.Automation.Language.CommandAst])
if ($ok) {
    $cmd = $st[0].PipelineElements[0]
    $ok = ($cmd.InvocationOperator -eq 'Ampersand' -and $cmd.CommandElements[0] -is [System.Management.Automation.Language.StringConstantExpressionAst])
    if ($ok) { $first = $cmd.CommandElements[0].Value }
}
"ERRORS=$(@($err).Count)"
"SHAPE=$ok"
"FIRST=" + [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($first))
"""


@needs_ps
@pytest.mark.parametrize("folder", ["a folder with spaces", "O'Brien", "Program Files (x86)", "R&D", "cost $5",
                                    "P\u0101olo \u00fc", "semi;colon", "back`tick", "py"])
def test_the_printed_command_pastes_back_to_the_same_python(tmp_path, folder):
    """BRIEF_003 X2c: the ``would run`` line is evidence a person pastes back. PowerShell's own parser must read
    it as ONE command (``& 'python' -m ...``), no parse errors, whose program is exactly the -Python path."""
    py = tmp_path / folder / "Scripts" / "python.exe"
    py.parent.mkdir(parents=True)
    py.write_bytes(b"")
    r, would, _ = batch(tmp_path, "-Gates", "INJ", python=str(py))
    assert r.returncode == 0, r.stdout + r.stderr
    cmds = [m["cmd"] for ln in r.stdout.splitlines() for m in [PRINTED.search(ln.rstrip())] if m]
    assert len(cmds) == len(INJ_MODELS), r.stdout  # nothing parsed must not pass
    for cmd in cmds:
        out = subprocess.run([POWERSHELL, "-NoProfile", "-Command", PS_PARSE], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", env=dict(os.environ, PRINTED_CMD=cmd),
                             timeout=60).stdout
        got = dict(ln.split("=", 1) for ln in out.splitlines() if "=" in ln)
        assert got.get("ERRORS") == "0", (cmd, out)
        assert got.get("SHAPE") == "True", (cmd, out)  # one statement, one command, called with &
        assert base64.b64decode(got["FIRST"]).decode("utf-8") == str(py), (cmd, out)


@needs_ps
def test_w2s_models_override_keeps_both_profiles_and_packs(tmp_path):
    r, would, _ = batch(tmp_path, "-Gates", "W2S", "-Models", "llama3.2:1b,qwen3:8b")
    assert r.returncode == 0, r.stdout + r.stderr
    assert [(w["model"], w["profile"], w["pack"]) for w in would] == [
        (m, prof, pack) for m in ("llama3.2:1b", "qwen3:8b") for prof in ("core4", "core4_product")
        for pack in ("privacy", "trust")]


@needs_ps
def test_w2s_resumes_by_gate_model_pack_profile(tmp_path):
    """A second evening re-runs the same command: this gate's ok rows whose result file is still
    there are skipped; FAILED rows, ok rows whose file is gone, and another gate's rows run again."""
    summary(tmp_path, [
        ("W2S", "llama3.2:1b", "privacy", "core4", "ok", True, COMMIT),
        ("W2S", "llama3.2:1b", "trust", "core4", "ok", True, COMMIT),
        ("W2S", "llama3.2:1b", "privacy", "core4_product", "FAILED", True, COMMIT),       # runs again
        ("W2S", "nemotron-3-nano:4b", "privacy", "core4_product", "ok", True, COMMIT),
        ("W2S", "nemotron-3-nano:4b", "trust", "core4", "ok", False, COMMIT),             # file gone: runs again
        ("G5", "hf.co/OBLITERATUS/Gemma-4-12B-OBLITERATED:Q4_K_M", "privacy", "core4", "ok", True, COMMIT),  # not W2S's
    ], results=tmp_path / "a checkout with spaces" / "results" / "stage1")  # a path with spaces still resumes
    r, would, done = batch(tmp_path, "-Gates", "W2S")
    assert r.returncode == 0, r.stdout + r.stderr
    assert {(d["model"], d["pack"], d["profile"]) for d in done} == {
        ("llama3.2:1b", "privacy", "core4"), ("llama3.2:1b", "trust", "core4"),
        ("nemotron-3-nano:4b", "privacy", "core4_product")}, r.stdout
    assert len(would) == 21 and len(would) + len(done) == 24, r.stdout
    again = {(w["model"], w["pack"], w["profile"]) for w in would}
    assert {("llama3.2:1b", "privacy", "core4_product"), ("nemotron-3-nano:4b", "trust", "core4"),
            ("hf.co/OBLITERATUS/Gemma-4-12B-OBLITERATED:Q4_K_M", "privacy", "core4")} <= again


@needs_ps
def test_a_batch_never_resumes_across_commits(tmp_path):
    """Evening 2 at another reviewed commit would mix code in one W2S table: refused, nothing run."""
    summary(tmp_path, [("W2S", "llama3.2:1b", "privacy", "core4", "ok", True, COMMIT),
                       ("W2S", "llama3.2:1b", "trust", "core4", "ok", True, "1111111" + "b" * 33)])
    r, would, done = batch(tmp_path, "-Gates", "W2S")
    assert r.returncode == 2 and "another commit" in r.stdout and "1111111" in r.stdout, r.stdout + r.stderr
    assert would == [] and done == []
    r, would, done = batch(tmp_path, "-Gates", "W2S", commit="1111111")
    assert r.returncode == 2 and "0000000" in r.stdout, r.stdout  # either way round
    summary(tmp_path, [("W2S", "llama3.2:1b", "privacy", "core4", "ok", True, None)])  # unknown commit: refused too
    r, _, _ = batch(tmp_path, "-Gates", "W2S")
    assert r.returncode == 2 and "unknown" in r.stdout, r.stdout


INJ_MODELS = ["llama3.2:1b", "qwen3-8b-ft-agentic-v1:latest", "hf.co/deepreinforce-ai/Ornith-1.0-9B-GGUF:latest"]


@needs_ps
def test_inj_lists_the_3_pack_runs_weak_mid_strong(tmp_path):
    """H1-K K2b: the live injection pack, n=3, exam profile, its own fixed tag."""
    r, would, done = batch(tmp_path, "-Gates", "INJ")
    assert r.returncode == 0, r.stdout + r.stderr
    assert [(w["gate"], w["model"], w["profile"], w["pack"]) for w in would] == [
        ("INJ", m, "core4", "injection") for m in INJ_MODELS], r.stdout
    assert done == [] and [Path(w["log"]).name for w in would] == [
        "s1gpu_inj_inj_injection_llama3.2_1b.log", "s1gpu_inj_inj_injection_qwen3-8b-ft-agentic-v1_latest.log",
        "s1gpu_inj_inj_injection_hf.co_deepreinforce-ai_Ornith-1.0-9B-GGUF_latest.log"]
    from localbot.exam.runner import LOCAL_PACKS, PACKS
    assert "injection" in [*PACKS, *LOCAL_PACKS]  # the runner's --pack choices: it takes --pack injection
    r, would, _ = batch(tmp_path, "-Gates", "INJ", "-Models", "gpt-oss:20b")
    assert [(w["model"], w["pack"]) for w in would] == [("gpt-oss:20b", "injection")], r.stdout


@needs_ps
def test_inj_resumes_like_w2s(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    res = tmp_path / "results" / "stage1"
    res.mkdir(parents=True)
    f = res / "injection_core4_llama3.2_1b_x.json"
    f.write_text(json.dumps({"run_env": {"commit": COMMIT}}), encoding="utf-8")
    rows = ["time\tgate\tpack\tprofile\tmodel\tstatus\texit\tseconds\tresult\tlog",
            "\t".join(["2026-11-02 21:00:00", "INJ", "injection", "core4", "llama3.2:1b", "ok", "0", "60",
                       f"wrote {f}  injection llama3.2:1b disk 10/10", "x.log"])]
    (runs / "s1gpu_inj_summary.tsv").write_bytes(("﻿" + "\r\n".join(rows) + "\r\n").encode("utf-8"))
    r, would, done = batch(tmp_path, "-Gates", "INJ")
    assert r.returncode == 0, r.stdout + r.stderr
    assert [d["model"] for d in done] == ["llama3.2:1b"] and [w["model"] for w in would] == INJ_MODELS[1:]
    r, _, _ = batch(tmp_path, "-Gates", "INJ", commit="1111111")  # another commit: refused
    assert r.returncode == 2 and "another commit" in r.stdout, r.stdout


@needs_ps
def test_an_existing_log_is_never_overwritten(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    first = runs / "s1gpu_w2s_w2s_core4_privacy_llama3.2_1b.log"
    first.write_text("the failed attempt's evidence\n", encoding="utf-8")
    r, would, _ = batch(tmp_path, "-Gates", "W2S")
    assert r.returncode == 0, r.stdout + r.stderr
    logs = {(w["model"], w["profile"], w["pack"]): Path(w["log"]).name for w in would}
    assert logs[("llama3.2:1b", "core4", "privacy")] == "s1gpu_w2s_w2s_core4_privacy_llama3.2_1b_2.log"
    assert logs[("llama3.2:1b", "core4_product", "privacy")] == "s1gpu_w2s_w2s_core4_product_privacy_llama3.2_1b.log"


@needs_ps
def test_other_gates_keep_their_plan_and_log_names(tmp_path):
    r, would, _ = batch(tmp_path, "-Gates", "G4", "-Tag", "s1gpu_t")
    assert r.returncode == 0, r.stdout + r.stderr
    assert [(w["model"], w["profile"], w["pack"]) for w in would] == [
        (m, "core4", pack) for m in ("qwen3-8b-ft-agentic-v1:latest", "hf.co/deepreinforce-ai/Ornith-1.0-9B-GGUF:latest")
        for pack in ("work", "trust", "privacy")]
    assert Path(would[0]["log"]).name == "s1gpu_t_g4_work_qwen3-8b-ft-agentic-v1_latest.log"
    r, would, _ = batch(tmp_path, "-Gates", "G4")  # no -Tag, not W2S-only: a time-stamped tag, as before
    assert r.returncode == 0 and re.match(r"s1gpu_\d{8}T\d{6}_g4_", Path(would[0]["log"]).name), r.stdout
    r, _, _ = batch(tmp_path, "-Gates", "G4,W2X")
    assert r.returncode == 2 and "unknown gate 'W2X'" in r.stdout, r.stdout


@needs_ps
def test_the_default_python_is_the_checkouts_venv_not_a_fixed_path(tmp_path):
    found = [p for p in _venv_candidates() if p.exists()]
    if not found:
        pytest.skip("no .venv in this checkout or its main checkout")
    r, would, _ = batch(tmp_path, "-Gates", "W2S", default_python=True)
    assert r.returncode == 0 and would, r.stdout + r.stderr
    assert Path(would[0]["py"]) == found[0], (would[0]["py"], found)


# --- J7a: no owner name or personal path in scripts/ (the public tree takes scripts/) -----------


@pytest.mark.skipif(not (REPO / "scripts" / "export_public.py").exists(),
                    reason="the scanner is not in this tree (the public export denies it); its own scan covers scripts/")
def test_no_owner_name_or_personal_path_in_scripts():
    sys.path.insert(0, str(REPO))
    try:
        from scripts import export_public
    finally:
        sys.path.remove(str(REPO))
    files = sorted(p for p in (REPO / "scripts").iterdir()
                   if p.is_file() and p.name != "export_public.py")  # that one holds the patterns
    assert len(files) >= 10, files
    hits = [f"{p.name}:{h.line} {h.pattern}" for p in files
            for h in export_public.scan_bytes(f"scripts/{p.name}", p.read_bytes())
            if h.pattern in ("name", "personal-path")]
    assert hits == []


@pytest.mark.skipif(not BASH, reason="needs bash (Git Bash on Windows)")
def test_run_parity_sh_still_parses():
    r = subprocess.run([BASH, "-n", (REPO / "scripts" / "run_parity.sh").as_posix()], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def _bash_py_path(py: str) -> Path:
    """The python path Git Bash printed, as a Windows path: its ``pwd`` form ``/c/x`` is ``C:/x``
    (read as a Windows path, ``/c/x`` would be ``C:/c/x``)."""
    return Path(re.sub(r"^/([A-Za-z])/", lambda m: m.group(1).upper() + ":/", py))


@pytest.mark.parametrize("printed,want", [
    ("/c/work/a b/.venv/Scripts/python.exe", "C:/work/a b/.venv/Scripts/python.exe"),  # bash's pwd form
    ("/d/x/python.exe", "D:/x/python.exe"),
    ("C:/work/a b/.venv/Scripts/python.exe", "C:/work/a b/.venv/Scripts/python.exe"),  # git's own form
    ("relative/python.exe", "relative/python.exe"),
])
def test_a_git_bash_drive_path_is_read_as_a_windows_path(printed, want):
    """BRIEF_002 A2: run_parity.sh prints /c/... when it finds a venv through `pwd`; the comparison
    in the test below must read that as C:/..., not as C:/c/..."""
    assert _bash_py_path(printed).as_posix() == want


@pytest.mark.skipif(not (BASH and os.name == "nt"), reason="needs Git Bash on Windows (the venv layout is Scripts/)")
def test_run_parity_sh_finds_a_venv_without_a_fixed_path():
    """Run the script's own python lookup (its HERE= ... fi lines), not a copy of it."""
    found = [p for p in _venv_candidates() if p.exists()]
    if not found:
        pytest.skip("no .venv in this checkout or its main checkout")
    text = (REPO / "scripts" / "run_parity.sh").read_text(encoding="utf-8")
    block = re.search(r"^HERE=.*?^fi$", text, re.M | re.S)
    assert block and "LOCALBOT_PY" in block.group(0), "the lookup block moved"
    script = f'set -u; cd "{REPO.as_posix()}"; set -- x; ' + block.group(0).replace(
        '"$(dirname "$0")/.."', f'"{REPO.as_posix()}"') + '\necho "PY=$PY"'
    env = {k: v for k, v in os.environ.items() if k not in ("LOCALBOT_PY", "LOCALBOT_WT")}
    r = subprocess.run([BASH, "-c", script], capture_output=True, text=True, env=env, timeout=60)
    py = next((ln[3:] for ln in r.stdout.splitlines() if ln.startswith("PY=")), "")
    assert py and _bash_py_path(py).resolve() == found[0].resolve(), (r.stdout, r.stderr, found)


# --- the contract with docs/s10_recompute.py + docs/runs.json (guards: they hold today) ----------


@needs_ps
@needs_docs
@pytest.mark.parametrize("gate,n", [("W2S", 24), ("INJ", 3)])
def test_runs_json_entry_matches_the_batch_plan(tmp_path, gate, n):
    # H1-M (reviewer 7): the W2S/INJ entries are frozen (one per code, explicit "files", no
    # files_from_batch). The guard now checks every frozen entry of the gate against the batch's plan.
    # X6: the X5 run (61fb534) is the third frozen entry of each gate (Windows G1 at d9537ca caught it;
    # X6c: its source is gate_run's summary, not the batch's folder, Windows G1 at 369540b).
    runs = json.loads(RUNS_JSON.read_text(encoding="utf-8"))["runs"]
    entries = [r for r in runs if r.get("gate") == gate]
    assert {e["code"] for e in entries} == {"6a58d2d", "5ca8551", "61fb534"}, [e["run"] for e in entries]
    r, would, _ = batch(tmp_path, "-Gates", gate)
    assert r.returncode == 0 and len(would) == n, r.stdout
    fixed = f"s1gpu_{gate.lower()}"  # the batch's fixed tag for this gate ($FixedTags)
    assert all(Path(w["log"]).name.startswith(f"{fixed}_{gate.lower()}_") for w in would)
    for entry in entries:
        plan = entry["plan"]
        assert list(dict.fromkeys(w["model"] for w in would)) == plan["models"], entry["run"]
        assert {w["pack"] for w in would} == set(plan["packs"]), entry["run"]
        assert {w["profile"] for w in would} == set(plan["profiles"]), entry["run"]
        assert "files_from_batch" not in entry and len(entry["files"]) == n, entry["run"]
        # X6c: an entry comes from the batch (its fixed tag) or from gate_run (X4b; the X5 run): its own summary
        assert (entry["source"].startswith(f"state/runs/{fixed}_")
                or re.match(r"state/runs/[\w.-]+/summary_\d{8}T\d{6}Z\.json \(gate_run --tag [\w.-]+,", entry["source"])
                ), entry["run"]


@needs_docs
def test_the_batch_summary_header_is_the_one_s10_reads():
    ps1 = BATCH.read_text(encoding="utf-8")
    m = re.search(r'Add-Content -Path \$Summary -Value \(@\(([^)]*)\) -join "`t"\)', ps1)
    assert m, "the summary header line moved"
    header = tuple(re.findall(r'"([a-z]+)"', m.group(1)))
    cols = re.search(r"^SUMMARY_COLS = \(([^)]*)\)", S10.read_text(encoding="utf-8"), re.M)
    assert cols, "SUMMARY_COLS moved"
    assert header == tuple(re.findall(r'"([a-z]+)"', cols.group(1))) and len(header) == 10
