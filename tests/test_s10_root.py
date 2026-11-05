"""BRIEF_004 P5: ``docs/s10_recompute.py`` finds its own checkout and says so when the private run logs
(``state/runs``) are missing. The script is copied into a throwaway tree; nothing in the repo is touched."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
S10 = REPO / "docs" / "s10_recompute.py"
NOTICE = ("The private run logs (state/runs) aren't in this copy, so section 11 and the batch/log columns of "
          "section 13 are skipped. No test number depends on them.")


def _copy_tree(base: Path, with_runs: bool) -> Path:
    """A checkout-shaped tree: the script, the manifest and every committed result file."""
    (base / "docs").mkdir(parents=True)
    shutil.copy2(S10, base / "docs" / "s10_recompute.py")
    shutil.copy2(REPO / "docs" / "runs.json", base / "docs" / "runs.json")
    shutil.copytree(REPO / "results" / "stage1", base / "results" / "stage1")
    if with_runs:
        (base / "state" / "runs").mkdir(parents=True)
    return base


def _run(script: Path, *args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(script), *args], cwd=cwd, capture_output=True,
                          text=True, encoding="utf-8", timeout=600)


def test_the_script_names_no_personal_folder():
    text = S10.read_text(encoding="utf-8")
    owner, users = "pao" + "lo", "c:" + chr(92) * 2 + "users"  # spelled in pieces: this file is exported too
    assert owner not in text.lower() and users not in text.lower()


def test_with_no_argument_it_uses_the_checkout_it_sits_in(tmp_path):
    root = _copy_tree(tmp_path / "copy", with_runs=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    r = _run(root / "docs" / "s10_recompute.py", cwd=elsewhere)
    assert r.returncode == 0, r.stderr[-400:]
    assert f"Results root: `{root.resolve()}`" in r.stdout
    assert NOTICE not in r.stdout  # state/runs is there


def test_without_state_runs_it_exits_0_and_says_what_is_skipped_at_the_top_and_in_section_11(tmp_path):
    root = _copy_tree(tmp_path / "copy", with_runs=False)
    r = _run(root / "docs" / "s10_recompute.py", cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-400:]
    assert r.stdout.count(NOTICE) == 2
    top, rest = r.stdout.split("## 0. Files read", 1)
    assert NOTICE in top
    sec11 = rest.split("## 11. ", 1)[1].split("\n## ", 1)[0]
    assert NOTICE in sec11