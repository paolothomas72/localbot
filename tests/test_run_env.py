"""S8f.1b (amendments C4): run_env git `dirty` flag must ignore untracked files under results/
(result JSONs a run writes must not flip it) while still flagging an untracked source file or a
modified tracked file. git_record takes a cwd so the three cases run in a throwaway repo."""
from __future__ import annotations

import subprocess

import pytest

from localbot.exam.runner import git_record


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "localbot").mkdir(parents=True)
    (r / "results").mkdir()
    (r / "localbot" / "x.py").write_text("x = 1\n", encoding="utf-8")
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "t")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "init")
    return r


def test_clean_repo_not_dirty(repo):
    rec = git_record(cwd=repo)
    assert rec["dirty"] is False and rec["commit"]


def test_untracked_under_results_is_not_dirty(repo):
    (repo / "results" / "run_20261030.json").write_text("{}\n", encoding="utf-8")
    assert git_record(cwd=repo)["dirty"] is False


def test_untracked_under_localbot_is_dirty(repo):
    (repo / "localbot" / "new.py").write_text("y = 2\n", encoding="utf-8")
    assert git_record(cwd=repo)["dirty"] is True


def test_modified_tracked_file_is_dirty(repo):
    (repo / "localbot" / "x.py").write_text("x = 999\n", encoding="utf-8")
    assert git_record(cwd=repo)["dirty"] is True


def test_results_untracked_does_not_mask_a_source_change(repo):
    (repo / "results" / "run.json").write_text("{}\n", encoding="utf-8")   # ignored
    (repo / "localbot" / "x.py").write_text("x = 2\n", encoding="utf-8")   # counts
    assert git_record(cwd=repo)["dirty"] is True
