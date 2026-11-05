"""H1-J J7a: no owner's name in Track G's code comments or tests (the public export scan, Track P).
A decision keeps its date: "(owner decision, 2026-10-30)"."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NAME = "pao" + "lo"  # built at run time, so this file does not match itself
G_FILES = [*sorted((REPO / "localbot" / "guard").glob("*.py")),
           *[REPO / "tests" / n for n in ("test_undo.py", "test_delete_policy.py", "test_git_protection.py",
                                          "test_workspace_fs.py", "test_false_positive_pack.py")],
           *sorted((REPO / "tests").glob("test_product_guard*.py")),
           *sorted((REPO / "tests").glob("test_bash_path_rules*.py"))]


def test_track_g_files_name_no_owner():
    hits = [f"{p.relative_to(REPO).as_posix()}:{i}" for p in G_FILES
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if NAME in line.lower()]
    assert len(G_FILES) >= 15 and hits == []
