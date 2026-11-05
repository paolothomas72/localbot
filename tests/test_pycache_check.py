"""E-8: the lab_ref/ "nothing written" checks fail only on a NEW bytecode cache (tests/_pycache_check.py).

The old checks (``not (lab_ref/lib/__pycache__).exists()``, ``not rglob("__pycache__")``) failed on
any cache, also an old one left in a checkout by an earlier run (the main checkout had
``lab_ref/lib/__pycache__/workspace_fs.cpython-312.pyc`` on 2026-10-31). These run on a temp tree,
never on lab_ref/.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

from tests._pycache_check import cache_snapshot, new_cache_entries


def _tree_with_an_old_cache(root):
    lib = root / "lib"
    (lib / "__pycache__").mkdir(parents=True)
    (lib / "graded_tasks.py").write_text("TASKS = []\n", encoding="utf-8")
    (lib / "__pycache__" / "workspace_fs.cpython-312.pyc").write_bytes(b"old bytecode")
    return lib


def test_an_old_cache_left_in_a_checkout_does_not_fail(tmp_path):
    lib = _tree_with_an_old_cache(tmp_path)
    assert (lib / "__pycache__").exists()  # the old check fails right here, before the step runs
    before = cache_snapshot(tmp_path)
    (lib / "graded_tasks.py").read_text(encoding="utf-8")  # a step that only reads
    assert new_cache_entries(before, cache_snapshot(tmp_path)) == []


def test_a_new_cache_is_reported(tmp_path):
    before = cache_snapshot(tmp_path)
    (tmp_path / "lib" / "__pycache__").mkdir(parents=True)
    (tmp_path / "lib" / "__pycache__" / "graded_tasks.cpython-312.pyc").write_bytes(b"new")
    (tmp_path / "x.pyc").write_bytes(b"loose")
    assert new_cache_entries(before, cache_snapshot(tmp_path)) == [
        "lib/__pycache__/", "lib/__pycache__/graded_tasks.cpython-312.pyc", "x.pyc"]


def test_a_new_file_in_an_old_cache_and_a_rewritten_one_are_reported(tmp_path):
    lib = _tree_with_an_old_cache(tmp_path)
    old = lib / "__pycache__" / "workspace_fs.cpython-312.pyc"
    before = cache_snapshot(tmp_path)
    (lib / "__pycache__" / "graded_tasks.cpython-312.pyc").write_bytes(b"new")
    old.write_bytes(b"rewritten bytecode")
    os.utime(old, ns=(time.time_ns(), time.time_ns() + 10**9))
    assert new_cache_entries(before, cache_snapshot(tmp_path)) == [
        "lib/__pycache__/graded_tasks.cpython-312.pyc", "lib/__pycache__/workspace_fs.cpython-312.pyc"]


def test_a_real_import_that_writes_bytecode_is_caught(tmp_path):
    lib = _tree_with_an_old_cache(tmp_path)
    before = cache_snapshot(tmp_path)
    r = subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path.insert(0, sys.argv[1]); import graded_tasks",
                        str(lib)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-2000:]
    new = new_cache_entries(before, cache_snapshot(tmp_path))
    assert len(new) == 1 and new[0].startswith("lib/__pycache__/graded_tasks.") and new[0].endswith(".pyc"), new
