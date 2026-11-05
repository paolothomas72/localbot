"""find_caller cost on a big disk tree (S4 watch item from S3).

The lab's find_caller reads every file when a bare filename that is an existing definer is
written (CallerGuard.veto_write). On a DiskView that is a full read of the tree. This times
it on real NTFS and checks the worst case stays bounded; the numbers go in BUILD_LOG.
"""
from __future__ import annotations

import time

import pytest

from localbot.guard.product import ProductGuard, find_caller
from localbot.guard.workspace_fs import WorkspaceFS
from localbot.tools.fs import DiskWorkspace

LIB = "def greet():\n    return 'hi'\n"


@pytest.mark.parametrize("n_files", [200, 2000])
def test_find_caller_cost_on_disk(disk, n_files):
    body = "# filler\n" + "value = 1\n" * 200  # ~2 KB each
    seed = {"lib.py": LIB}
    seed.update({f"pkg{i // 100:02}/mod{i:05}.py": body for i in range(n_files)})
    seed["zz/app.py"] = "from lib import greet\nprint(greet())\n"  # the caller is the last file
    ws = disk.ws(seed)
    t0 = time.perf_counter()
    assert find_caller(ws.view, "lib.py") == "zz/app.py"
    t_find = time.perf_counter() - t0
    t0 = time.perf_counter()
    why = ProductGuard().check("write", {"path": "callers.txt", "content": "lib.py"}, ws.view)
    t_check = time.perf_counter() - t0
    assert why == ("harness: lib.py defines the function; zz/app.py calls it. "
                   "Write the caller filename, not the definer.")
    t0 = time.perf_counter()
    assert ProductGuard().check("write", {"path": "notes.txt", "content": "hello world"}, ws.view) is None
    t_plain = time.perf_counter() - t0
    print(f"\n[find_caller] files={n_files + 2} bytes~{(n_files * len(body)) // 1024}KB "
          f"find_caller={t_find * 1000:.0f}ms full_check={t_check * 1000:.0f}ms plain_write_check={t_plain * 1000:.1f}ms")
    assert t_check < 30.0  # generous bound; the log records the real numbers


def test_find_caller_only_runs_for_bare_filename_of_existing_file(disk, monkeypatch):
    ws = disk.ws({"lib.py": LIB, "app.py": "greet()\n"})
    calls = []
    import localbot.guard.product as product
    real = product.find_caller
    monkeypatch.setattr(product, "find_caller", lambda f, d: calls.append(d) or real(f, d))
    g = ProductGuard()
    g.check("write", {"path": "x.txt", "content": "some text"}, ws.view)
    g.check("write", {"path": "x.txt", "content": "missing.py"}, ws.view)
    g.check("write", {"path": "x.txt", "content": "lib.py"}, ws.view)
    assert calls == ["lib.py"]
    assert isinstance(ws.fs, WorkspaceFS) and isinstance(ws, DiskWorkspace)
