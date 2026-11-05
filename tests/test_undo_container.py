"""Undo restore against a real container (docs/STAGE1_SPEC.md §5.7, §7 "Destruction inside the
workspace", step S7b.2a) on Podman AND Docker.

Snapshot the seeded work/ tree with ``UndoStore``, then destroy it from inside the locked-down
container — ``rm -rf *``, ``find . -delete``, ``> file`` — mixed with a planted symlink, a
read-only file and folder (``chmod 000``), a deep nested tree, a file over the size cap, and a
bash-made ``nul`` / ``CON``. Restore must bring the snapshot back byte for byte (or name the
file that was too large to store), and it must run only after the reaper confirms no process
is left alive in the container (``quiet()`` = ``alive_pids() == []``).

The host-side restore logic (never follow a link, refused names, the size limit, pruning) is
already covered in tests/test_undo.py; here the same guarantees are exercised with a real
container on both runtimes, so ``chmod 000``, ``ln -s``, ``nul`` and ``CON`` arrive through the
WSL2 bind mount as they would in a real run.
"""
from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import time
import uuid
from pathlib import Path

import pytest

from localbot.config import ContainerConfig, UndoConfig
from localbot.guard.undo import UndoError, UndoStore
from localbot.guard.workspace_fs import WorkspaceFS, extended_path, new_work_folder, remove_tree
from localbot.tools.shell_container import ContainerShell

REPO = Path(__file__).resolve().parents[1]
TRIALS = REPO / "workspaces" / "_trials"
OUTSIDE_SECRET = b"OUTSIDE_SECRET_s7b2a_do_not_copy\n"


from tests.runtimes import RUNTIMES  # noqa: E402  (the one probe + selection, S8i A2)
pytestmark = pytest.mark.skipif(not RUNTIMES, reason="no podman/docker runtime available")


def tree(root: Path) -> dict[str, str]:
    """Everything under root, never following anything: rel -> 'dir' | 'link' | sha256 | 'other'."""
    out: dict[str, str] = {}

    def walk(folder: Path, prefix: str) -> None:
        for e in os.scandir(folder):
            rel, st = prefix + e.name, os.lstat(e.path)
            if getattr(st, "st_file_attributes", 0) & 0x400 or stat.S_ISLNK(st.st_mode):
                out[rel] = "link"
            elif stat.S_ISDIR(st.st_mode):
                out[rel + "/"] = "dir"
                walk(Path(e.path), rel + "/")
            elif stat.S_ISREG(st.st_mode):
                out[rel] = hashlib.sha256(open(e.path, "rb").read()).hexdigest()
            else:
                out[rel] = "other"

    walk(extended_path(root), "")
    return out


class Box:
    def __init__(self, runtime: str, **cfg) -> None:
        self.base = TRIALS / f"s7b2a_{runtime}_{uuid.uuid4().hex[:10]}"
        self.work = self.base / "work"
        self.outside = self.base / "outside"
        self.state = self.base / "state"
        new_work_folder(self.work)  # H1-K D1: case-sensitive before the first file
        for d in (self.outside, self.state):
            d.mkdir(parents=True)
        (self.outside / "secret.txt").write_bytes(OUTSIDE_SECRET)
        self.outside_before = tree(self.outside)
        self.cfg = UndoConfig(store_dir=self.state, **cfg)
        self.store = UndoStore(self.cfg, "bot1", self.work)
        self.shell = ContainerShell(ContainerConfig(runtime=runtime), self.work)
        self.shell.start()

    def put(self, rel: str, data: bytes) -> None:
        p = self.work / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def run(self, cmd: str, timeout: float = 30) -> str:
        return self.shell.run(cmd, timeout)

    def restore(self, steps: int = 1) -> dict:
        return self.store.restore(steps, quiet=lambda: self.shell.alive_pids() == [])

    def assert_outside_untouched(self) -> None:
        assert tree(self.outside) == self.outside_before
        assert os.lstat(self.outside / "secret.txt").st_nlink == 1

    def close(self) -> None:
        self.shell.close()
        remove_tree(self.base)
        assert not self.base.exists()


@pytest.fixture(params=RUNTIMES)
def box(request):
    b = Box(request.param)
    try:
        yield b
    finally:
        b.close()


def seed(b: Box) -> None:
    b.put("crlf.txt", b"one\r\ntwo\nthree\r\n")
    b.put("bin.dat", bytes(range(256)) * 3)
    b.put("empty.txt", b"")
    b.put("naïve été.txt", "café\n".encode())
    b.put("a b.txt", b"space\n")
    b.put("src/pkg/mod.py", b"def f():\n    return 1\n")
    (b.work / "empty").mkdir()
    (b.work / "x" / "y" / "z").mkdir(parents=True)
    deep = "/".join(f"d{i}" for i in range(40))  # 40 levels, ~119 chars: inside the 240 limit
    b.put(f"{deep}/f.txt", b"deep\n")


# --- destruction -> restore -------------------------------------------------------------------


def test_rm_rf_find_delete_and_truncate_then_restore(box):
    seed(box)
    before = tree(box.work)
    box.store.snapshot("bash")
    # rm -rf * (and hidden), find . -delete, and `> file` (crlf.txt comes back empty) + additions.
    box.run("cd /work && rm -rf * .[!.]* 2>/dev/null; find . -delete 2>/dev/null; "
            ": > crlf.txt; printf added > new.txt; mkdir -p newdir/deep; printf x > newdir/deep/f.txt")
    assert box.shell.alive_pids() == []
    assert tree(box.work) != before
    rep = box.restore(1)
    assert tree(box.work) == before
    assert rep["verified"] and rep["restored_to"] == 1 and box.store.seqs() == []
    assert "new.txt" in rep["removed"] and "newdir/" in rep["removed"]
    assert "crlf.txt" in rep["written"]  # truncated (brought back empty) -> rewritten
    assert "empty" in rep["dirs_made"] and "x/y/z" in rep["dirs_made"]
    box.assert_outside_untouched()


@pytest.mark.parametrize("runtime", RUNTIMES)
def test_destruction_mixed_with_link_readonly_deep_large_and_bad_names(runtime):
    b = Box(runtime, skip_files_over_mb=0.01)  # 10 KiB: big.bin will be skipped, not stored
    try:
        seed(b)
        b.put("big.bin", b"B" * 20_000)
        before = tree(b.work)
        snap = b.store.snapshot("bash")
        assert snap.skipped_large == {"big.bin": 20_000}
        # destroy, then plant a symlink, a read-only file and folder (on the host: a container
        # chmod 000 is a no-op), a bash-made nul/CON, and extras.
        b.run("cd /work && rm -rf * .[!.]* 2>/dev/null; find . -delete 2>/dev/null; : > crlf.txt")
        b.run("cd /work && printf new > planted.txt; mkdir ro_dir; printf ro > ro_dir/inner.txt; "
              "ln -s crlf.txt sneak; printf x > nul; printf x > CON; printf added > new.txt")
        os.chmod(b.work / "planted.txt", stat.S_IREAD)
        os.chmod(b.work / "ro_dir", stat.S_IREAD)
        assert b.shell.alive_pids() == []
        rep = b.restore(1)
        # everything except the oversized file comes back byte for byte; the file is named, not restored
        restored = {k: v for k, v in tree(b.work).items() if k.rstrip("/") != "big.bin"}
        assert restored == {k: v for k, v in before.items() if k.rstrip("/") != "big.bin"}
        assert rep["verified"] and rep["not_restored_large"] == ["big.bin"]
        assert not (b.work / "big.bin").exists()
        assert "crlf.txt" in rep["written"]
        # the planted link, refused names and read-only entries were removed, never followed
        assert "sneak" in rep["removed"] and "nul" in rep["removed"] and "CON" in rep["removed"]
        assert "planted.txt" in rep["removed"] and "ro_dir/" in rep["removed"] and "new.txt" in rep["removed"]
        assert "x/y/z" in rep["dirs_made"]  # the deep tree and empty folders came back
        b.assert_outside_untouched()
    finally:
        b.close()


def test_restore_handles_read_only_files_and_folders(box):
    # A container cannot make a host file read-only: `chmod 000` through the WSL2 bind mount is a
    # no-op ("Operation not permitted"; the host sees the file unchanged). The read-only state is
    # planted on the host (os.chmod S_IREAD = the DOS read-only attribute), where it actually
    # lives and where undo.restore must clear it (S7b.1, test_read_only_file_replaced_writable).
    box.put("ro.txt", b"same bytes\n")
    box.put("ro_dir/inner.txt", b"inner\n")
    box.put("gone.txt", b"gone\n")
    before = tree(box.work)
    box.store.snapshot("bash")
    os.chmod(box.work / "ro.txt", stat.S_IREAD)
    os.chmod(box.work / "ro_dir", stat.S_IREAD)
    # the container cannot truncate the read-only file (it sees r-xr-xr-x); it removes gone.txt
    box.run("cd /work && rm -f gone.txt; : > ro.txt 2>/dev/null || true")
    rep = box.restore(1)
    assert tree(box.work) == before and rep["verified"]
    assert "ro.txt" in rep["written"] and "gone.txt" in rep["written"]
    fs = WorkspaceFS(box.work)
    fs.write("ro.txt", "writable again\n")          # the read-only file is writable after restore
    assert (box.work / "ro.txt").read_bytes() == b"writable again\n"
    fs.write("ro_dir/new.txt", "works\n")           # a read-only folder does not block writes
    assert (box.work / "ro_dir" / "new.txt").read_bytes() == b"works\n"
    box.assert_outside_untouched()


def test_restore_refused_until_the_reaper_confirms_clean(box):
    seed(box)
    before = tree(box.work)
    box.store.snapshot("bash")
    box.put("new.txt", b"added\n")
    # a detached process started with a raw exec -d is not reaped by run(): restore must refuse
    src = "import time; time.sleep(30)"
    r = subprocess.run([box.shell.exe, "exec", "-d", box.shell.cid, "python3", "-c", src],
                       capture_output=True, timeout=60)
    assert r.returncode == 0, r.stderr
    deadline = time.time() + 10
    while time.time() < deadline and box.shell.alive_pids() == []:
        time.sleep(0.3)
    assert box.shell.alive_pids() != []  # a live process: quiet() is False
    with pytest.raises(UndoError, match="container process may be running"):
        box.restore(1)
    assert (box.work / "new.txt").read_bytes() == b"added\n"  # nothing changed
    box.run("true")  # the reaper kills the leftover process
    assert box.shell.alive_pids() == []
    box.restore(1)
    assert tree(box.work) == before
    box.assert_outside_untouched()
