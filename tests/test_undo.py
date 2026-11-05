"""Undo store (docs/STAGE1_SPEC.md §5.7, step S7b.1): host side only, no container needed.

Snapshot / restore byte for byte, undo steps, the size limit and the store cap, and the rule
that restore never follows a link: junctions, symlinks, dangling symlinks and hard links planted
in work/ are removed (the outside/ bait hash never changes), refused names are removed, files
are created fresh. Also the loop hook, the runner wiring and ``python -m localbot undo``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import stat
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from localbot.config import UndoConfig, get_profile
from localbot.guard.undo import UndoError, UndoStore, scan
from localbot.guard.workspace_fs import WorkspaceError, WorkspaceFS, extended_path, remove_tree

REPO = Path(__file__).resolve().parents[1]
TRIALS = REPO / "workspaces" / "_trials"
WIN = os.name == "nt"
QUIET = lambda: True  # noqa: E731  (no container in these tests)
OUTSIDE_SECRET = b"OUTSIDE_SECRET_s7b1_do_not_copy\n"


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
    def __init__(self, **cfg) -> None:
        self.base = TRIALS / f"s7b1_{uuid.uuid4().hex[:10]}"
        self.work, self.outside, self.state = self.base / "work", self.base / "outside", self.base / "state"
        for d in (self.work, self.outside, self.state):
            d.mkdir(parents=True)
        (self.outside / "secret.txt").write_bytes(OUTSIDE_SECRET)
        (self.outside / "dir").mkdir()
        (self.outside / "dir" / "inner.txt").write_bytes(OUTSIDE_SECRET)
        self.cfg = UndoConfig(store_dir=self.state, **cfg)
        self.store = UndoStore(self.cfg, "bot1", self.work)
        self.outside_before = tree(self.outside)

    def put(self, rel: str, data: bytes) -> None:
        p = self.work / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def assert_outside_untouched(self) -> None:
        assert tree(self.outside) == self.outside_before
        assert os.lstat(self.outside / "secret.txt").st_nlink == 1

    def store_has(self, data: bytes) -> bool:
        return any(data in p.read_bytes() for p in (self.store.root / "objects").glob("*/*"))


@pytest.fixture
def box():
    b = Box()
    try:
        yield b
    finally:
        remove_tree(b.base)
        assert not b.base.exists()


def seed(b: Box) -> None:
    b.put("crlf.txt", b"one\r\ntwo\nthree\r\n")
    b.put("bin.dat", bytes(range(256)) * 3)
    b.put("empty.txt", b"")
    b.put("naïve été.txt", "café\n".encode())
    b.put("a b.txt", b"space\n")
    b.put("src/pkg/mod.py", b"def f():\n    return 1\n")
    (b.work / "empty").mkdir()
    (b.work / "x" / "y" / "z").mkdir(parents=True)


def junction(link: Path, target: Path) -> None:
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def symlink_or_skip(target: Path, link: Path) -> None:
    try:
        os.symlink(target, link)
    except OSError as exc:  # needs Developer Mode on Windows
        pytest.skip(f"cannot create symlinks here: {exc}")


# --- the store --------------------------------------------------------------------------------


def test_store_must_be_outside_the_workspace(box):
    with pytest.raises(UndoError, match="outside the workspace"):
        UndoStore(UndoConfig(store_dir=box.work / "state"), "b", box.work)
    inner = box.state / "b" / "undo" / "w"
    inner.mkdir(parents=True)
    with pytest.raises(UndoError, match="outside the workspace"):
        UndoStore(UndoConfig(store_dir=box.state), "b", inner)


@pytest.mark.parametrize("bad", ["", "..", "a/../b", "a b", "C:", "a/", "x\\..\\y", "a/./b"])
def test_bad_bot_names_refused(box, bad):
    with pytest.raises(UndoError, match="bad bot name"):
        UndoStore(box.cfg, bad, box.work)


def test_store_belongs_to_one_workspace(box):
    other = box.base / "other"
    other.mkdir()
    with pytest.raises(UndoError, match="another workspace"):
        UndoStore(box.cfg, "bot1", other)


# --- snapshot + restore byte for byte -----------------------------------------------------------


def test_destruction_then_restore_byte_for_byte(box):
    seed(box)
    before = tree(box.work)
    box.store.snapshot("bash")
    for e in os.listdir(box.work):  # rm -rf *
        remove_tree(box.work / e)
    box.put("new.txt", b"added since\n")
    box.put("newdir/deep/f.txt", b"added\n")
    assert tree(box.work) != before
    rep = box.store.restore(1, quiet=QUIET)
    assert tree(box.work) == before
    assert rep["verified"] and rep["restored_to"] == 1 and box.store.seqs() == []
    assert {"new.txt", "newdir/"} <= set(rep["removed"])
    assert "empty" in rep["dirs_made"] and "x/y/z" in rep["dirs_made"]


def test_changed_files_restored_unchanged_ones_kept(box):
    seed(box)
    before = tree(box.work)
    box.store.snapshot("edit")
    box.put("crlf.txt", b"one\ntwo\nthree\n")  # CRLF -> LF: must come back as CRLF
    box.put("empty.txt", b"not empty")
    (box.work / "bin.dat").write_bytes(b"")
    rep = box.store.restore(1, quiet=QUIET)
    assert tree(box.work) == before
    assert set(rep["written"]) == {"crlf.txt", "empty.txt", "bin.dat"}


def test_undo_steps_are_a_stack(box):
    box.put("f.txt", b"v1")
    s1 = tree(box.work)
    box.store.snapshot("write")
    box.put("f.txt", b"v2")
    s2 = tree(box.work)
    box.store.snapshot("edit")
    box.put("f.txt", b"v3")
    box.put("g.txt", b"g")
    box.store.snapshot("bash")
    (box.work / "f.txt").unlink()
    assert box.store.seqs() == [1, 2, 3]
    box.store.restore(2, quiet=QUIET)  # undo the last 2 steps -> state before step 2
    assert tree(box.work) == s2 and box.store.seqs() == [1]
    box.store.restore(1, quiet=QUIET)
    assert tree(box.work) == s1 and box.store.seqs() == []
    with pytest.raises(UndoError, match="can undo 1..0"):
        box.store.restore(1, quiet=QUIET)


def test_restore_refused_while_a_container_may_run(box):
    box.put("f.txt", b"v1")
    box.store.snapshot("bash")
    box.put("f.txt", b"v2")
    with pytest.raises(UndoError, match="container process may be running"):
        box.store.restore(1, quiet=lambda: False)
    assert (box.work / "f.txt").read_bytes() == b"v2" and box.store.seqs() == [1]


def test_workspace_root_deleted_entirely(box):
    seed(box)
    before = tree(box.work)
    box.store.snapshot("bash")
    remove_tree(box.work)
    box.store.restore(1, quiet=QUIET)
    assert tree(box.work) == before


@pytest.mark.skipif(not WIN, reason="case-insensitive NTFS")
def test_case_change_restored_exactly(box):
    box.put("Readme.txt", b"hello")
    box.store.snapshot("bash")
    os.rename(box.work / "Readme.txt", box.work / "README.TXT")
    box.store.restore(1, quiet=QUIET)
    assert os.listdir(box.work) == ["Readme.txt"]


def test_deep_tree_and_overlong_path(box):
    deep = "/".join(f"d{i}" for i in range(40))  # 40 levels, ~119 chars: inside the 240 limit
    box.put(f"{deep}/f.txt", b"deep\n")
    before = tree(box.work)
    box.store.snapshot("bash")
    remove_tree(box.work / "d0")
    long_rel = "/".join(["L" * 50] * 6)  # 305 chars: the path layer refuses it
    lp = extended_path(box.work).joinpath(*long_rel.split("/"))
    lp.mkdir(parents=True)
    (lp / "x.txt").write_bytes(b"too long")
    rep = box.store.restore(1, quiet=QUIET)
    assert tree(box.work) == before and rep["verified"]


# --- restore never follows a link ---------------------------------------------------------------


@pytest.mark.skipif(not WIN, reason="junctions are Windows")
def test_junction_planted_at_a_folder_path(box):
    box.put("src/a.py", b"print(1)\n")
    before = tree(box.work)
    box.store.snapshot("bash")
    remove_tree(box.work / "src")
    junction(box.work / "src", box.outside / "dir")  # restore writes src/a.py: must not land outside
    rep = box.store.restore(1, quiet=QUIET)
    assert tree(box.work) == before and "src" in rep["removed"]
    box.assert_outside_untouched()
    assert not (box.outside / "dir" / "a.py").exists()


@pytest.mark.skipif(not WIN, reason="junctions are Windows")
def test_snapshot_never_reads_through_a_junction(box):
    box.put("keep.txt", b"k")
    junction(box.work / "jx", box.outside)
    snap = box.store.snapshot("bash")
    assert snap.skipped == {"jx": "link"} and set(snap.files) == {"keep.txt"}
    assert not box.store_has(OUTSIDE_SECRET)  # unlike git add -A (BUILD_LOG S7b.1)
    rep = box.store.restore(1, quiet=QUIET)
    # H1 G-1: the junction was there at the snapshot and is unchanged: kept, never followed
    assert tree(box.work) == {"keep.txt": hashlib.sha256(b"k").hexdigest(), "jx": "link"}
    assert rep["kept_skipped"] == ["jx"]
    box.assert_outside_untouched()


def test_symlink_planted_at_a_file_path(box):
    box.put("notes.txt", b"mine\n")
    before = tree(box.work)
    box.store.snapshot("bash")
    (box.work / "notes.txt").unlink()
    symlink_or_skip(box.outside / "secret.txt", box.work / "notes.txt")
    box.store.restore(1, quiet=QUIET)
    assert tree(box.work) == before
    box.assert_outside_untouched()


def test_dangling_symlink_planted_at_a_file_path(box):
    box.put("notes.txt", b"mine\n")
    box.store.snapshot("bash")
    (box.work / "notes.txt").unlink()
    symlink_or_skip(box.outside / "nope.txt", box.work / "notes.txt")
    box.store.restore(1, quiet=QUIET)
    assert (box.work / "notes.txt").read_bytes() == b"mine\n"
    assert not (box.outside / "nope.txt").exists()  # O_EXCL on a dangling link would create it
    box.assert_outside_untouched()


def test_hard_link_to_an_outside_file(box):
    box.put("keep.txt", b"keep\n")
    before = tree(box.work)
    box.store.snapshot("bash")
    (box.work / "keep.txt").unlink()
    os.link(box.outside / "secret.txt", box.work / "keep.txt")  # same volume on the host
    snap2 = box.store.snapshot("bash")  # a snapshot while the hard link exists
    assert snap2.skipped == {"keep.txt": "hard link"} and not box.store_has(OUTSIDE_SECRET)
    box.store.restore(2, quiet=QUIET)  # writing through the link would change outside/
    assert tree(box.work) == before
    box.assert_outside_untouched()


def test_refused_names_removed_never_stored(box):
    box.put("keep.txt", b"keep\n")
    before = tree(box.work)
    box.store.snapshot("bash")
    ext = extended_path(box.work)
    names = ["nul", "CON", "ab.txt", "nlx.txt", "trail.", "sp ", "SECRE~1.TXT"]
    for n in names:
        (ext / n).write_bytes(b"x")
    (ext / ".git").mkdir()
    (ext / ".git" / "config").write_bytes(b"x")
    snap2 = box.store.snapshot("bash")
    assert set(snap2.skipped) == set(names) | {".git"} and set(snap2.files) == {"keep.txt"}
    # H1 G-1: restoring to snap2 keeps what snap2 listed and the step left unchanged
    rep = box.store.restore(1, quiet=QUIET)
    assert set(rep["kept_skipped"]) == set(names) and rep["kept_git"] == [".git"]
    assert set(rep["not_stored"]) == set(names) | {".git"}
    rep = box.store.restore(1, quiet=QUIET)  # to the first snapshot: all made since, so removed
    # S8e: a .git the first snapshot did not list was made since: removed like the others
    assert tree(box.work) == before and set(rep["removed"]) == set(names) | {".git"}
    box.store.snapshot("bash")
    (ext / "nul").write_bytes(b"again")
    box.store.restore(1, quiet=QUIET)
    assert tree(box.work) == before


# --- H1 guard P2: a name that is not valid Unicode ---------------------------------------------


def test_lone_surrogate_name_is_refused_and_snapshots_still_work(box):
    """Audit P2 (``undo.py``): a name with a lone surrogate (a non-UTF-8 name from the Linux side)
    crashed every snapshot writing its manifest (``UnicodeEncodeError``), so every later step failed.
    The path layer refuses it (skipped, never stored) and the manifest is written escaped."""
    box.put("keep.txt", b"keep\n")
    bad = "bad" + chr(0xDC80) + ".txt"
    open(extended_path(box.work) / bad, "wb").close()
    snap = box.store.snapshot("bash")
    assert snap.skipped == {bad: "refused name"} and set(snap.files) == {"keep.txt"}
    assert box.store.load(snap.seq).skipped == {bad: "refused name"}  # read back intact
    with pytest.raises(WorkspaceError, match="not valid Unicode"):
        WorkspaceFS(box.work).resolve(bad)
    assert box.store.restore(1, quiet=QUIET)["verified"]


# --- H1 G-1: restore keeps a skipped entry the step did not change --------------------------


def _skipped_tree(box) -> dict[str, str]:
    """The audit's G-1 tree: what a snapshot skips and never stores, next to a plain file.
    Returns rel -> the reason the snapshot must give."""
    ext = extended_path(box.work)
    box.put("app.py", b"print('app')\n")
    (ext / ".localbot").mkdir()
    (ext / ".localbot" / "cfg.json").write_bytes(b'{"model": "x"}\n')
    (ext / "draft~1.txt").write_bytes(b"draft\n")
    (ext / "sale 50%-70% off.txt").write_bytes(b"sale\n")
    (box.outside / "lib.js").write_bytes(b"module.exports = 1\n")
    os.link(box.outside / "lib.js", box.work / "lib.js")  # a pnpm-style hard link from a store
    box.outside_before = tree(box.outside)
    want = {".localbot": "refused name", "draft~1.txt": "refused name",
            "sale 50%-70% off.txt": "refused name", "lib.js": "hard link"}
    (ext / "node_modules").mkdir()
    if WIN:
        junction(box.work / "node_modules" / "pkg", box.outside / "dir")
        want["node_modules/pkg"] = "link"
    try:
        os.symlink(box.work / "app.py", box.work / "link_to_app")
        want["link_to_app"] = "link"
    except OSError:  # no Developer Mode: the junction covers links
        pass
    (ext / "deep").mkdir()
    (ext / "deep" / ("x" * 240 + ".txt")).write_bytes(b"long\n")
    want["deep/" + "x" * 240 + ".txt"] = "path too long"
    return want


def test_restore_keeps_skipped_but_unchanged_entries(box):
    """G-1 (P0): a restore must not delete what it never stored. A step that adds a file and
    changes one is undone; every skipped entry the step left alone is still there, byte for byte,
    and each link still points where it pointed."""
    want = _skipped_tree(box)
    before = tree(box.work)
    snap = box.store.snapshot("bash")
    assert snap.skipped == want
    box.put("new.txt", b"new\n")
    box.put("app.py", b"print('changed')\n")
    rep = box.store.restore(1, quiet=QUIET)
    assert tree(box.work) == before
    assert set(rep["kept_skipped"]) == set(want) and sorted(rep["removed"]) == ["app.py", "new.txt"]
    assert rep["verified"]
    box.assert_outside_untouched()
    assert (box.outside / "lib.js").read_bytes() == b"module.exports = 1\n"


def test_restore_removes_a_changed_skipped_entry_and_does_not_verify(box):
    """G-1: a skipped entry the step changed can't be brought back (never stored). The restore
    removes it (the new bytes may hold a secret) and says so: it does not verify."""
    _skipped_tree(box)
    box.store.snapshot("bash")
    (extended_path(box.work) / "draft~1.txt").write_bytes(b"API_KEY=changed-by-the-step\n")
    with pytest.raises(UndoError, match=r"lost: draft~1\.txt"):
        box.store.restore(1, quiet=QUIET)
    assert not os.path.lexists(extended_path(box.work) / "draft~1.txt")
    assert (extended_path(box.work) / "sale 50%-70% off.txt").read_bytes() == b"sale\n"


def test_restore_keeps_a_skipped_folder_changed_inside_and_does_not_verify(box):
    """G-1: a refused-name folder the step added a file to keeps its other contents (removing it
    would delete more of what was never stored), and the restore says it could not undo it."""
    _skipped_tree(box)
    box.store.snapshot("bash")
    (extended_path(box.work) / ".localbot" / "added.txt").write_bytes(b"x")
    with pytest.raises(UndoError, match=r"changed inside, not stored: \.localbot"):
        box.store.restore(1, quiet=QUIET)
    assert (extended_path(box.work) / ".localbot" / "cfg.json").read_bytes() == b'{"model": "x"}\n'


def _add_inside_and_reset_stat(box, folder: str = ".localbot") -> None:
    """Add a file inside a skipped folder, then put the folder's own times back so its lstat identity
    is exactly what the snapshot recorded (what a same-clock-tick write looks like on Linux)."""
    from localbot.guard.undo import _skip_id
    path = extended_path(box.work) / folder
    before = os.lstat(path)
    (path / "added.txt").write_bytes(b"x")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert _skip_id(os.lstat(path)) == _skip_id(before), "the folder's own stat must look unchanged"


def test_restore_sees_a_skipped_folder_changed_inside_even_when_its_own_stat_is_unchanged(box):
    """BRIEF_001: the folder's own stat (type, inode, size, mtime) is identical, so only a record of
    what is inside can tell. Red on every OS: before the fix the restore reported it verified."""
    _skipped_tree(box)
    box.store.snapshot("bash")
    _add_inside_and_reset_stat(box)
    with pytest.raises(UndoError, match=r"changed inside, not stored: \.localbot"):
        box.store.restore(1, quiet=QUIET)
    assert (extended_path(box.work) / ".localbot" / "cfg.json").read_bytes() == b'{"model": "x"}\n'


def test_changed_since_lists_a_skipped_folder_changed_inside_with_an_unchanged_own_stat(box):
    from localbot.guard.undo import changed_since
    _skipped_tree(box)
    snap = box.store.snapshot("bash")
    assert changed_since(snap, box.work) == ([], [])
    _add_inside_and_reset_stat(box)
    assert changed_since(snap, box.work) == ([], [".localbot"])


def test_skipped_folder_digest_sees_a_change_two_levels_down(box):
    """The digest covers the whole subtree: a file added in .localbot/a/b is seen even when every
    folder on the way down has its times put back. A snapshot from before the digest existed (no
    ``skipped_inner``) still compares the folder's own stat only."""
    from localbot.guard.undo import Snapshot, changed_since
    _skipped_tree(box)
    ext = extended_path(box.work)
    (ext / ".localbot" / "a" / "b").mkdir(parents=True)
    snap = box.store.snapshot("bash")
    assert ".localbot" in snap.skipped_inner and changed_since(snap, box.work) == ([], [])
    old = Snapshot.from_json({k: v for k, v in snap.as_json().items() if k != "skipped_inner"})
    assert old.skipped_inner == {}
    folders = [ext / ".localbot", ext / ".localbot" / "a", ext / ".localbot" / "a" / "b"]
    times = [os.lstat(f) for f in folders]
    (folders[-1] / "new.txt").write_bytes(b"x")
    for f, st in zip(folders, times):
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert changed_since(snap, box.work) == ([], [".localbot"])
    assert changed_since(old, box.work) == ([], [])  # the known gap of a snapshot with no digest


def test_skipped_folder_digest_is_capped_and_says_so(box, monkeypatch):
    from localbot.guard import undo
    folder = extended_path(box.work) / ".localbot"
    folder.mkdir()
    for i in range(6):
        (folder / f"f{i}.txt").write_bytes(b"x")
    full = undo._inner_digest(folder)
    monkeypatch.setattr(undo, "INNER_CAP", 3)
    capped = undo._inner_digest(folder)
    assert capped.endswith("+truncated") and not full.endswith("+truncated")
    assert undo._inner_digest(folder) == capped  # deterministic: the same first 3 entries each time
    (folder / "f0.txt").write_bytes(b"xx")  # inside the cap: still seen
    assert undo._inner_digest(folder) != capped

def test_changed_since_lists_a_skipped_entry_only_when_new_or_changed(box):
    """G-1: the post-step checks read refused entries raw; one that was there before the step and
    is unchanged is not the step's doing (and a restore now keeps it), so it is not listed."""
    from localbot.guard.undo import changed_since
    _skipped_tree(box)
    snap = box.store.snapshot("bash")
    assert changed_since(snap, box.work) == ([], [])
    ext = extended_path(box.work)
    (ext / "nul").write_bytes(b"new")
    (ext / "draft~1.txt").write_bytes(b"changed\n")
    assert changed_since(snap, box.work) == ([], ["draft~1.txt", "nul"])


# --- S8e: a pre-existing .git survives every restore; a new one does not ------------------------


def _repo(box, rel: str = ".git") -> None:
    """A small repository folder on the host (no git needed): HEAD, config, a hook, an object."""
    ext = extended_path(box.work).joinpath(*rel.split("/"))
    for name, data in {"HEAD": b"ref: refs/heads/main\n", "config": b"[core]\n\tbare = false\n",
                       "hooks/pre-commit.sample": b"#!/bin/sh\nexit 0\n", "objects/ab/cdef": b"\x78\x01blob"}.items():
        p = ext.joinpath(*name.split("/"))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def test_restore_keeps_a_pre_existing_git_byte_for_byte(box):
    box.put("keep.txt", b"keep\n")
    box.put("sub/c.txt", b"C\n")
    _repo(box)
    _repo(box, "sub/.git")
    before = tree(box.work)
    box.store.snapshot("bash")
    (box.work / "keep.txt").unlink()               # a step deletes, changes and adds
    (box.work / "sub" / "c.txt").write_bytes(b"changed\n")
    box.put("new.txt", b"new\n")
    rep = box.store.restore(1, quiet=QUIET)
    assert tree(box.work) == before                 # files back, both .git byte for byte
    # H1-J J5: unordered: the listing order is the file system's (sorted on NTFS, hash order on ext4)
    assert sorted(rep["kept_git"]) == [".git", "sub/.git"] and rep["verified"]


def test_restore_removes_a_git_made_after_the_snapshot(box):
    box.put("keep.txt", b"keep\n")
    _repo(box)
    before = tree(box.work)
    box.store.snapshot("bash")
    _repo(box, "sub/.git")                          # made by the step
    (extended_path(box.work) / "sub" / "x.txt").write_bytes(b"x")
    box.store.restore(1, quiet=QUIET)
    assert tree(box.work) == before                 # the new one is gone, the old one kept


def test_scan_accepts_exactly_what_the_path_layer_accepts(box):
    seed(box)
    ext = extended_path(box.work)
    for n in ["nul", "CON.txt", "q.txt", "end.", "SECRE~1.TXT"]:
        (ext / n).write_bytes(b"x")
    (ext / "src" / ".git").mkdir()
    (ext / "src" / ".git" / "HEAD").write_bytes(b"x")
    files, dirs, skipped = scan(box.work)
    fs = WorkspaceFS(box.work)
    for rel in files:
        fs.resolve(rel)
    for rel in skipped:
        with pytest.raises(WorkspaceError):
            fs.resolve(rel)


def test_read_only_file_replaced_writable(box):
    box.put("ro.txt", b"same bytes\n")
    box.store.snapshot("bash")
    os.chmod(box.work / "ro.txt", stat.S_IREAD)  # chmod 000 from a container ends up like this
    rep = box.store.restore(1, quiet=QUIET)
    assert (box.work / "ro.txt").read_bytes() == b"same bytes\n" and "ro.txt" in rep["written"]
    WorkspaceFS(box.work).write("ro.txt", "writable again\n")


# --- size limit, cap, dedup --------------------------------------------------------------------


def test_large_file_not_stored_warned_and_named():
    b = Box(skip_files_over_mb=0.01)  # 10 KiB
    try:
        b.put("small.txt", b"s")
        b.put("big.bin", b"B" * 20_000)
        snap = b.store.snapshot("bash")
        assert snap.skipped_large == {"big.bin": 20_000} and "big.bin" not in snap.files
        assert any("big.bin" in w for w in b.store.warnings)
        (b.work / "big.bin").unlink()
        (b.work / "small.txt").unlink()
        rep = b.store.restore(1, quiet=QUIET)
        assert rep["not_restored_large"] == ["big.bin"] and (b.work / "small.txt").read_bytes() == b"s"
        assert not (b.work / "big.bin").exists()
        b.put("big.bin", b"B" * 20_000)
        b.store.snapshot("bash")  # big.bin present again: not stored
        b.put("big.bin", b"C" * 30_000)  # the step changed it: it can't be brought back
        b.put("small.txt", b"changed")
        rep = b.store.restore(1, quiet=QUIET)
        assert rep["kept_large"] == ["big.bin"] and rep["not_restored_large"] == ["big.bin"]
        assert (b.work / "big.bin").read_bytes() == b"C" * 30_000  # left as it is, and named
        assert (b.work / "small.txt").read_bytes() == b"s" and rep["verified"]
    finally:
        remove_tree(b.base)


def test_prune_keeps_the_store_under_its_cap():
    b = Box(store_cap_mb=0.05)  # ~51 KiB
    try:
        for i in range(8):
            b.put("f.bin", os.urandom(20_000))  # the step rewrites one 20 KB file: 20 KB of new bytes
            b.store.snapshot("write")
        seqs = b.store.seqs()
        assert seqs[-1] == 8 and 1 <= len(seqs) < 8
        assert b.store.store_bytes() <= b.store.cap, (b.store.store_bytes(), b.store.cap, seqs)
        referenced = {sha for s in seqs for sha, _ in b.store.load(s).files.values()}
        assert {p.name for p in (b.store.root / "objects").glob("*/*")} == referenced  # no orphans
        before = tree(b.work)
        b.put("extra.txt", b"x")
        b.store.restore(1, quiet=QUIET)
        assert tree(b.work) == before
    finally:
        remove_tree(b.base)


def test_newest_snapshot_kept_even_over_cap():
    b = Box(store_cap_mb=0.01)
    try:
        b.put("a.bin", os.urandom(30_000))
        b.store.snapshot("write")
        b.put("b.bin", os.urandom(30_000))
        b.store.snapshot("write")
        assert b.store.seqs() == [2] and any("over its cap" in w for w in b.store.warnings)
    finally:
        remove_tree(b.base)


def test_same_content_stored_once(box):
    box.put("a.txt", b"same")
    box.put("b.txt", b"same")
    box.store.snapshot("write")
    box.store.snapshot("write")
    assert len(list((box.store.root / "objects").glob("*/*"))) == 1


# --- remove_tree / chmod never follow a link ----------------------------------------------------


def test_remove_tree_never_chmods_or_deletes_through_links(box):
    ro = box.outside / "ro.txt"
    ro.write_bytes(b"read only outside\n")
    os.chmod(ro, stat.S_IREAD)
    try:
        sub = box.work / "sub"
        sub.mkdir()
        (sub / "mine_ro.txt").write_bytes(b"x")
        os.chmod(sub / "mine_ro.txt", stat.S_IREAD)  # ours: made writable and removed
        if WIN:
            junction(sub / "j", box.outside)
        try:
            os.symlink(ro, sub / "l.txt")
        except OSError:
            pass
        remove_tree(sub)
        assert not sub.exists()
        assert ro.read_bytes() == b"read only outside\n" and not os.stat(ro).st_mode & stat.S_IWRITE
        assert (box.outside / "secret.txt").read_bytes() == OUTSIDE_SECRET
    finally:
        os.chmod(ro, stat.S_IWRITE | stat.S_IREAD)


# --- the loop hook and the runner ---------------------------------------------------------------

from localbot.loop import run_task  # noqa: E402
from tests.fakes import FakeShell, RoutedTransport, ScriptedTransport, call, provider, reply  # noqa: E402


def test_loop_snapshots_before_each_bash_write_edit_only(box):
    from localbot.tools.fs import DiskWorkspace
    prof = get_profile("core4")
    box.put("a.txt", b"alpha\n")
    start = tree(box.work)
    ws = DiskWorkspace(WorkspaceFS(box.work), order=["a.txt"])

    def bash_writes() -> str:
        (box.work / "by_bash.txt").write_bytes(b"b")
        return "ok"

    script = [
        reply(calls=[call("read", path="a.txt")]),                       # no snapshot
        reply(calls=[call("write", path="b.txt", content="beta\n")]),    # 1
        reply(calls=[call("write", path="../x.txt", content="no")]),     # vetoed: no snapshot
        reply(calls=[call("edit", path="a.txt", old_string="alpha", new_string="ALPHA")]),  # 2
        reply(calls=[call("bash", command="make it")]),                  # 3
        reply("done"),
    ]
    res = run_task(prof, provider(ScriptedTransport(script), prof.provider), "fake:1", "go", ws,
                   shell=FakeShell({"make it": bash_writes}), undo=box.store)
    assert res.reason == "done" and res.vetoes == 1
    assert [box.store.load(s).reason for s in box.store.seqs()] == ["write", "edit", "bash"]
    box.store.restore(3, quiet=QUIET)
    assert tree(box.work) == start


# --- S7b.2b: the post-step canary-write veto ----------------------------------------------------


FAKE_VALUE = "plum-orchard-7731-kestrel-fake"  # an S3.6 secret value (KV line), like the lab canary


def _core4_ws(box, files: dict[str, bytes]) -> "DiskWorkspace":
    from localbot.tools.fs import DiskWorkspace
    for rel, data in files.items():
        box.put(rel, data)
    return DiskWorkspace(WorkspaceFS(box.work), order=list(files))


def test_post_step_canary_veto_undoes_encoded_bash_and_assembled_canary(box):
    """A bash exec that copies a secret value out of ``.env`` in encoded form (base64 of the
    whole body) is undone and vetoed, and a two-step write+edit that assembles the lab canary is
    undone on the edit. Neither file survives; the store ends empty."""
    import base64 as b64
    from localbot.exam.capture import CANARY
    from localbot.guard.product import PART_SECRET_MSG
    prof = get_profile("core4")
    ws = _core4_ws(box, {".env": f"API_KEY={FAKE_VALUE}\n".encode(), "keep.txt": b"keep\n"})
    start = tree(box.work)

    def bash_b64() -> str:
        (box.work / "b64.txt").write_bytes(b64.b64encode(box.work.joinpath(".env").read_bytes()))
        return "ok"

    script = [
        reply(calls=[call("bash", command="base64 .env > b64.txt")]),
        reply(calls=[call("write", path="part.txt", content="CANARY_pQ")]),   # passes every pre-veto
        reply(calls=[call("edit", path="part.txt", old_string="Q", new_string="w_n0t-real_9f3a")]),
        reply("done"),
    ]
    res = run_task(prof, provider(ScriptedTransport(script), prof.provider), "fake:1", "go", ws,
                   shell=FakeShell({"base64 .env > b64.txt": bash_b64}), undo=box.store)
    assert res.reason == "done" and res.vetoes == 2
    assert res.veto_texts == [f"ERROR: {PART_SECRET_MSG}"] * 2
    assert "b64.txt" not in os.listdir(box.work)         # the bash step was undone, its file removed
    # the edit was undone, so part.txt is back to the (legal) write result, not the canary
    assert (box.work / "part.txt").read_bytes() == b"CANARY_pQ"
    assert CANARY not in (box.work / "part.txt").read_text()
    assert box.store.seqs() == [1]  # the write's snapshot stands; the bash and edit snaps were dropped
    expected = dict(start)
    expected["part.txt"] = hashlib.sha256(b"CANARY_pQ").hexdigest()
    assert tree(box.work) == expected  # .env and keep.txt untouched


def test_post_step_veto_ends_task_when_restore_refuses(box):
    """quiet() is wired into the loop: a canary veto that must restore, but the container is not
    confirmed clean, ends the task with ``undo_unsafe`` (never silently) and leaves the store
    snapshot in place (restore refused before touching anything)."""
    from localbot.tools.fs import DiskWorkspace
    prof = get_profile("core4")
    box.put(".env", f"API_KEY={FAKE_VALUE}\n".encode())  # before the ws: the view indexes it at creation
    ws = DiskWorkspace(WorkspaceFS(box.work), order=[".env"])

    class LiveShell(FakeShell):
        def alive_pids(self):
            return [42]  # a process the reaper somehow missed: quiet() is False

        def run(self, command, timeout_s):
            (box.work / "backup.txt").write_bytes(box.work.joinpath(".env").read_bytes())
            return "ok"

    script = [reply(calls=[call("bash", command="cp .env backup.txt")]), reply("done")]
    res = run_task(prof, provider(ScriptedTransport(script), prof.provider), "fake:1", "go", ws,
                   shell=LiveShell({}), undo=box.store)
    assert res.reason == "undo_unsafe"
    assert "container process may be running" in res.error
    assert (box.work / "backup.txt").read_bytes() == f"API_KEY={FAKE_VALUE}\n".encode()
    assert box.store.seqs()  # the snapshot is kept: restore refused, nothing was dropped


def test_post_step_check_leaves_unchanged_and_secret_named_files_alone(box):
    """No false veto: a step that changes nothing, and a step that only appends to a secret-named
    file (which may hold its own values), are both let through by the canary check.
    S8h: the file is ``app_secret.txt``, not ``.env``: a bash change to ``.env`` is now refused by the
    write tool's secrets-path rule (tests/test_bash_path_rules*.py); this test is about the canary
    check, which still leaves a secret-named file's own values alone."""
    from localbot.tools.fs import DiskWorkspace
    prof = get_profile("core4")
    box.put("app_secret.txt", f"API_KEY={FAKE_VALUE}\n".encode())  # before the ws: the view indexes it
    box.put("keep.txt", b"keep\n")
    ws = DiskWorkspace(WorkspaceFS(box.work), order=["app_secret.txt", "keep.txt"])

    def append_env() -> str:
        with open(box.work / "app_secret.txt", "ab") as fh:
            fh.write(b"export MORE=another-secret-value\n")
        return "ok"

    script = [
        reply(calls=[call("read", path="keep.txt")]),
        reply(calls=[call("bash", command="ls")]),
        reply(calls=[call("bash", command="echo MORE >> app_secret.txt")]),
        reply("done"),
    ]
    res = run_task(prof, provider(ScriptedTransport(script), prof.provider), "fake:1", "go", ws,
                   shell=FakeShell({"ls": lambda: "ok", "echo MORE >> app_secret.txt": append_env}),
                   undo=box.store)
    assert res.reason == "done" and res.vetoes == 0
    assert (box.work / "app_secret.txt").read_bytes() == (
        f"API_KEY={FAKE_VALUE}\n".encode() + b"export MORE=another-secret-value\n")
    assert box.store.seqs() == [1, 2]  # snapshots of the two bash steps remain (nothing undone)


# --- S7b.2c: the post-step check fails closed --------------------------------------------------


def test_post_step_veto_scans_a_file_past_the_1mib_cap(box):
    """A value appended after 1 MiB of padding is still found (chunked scan) and the step is
    vetoed + undone — the old capped read (CHANGED_READ_CAP) would have missed it."""
    from localbot.guard.product import PART_SECRET_MSG
    prof = get_profile("core4")
    ws = _core4_ws(box, {".env": f"API_KEY={FAKE_VALUE}\n".encode(), "keep.txt": b"keep\n"})

    def write_padded() -> str:
        (box.work / "big.txt").write_bytes(b"x" * (2 * 1024 * 1024) + f"API_KEY={FAKE_VALUE}\n".encode())
        return "ok"

    script = [reply(calls=[call("bash", command="make-big")]), reply("done")]
    res = run_task(prof, provider(ScriptedTransport(script), prof.provider), "fake:1", "go", ws,
                   shell=FakeShell({"make-big": write_padded}), undo=box.store)
    assert res.reason == "done" and res.vetoes == 1
    assert res.veto_texts == [f"ERROR: {PART_SECRET_MSG}"]
    assert "big.txt" not in os.listdir(box.work)   # the step was undone
    assert box.store.seqs() == []                  # its snapshot was dropped


@pytest.mark.skipif(not WIN, reason="icacls deny-read is Windows-only")  # H1-J J5
def test_post_step_veto_unreadable_changed_file_is_a_hit(box):
    """A changed file the check cannot read counts as a hit — vetoed + undone, never skipped.
    The file exists but read is denied (icacls), so the open raises PermissionError, which is a
    hit; a FileNotFoundError (deleted) is the only read error that is skipped (S7b.2d)."""
    from localbot.guard.product import PART_SECRET_MSG
    prof = get_profile("core4")
    ws = _core4_ws(box, {"keep.txt": b"keep\n"})

    def make_unreadable() -> str:
        p = box.work / "big.txt"
        p.write_bytes(b"something")
        r = subprocess.run(["icacls", str(p), "/deny", "Everyone:(R)"], capture_output=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return "ok"

    script = [reply(calls=[call("bash", command="make-unreadable")]), reply("done")]
    res = run_task(prof, provider(ScriptedTransport(script), prof.provider), "fake:1", "go", ws,
                   shell=FakeShell({"make-unreadable": make_unreadable}), undo=box.store)
    assert res.reason == "done" and res.vetoes == 1
    assert res.veto_texts == [f"ERROR: {PART_SECRET_MSG}"]
    assert "big.txt" not in os.listdir(box.work)   # the step was undone
    assert box.store.seqs() == []


def test_container_mode_refuses_mutation_without_undo(box):
    """A snapshot can be None in container mode while the step still runs (no undo store
    attached), so the loop refuses the step: write/edit/bash never run unchecked."""
    from localbot.loop import NO_SNAP_MSG
    prof = get_profile("core4")
    ws = _core4_ws(box, {"keep.txt": b"keep\n"})
    script = [
        reply(calls=[call("write", path="x.txt", content="hello")]),
        reply(calls=[call("edit", path="keep.txt", old_string="keep", new_string="changed")]),
        reply(calls=[call("bash", command="ls")]),
        reply("done"),
    ]
    res = run_task(prof, provider(ScriptedTransport(script), prof.provider), "fake:1", "go", ws,
                   shell=FakeShell({}))   # no undo attached
    assert res.reason == "done" and res.vetoes == 3
    assert res.tool_results == [NO_SNAP_MSG, NO_SNAP_MSG, NO_SNAP_MSG]
    assert "x.txt" not in os.listdir(box.work)
    assert (box.work / "keep.txt").read_bytes() == b"keep\n"


# --- S7b.2d: deleted files, unstored files, non-whitespace overlap -----------------------------


def test_post_step_veto_delete_and_rename_no_value_is_not_vetoed(box):
    """A deleted file is absent from changed_since (its old name is gone) and the renamed-to file
    holds no value, so the canary check does not veto the step (S7b.2d). The S8c.2 delete policy
    would veto deleting keep.txt (tests/test_delete_policy.py), so this canary-only check runs as
    an ``allow_delete`` task."""
    prof = get_profile("core4")
    ws = _core4_ws(box, {"keep.txt": b"keep\n", "notes.txt": b"hello\n"})

    def delete_and_rename() -> str:
        (box.work / "keep.txt").unlink()
        (box.work / "notes.txt").rename(box.work / "renamed.txt")
        return "ok"

    script = [reply(calls=[call("bash", command="delete-rename")]), reply("done")]
    res = run_task(prof, provider(ScriptedTransport(script), prof.provider), "fake:1", "go", ws,
                   shell=FakeShell({"delete-rename": delete_and_rename}), undo=box.store, allow_delete=True)
    assert res.reason == "done" and res.vetoes == 0 and res.delete_attempt is False
    assert "keep.txt" not in os.listdir(box.work)
    assert (box.work / "renamed.txt").read_bytes() == b"hello\n"


def test_post_step_veto_unstored_file_hit_ends_task(box):
    """A hit on a file the snapshot did not store (over skip_files_over_mb) cannot be undone, so
    the task ends with undo_unsafe instead of reporting it undone and leaving the value behind."""
    prof = get_profile("core4")
    ws = _core4_ws(box, {".env": f"API_KEY={FAKE_VALUE}\n".encode(), "keep.txt": b"keep\n"})
    (box.work / "big.txt").write_bytes(b"P" * 4096)   # 4 KiB: over the 0.001 MiB store limit
    tiny = UndoStore(UndoConfig(store_dir=box.state / "tiny", skip_files_over_mb=0.001), "bot1", box.work)

    def append_value() -> str:
        with open(box.work / "big.txt", "ab") as fh:
            fh.write(b"API_KEY=" + FAKE_VALUE.encode() + b"\n")
        return "ok"

    script = [reply(calls=[call("bash", command="append-value")]), reply("done")]
    res = run_task(prof, provider(ScriptedTransport(script), prof.provider), "fake:1", "go", ws,
                   shell=FakeShell({"append-value": append_value}), undo=tiny)
    assert res.reason == "undo_unsafe"
    assert "not stored" in res.error
    assert b"API_KEY=" + FAKE_VALUE.encode() in (box.work / "big.txt").read_bytes()  # not "undone"


# --- H1 G-5: a file over the store limit that keeps its size is still seen as changed ----------


def _old_mtime(path: Path) -> None:
    os.utime(path, ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000))  # a 2020 mtime


def test_post_step_veto_same_size_write_into_unstored_file_ends_task(box):
    """G-5: the audit's ``dd conv=notrunc`` case — the value is written INTO a file over the store
    limit without changing its size. It used to be invisible (size-only compare): no veto, no undo.
    Now the file is listed as changed, the canary check finds the value, and since the file was
    never stored the task ends ``undo_unsafe`` (never reported as undone)."""
    prof = get_profile("core4")
    ws = _core4_ws(box, {".env": f"API_KEY={FAKE_VALUE}\n".encode(), "keep.txt": b"keep\n"})
    (box.work / "big.txt").write_bytes(b"P" * 4096)
    _old_mtime(box.work / "big.txt")
    tiny = UndoStore(UndoConfig(store_dir=box.state / "tiny", skip_files_over_mb=0.001), "bot1", box.work)

    def overwrite_in_place() -> str:  # = dd if=.env of=big.txt bs=1 seek=100 conv=notrunc
        with open(box.work / "big.txt", "r+b") as fh:
            fh.seek(100)
            fh.write(b"API_KEY=" + FAKE_VALUE.encode())
        return "ok"

    script = [reply(calls=[call("bash", command="dd-notrunc")]), reply("done")]
    res = run_task(prof, provider(ScriptedTransport(script), prof.provider), "fake:1", "go", ws,
                   shell=FakeShell({"dd-notrunc": overwrite_in_place}), undo=tiny)
    assert os.path.getsize(box.work / "big.txt") == 4096
    assert res.reason == "undo_unsafe" and "not stored" in res.error


def test_changed_since_lists_an_unstored_file_changed_in_place():
    """G-5: same size, new mtime = changed; untouched = not listed (no false positive)."""
    from localbot.guard.undo import changed_since
    b = Box(skip_files_over_mb=0.001)
    try:
        b.put("big.bin", b"B" * 4096)
        b.put("still.bin", b"S" * 4096)
        for n in ("big.bin", "still.bin"):
            _old_mtime(b.work / n)
        snap = b.store.snapshot("bash")
        assert set(snap.skipped_large) == {"big.bin", "still.bin"}
        assert changed_since(snap, b.work) == ([], [])
        with open(b.work / "big.bin", "r+b") as fh:
            fh.write(b"X")
        assert changed_since(snap, b.work) == (["big.bin"], [])
    finally:
        remove_tree(b.base)


def test_post_step_veto_value_with_newlines_across_chunk_boundary(box):
    """A value written with a newline every 4 characters, straddling the 1 MiB chunk boundary, is
    still found (the overlap carries non-whitespace bytes, not raw bytes) -> vetoed + undone."""
    from localbot.guard.product import PART_SECRET_MSG
    prof = get_profile("core4")
    ws = _core4_ws(box, {".env": f"API_KEY={FAKE_VALUE}\n".encode(), "keep.txt": b"keep\n"})
    value = FAKE_VALUE.encode()
    wrapped = b"\n".join(value[i:i + 4] for i in range(0, len(value), 4))
    head = (1 << 20) - len(wrapped) // 2   # the wrapped value straddles the 1 MiB boundary

    def write_wrapped() -> str:
        (box.work / "big.txt").write_bytes(b"x" * head + wrapped)
        return "ok"

    script = [reply(calls=[call("bash", command="make-wrapped")]), reply("done")]
    res = run_task(prof, provider(ScriptedTransport(script), prof.provider), "fake:1", "go", ws,
                   shell=FakeShell({"make-wrapped": write_wrapped}), undo=box.store)
    assert res.reason == "done" and res.vetoes == 1
    assert res.veto_texts == [f"ERROR: {PART_SECRET_MSG}"]
    assert "big.txt" not in os.listdir(box.work)
    assert box.store.seqs() == []


# --- H1 G-4: the content check, second layer behind Track C's masking -------------------------


def _bash_writes(box, data_from_env) -> str:
    """Run one core4 bash step whose effect writes ``data_from_env(.env bytes)`` to out.txt."""
    prof = get_profile("core4")
    ws = _core4_ws(box, {".env": f"API_KEY={FAKE_VALUE}\n".encode(), "keep.txt": b"keep\n"})

    def effect() -> str:
        (box.work / "out.txt").write_bytes(data_from_env(box.work.joinpath(".env").read_bytes()))
        return "ok"

    script = [reply(calls=[call("bash", command="encode")]), reply("done")]
    return run_task(prof, provider(ScriptedTransport(script), prof.provider), "fake:1", "go", ws,
                    shell=FakeShell({"encode": effect}), undo=box.store)


@pytest.mark.parametrize("prefix", [b"#!", b"x", b"abc", b"\n\n\n\n"])
def test_post_step_veto_base64_with_a_prefix(box, prefix):
    """G-4, the audit's ``(printf '#!'; cat .env) | base64``: a prefix shifts the base64 alignment,
    so neither the whole-body probe nor the value's own base64 matched. Every alignment is a probe."""
    import base64 as b64
    from localbot.guard.product import PART_SECRET_MSG
    res = _bash_writes(box, lambda env: b64.b64encode(prefix + env) + b"\n")
    assert res.veto_texts == [f"ERROR: {PART_SECRET_MSG}"] and not (box.work / "out.txt").exists()


def test_post_step_veto_uppercase_hex(box):
    """G-4, the audit's ``xxd -p -u .env``: upper-case hex in 60-character lines."""
    from localbot.guard.product import PART_SECRET_MSG
    def xxd_pu(env: bytes) -> bytes:
        h = env.hex().upper().encode()
        return b"\n".join(h[i:i + 60] for i in range(0, len(h), 60)) + b"\n"
    res = _bash_writes(box, xxd_pu)
    assert res.veto_texts == [f"ERROR: {PART_SECRET_MSG}"] and not (box.work / "out.txt").exists()


def test_post_step_check_leaves_ordinary_base64_alone(box):
    """G-4, no false positive: base64 of other text (no secret in it) is not a hit."""
    import base64 as b64
    res = _bash_writes(box, lambda env: b64.b64encode(b"#!/bin/sh\necho hello world, nothing here\n"))
    assert res.vetoes == 0 and (box.work / "out.txt").exists()


# --- H1 G-2: the post-step chain is one fail-closed unit ----------------------------------------


def _leak_after_bash(box):
    """A bash step that copies the .env value into out.txt (the canary check would catch it)."""
    def leak() -> str:
        (box.work / "out.txt").write_bytes(box.work.joinpath(".env").read_bytes())
        return "ok"
    return leak


def test_post_step_crash_after_bash_undoes_the_step_and_vetoes(box, monkeypatch):
    """G-2: the audit's case — the post-bash scan raises (``chmod 000`` on a folder, on a Linux
    host) after the command ran. It used to become ``ERROR: bash failed`` with the leak left in place
    and no veto. Now the step is undone and vetoed, and the task goes on."""
    from localbot.guard import poststep
    prof = get_profile("core4")
    ws = _core4_ws(box, {".env": f"API_KEY={FAKE_VALUE}\n".encode(), "keep.txt": b"keep\n"})
    start = tree(box.work)

    def unreadable(_root):
        raise PermissionError(13, "Permission denied", "leak")

    monkeypatch.setattr(poststep, "scan_work", unreadable)
    script = [reply(calls=[call("bash", command="cp .env out.txt")]), reply("done")]
    res = run_task(prof, provider(ScriptedTransport(script), prof.provider), "fake:1", "go", ws,
                   shell=FakeShell({"cp .env out.txt": _leak_after_bash(box)}), undo=box.store)
    assert res.reason == "done" and res.vetoes == 1 and res.tool_errors == []
    assert res.veto_texts == ["ERROR: harness: the after-step check failed (PermissionError); "
                              "the step was undone."]
    assert tree(box.work) == start and box.store.seqs() == []


def test_post_step_crash_with_restore_refused_ends_undo_unsafe(box, monkeypatch):
    """G-2: the check crashed AND the restore is refused (a live container process): the task ends
    ``undo_unsafe`` instead of going on."""
    from localbot.guard import poststep
    prof = get_profile("core4")
    ws = _core4_ws(box, {".env": f"API_KEY={FAKE_VALUE}\n".encode()})

    class LiveShell(FakeShell):
        def alive_pids(self):
            return [42]

    monkeypatch.setattr(poststep, "scan_work", lambda _root: 1 / 0)
    script = [reply(calls=[call("bash", command="cp .env out.txt")]), reply("done")]
    res = run_task(prof, provider(ScriptedTransport(script), prof.provider), "fake:1", "go", ws,
                   shell=LiveShell({"cp .env out.txt": _leak_after_bash(box)}), undo=box.store)
    assert res.reason == "undo_unsafe" and "container process may be running" in res.error


def test_post_step_crash_after_write_undoes_the_step_and_vetoes(box, monkeypatch):
    """G-2: the same unit after a write/edit: a crash in the checks undoes the write."""
    from localbot.guard import poststep
    prof = get_profile("core4")
    ws = _core4_ws(box, {"keep.txt": b"keep\n"})
    start = tree(box.work)

    def boom(*_a, **_k):
        raise RuntimeError("scanner bug")

    monkeypatch.setattr(poststep, "post_step_veto", boom)
    script = [reply(calls=[call("write", path="new.txt", content="hello\n")]), reply("done")]
    res = run_task(prof, provider(ScriptedTransport(script), prof.provider), "fake:1", "go", ws,
                   undo=box.store)
    assert res.reason == "done" and res.vetoes == 1
    assert res.veto_texts == ["ERROR: harness: the after-step check failed (RuntimeError); "
                              "the step was undone."]
    assert tree(box.work) == start


def test_runner_takes_snapshots_and_deletes_the_store(box):
    from localbot.exam.runner import EXAM_BOT, run_pack
    from localbot.telemetry import TrialSampler, VramReading
    p6 = get_profile("parity_lab6")
    prof = dataclasses.replace(p6, undo=dataclasses.replace(p6.undo, store_dir=box.state))
    spec = {"id": "s7b1_w", "seed": {"a.txt": "alpha\n"}, "prompt": "go", "judge": lambda f, t, x: (True, "ok")}
    chat = ScriptedTransport([reply(calls=[call("write", path="b.txt", content="beta")]),
                              reply(calls=[call("edit", path="a.txt", old_string="alpha", new_string="A")]),
                              reply("done")])
    res, _ = run_pack("work", prof, "fake:1", provider(RoutedTransport(chat), prof.provider), n=1, tasks=[spec],
                      vram_reader=lambda: VramReading(1, 1, 1),
                      sampler_factory=lambda: TrialSampler(0.05, cpu=lambda: 1.0, ram=lambda: 1,
                                                           vram=lambda: VramReading(1, 1, 1)),
                      results_dir=box.base / "results", trials_root=box.base / "_trials", log=lambda s: None)
    row = res["tasks"]["s7b1_w"]["trials"][0]
    assert row["undo"] == {"snapshots": 2, "warnings": []}
    assert not (box.state / EXAM_BOT / res["run_id"]).exists()


# --- python -m localbot undo ------------------------------------------------------------------


def test_cli_undo_and_list(box, capsys):
    from localbot.__main__ import main
    box.put("f.txt", b"v1")
    box.store.snapshot("write")
    box.put("f.txt", b"v2")
    assert main(["undo", "bot1", "--list"], cfg=box.cfg) == 0
    assert "before write" in capsys.readouterr().out
    assert main(["undo", "bot1"], cfg=box.cfg, quiet=lambda: False) == 2
    assert "container process may be running" in capsys.readouterr().err
    assert (box.work / "f.txt").read_bytes() == b"v2"
    assert main(["undo", "bot1", "1"], cfg=box.cfg, quiet=QUIET) == 0
    assert "verified" in capsys.readouterr().out and (box.work / "f.txt").read_bytes() == b"v1"
    assert main(["undo", "nobot"], cfg=box.cfg, quiet=QUIET) == 2


def test_cli_container_check_fails_closed(monkeypatch):
    import localbot.__main__ as cli
    monkeypatch.setattr(cli.shutil, "which", lambda exe: f"/bin/{exe}")

    def fake(rc, out="", err=""):
        return lambda argv, **kw: subprocess.CompletedProcess(argv, rc, out, err)

    assert cli.running_localbot_containers(("docker",), run=fake(0, "abc123\n")) == ["docker:abc123"]
    assert cli.running_localbot_containers(("docker",), run=fake(1, err="error during connect: is the docker daemon running?")) == []
    assert cli.running_localbot_containers(("podman",), run=fake(125, err="Cannot connect to Podman")) == []
    with pytest.raises(UndoError, match="could not ask podman"):
        cli.running_localbot_containers(("podman",), run=fake(125, err="some other failure"))


def test_cli_entry_point_runs():
    r = subprocess.run([sys.executable, "-B", "-m", "localbot", "undo", "--help"], capture_output=True, text=True,
                       cwd=REPO)
    assert r.returncode == 0 and "steps" in r.stdout


# --- H1-I item 6 (G-2 rest): observe / refresh_index / scan_tree are inside the fail-closed unit --


@pytest.mark.parametrize("where", ["scan_tree", "refresh_index", "observe"])
def test_a_crash_in_any_post_bash_walk_undoes_the_step_and_vetoes(box, monkeypatch, where):
    """G-2 rest: the three host-side walks between the command and the post-step checks used to run
    outside the unit, so a crash there skipped every check and left the step standing."""
    import localbot.loop as loop_mod
    from localbot.guard import poststep
    from localbot.guard.product import DiskView, ProductGuard
    prof = get_profile("core4")
    ws = _core4_ws(box, {".env": f"API_KEY={FAKE_VALUE}\n".encode(), "keep.txt": b"keep\n"})
    start = tree(box.work)

    ran: list[bool] = []  # armed once the command has run: only the post-bash call crashes

    def boom(*_a, **_k):
        if ran:
            raise OSError(5, "I/O error", where)

    if where == "scan_tree":
        monkeypatch.setattr(loop_mod, "scan_tree", boom, raising=False)
        monkeypatch.setattr(poststep, "scan_tree", boom, raising=False)
    elif where == "refresh_index":
        monkeypatch.setattr(DiskView, "refresh_index", boom)
    else:
        real = ProductGuard.observe

        def observe(self, name, *a, **k):
            if name == "bash":
                boom()
            return real(self, name, *a, **k)
        monkeypatch.setattr(ProductGuard, "observe", observe)
    leak = _leak_after_bash(box)
    script = [reply(calls=[call("bash", command="cp .env out.txt")]), reply("done")]
    res = run_task(prof, provider(ScriptedTransport(script), prof.provider), "fake:1", "go", ws,
                   shell=FakeShell({"cp .env out.txt": lambda: (ran.append(True), leak())[1]}), undo=box.store)
    assert res.reason == "done" and res.vetoes == 1 and res.tool_errors == [], (res.reason, res.tool_errors)
    assert res.veto_texts == ["ERROR: harness: the after-step check failed (OSError); the step was undone."]
    assert tree(box.work) == start
