"""Protect the workspace's own ``.git`` from a bash step (docs/STAGE1_SPEC.md S8e.1).

The container masks each task-start ``.git`` read-only and empty (``shell_container``), but a
case or 8.3 spelling of a parent folder can reach the real repository past a by-name mask, so
masking can never be complete (reviewer probe, 2026-10-30: ``.GIT`` / ``GIT~1`` reached the host
``.git``). This is the authoritative layer: record every task-start ``.git`` at the start (a
manifest of every entry — path, type, size, sha256 — and a byte copy of the tree in harness
state outside ``work/``), and after every mutating step, and once at task end, compare the real
host ``.git`` with its manifest. Any change — an added / removed / changed entry, or a link
appearing inside — means the step reached the repository: restore it from the copy (never
following a link), verify it byte-for-byte, and end the task ``git_changed``. A ``.git`` over the
size cap is watched but not copied; a change to it ends the task ``undo_unsafe`` (it cannot be
brought back).

Never follows a link or junction: the walk, the copy and the restore all work on the
extended-length (``\\\\?\\``) form and remove a link rather than its target.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from localbot.guard.workspace_fs import (
    FILE_ATTRIBUTE_REPARSE_POINT,
    _create_new_file,
    extended_path,
    git_entries,
    remove_tree,
)

_CHUNK = 1 << 20
MIB = 1024 * 1024


class GitError(RuntimeError):
    """A task-start ``.git`` changed and could not be safely restored (quiet check failed, over
    the cap, or the restore did not verify)."""


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_link(p: Path, st: os.stat_result) -> bool:
    return bool(getattr(st, "st_file_attributes", 0) & FILE_ATTRIBUTE_REPARSE_POINT) or stat.S_ISLNK(st.st_mode)


def _walk_git(real: Path) -> dict[str, tuple[str, int, str]]:
    """The tree under a ``.git`` folder as ``rel -> (kind, size, sha256)``: ``dir`` / ``file`` /
    ``link`` / ``other``. Never follows a link. The root itself is not listed."""
    out: dict[str, tuple[str, int, str]] = {}

    def rec(folder: Path, prefix: str) -> None:
        with os.scandir(folder) as it:
            entries = list(it)
        for e in entries:
            rel = prefix + e.name
            p = Path(e.path)
            st = os.lstat(e.path)
            if _is_link(p, st):
                out[rel] = ("link", 0, "")
            elif stat.S_ISDIR(st.st_mode):
                out[rel] = ("dir", 0, "")
                rec(p, rel + "/")
            elif stat.S_ISREG(st.st_mode):
                out[rel] = ("file", st.st_size, _sha(p))
            else:
                out[rel] = ("other", 0, "")

    rec(real, "")
    return out


@dataclass
class GitEntry:
    rel: str  # the .git path within work/ (".git", "sub/.git")
    is_dir: bool
    manifest: dict[str, tuple[str, int, str]]  # the task-start tree (folder) or {"": file} (gitfile)
    size_total: int
    backup: Path
    restorable: bool = True
    reason: str = ""


@dataclass
class GitGuard:
    """Every task-start ``.git`` of one trial, its manifest and byte copy, and whether a change
    has been seen (``flagged`` = the ``git_changed`` info field)."""

    work: Path
    backup_dir: Path
    cap_bytes: int
    entries: list[GitEntry] = field(default_factory=list)
    flagged: bool = False

    @classmethod
    def record(cls, work: str | os.PathLike[str], backup_dir: str | os.PathLike[str],
               cap_bytes: int) -> "GitGuard":
        work = Path(work)
        g = cls(work, Path(backup_dir), cap_bytes)
        root = extended_path(work)
        for i, (rel, is_dir) in enumerate(git_entries(work)):
            real = root.joinpath(*rel.split("/"))
            if is_dir:
                man = _walk_git(real)
                total = sum(sz for k, sz, _ in man.values() if k == "file")
            else:
                st = os.lstat(real)
                man = {"": ("file", st.st_size, _sha(real))}
                total = st.st_size
            restorable, reason = True, ""
            if total > cap_bytes:
                restorable, reason = False, f"over the {cap_bytes // MIB} MB cap"
            elif any(k in ("link", "other") for k, _, _ in man.values()):
                restorable, reason = False, "contains a link or special file at task start"
            bdir = g.backup_dir / f"g{i}"
            if restorable:
                g._copy(real, is_dir, man, bdir)
            g.entries.append(GitEntry(rel, is_dir, man, total, bdir, restorable, reason))
        return g

    def _copy(self, real: Path, is_dir: bool, man: dict, bdir: Path) -> None:
        bdir.mkdir(parents=True, exist_ok=True)
        if not is_dir:
            shutil.copyfile(real, bdir / "blob")
            return
        for rel, (kind, _sz, _sha256) in sorted(man.items()):
            dst = bdir.joinpath(*rel.split("/"))
            if kind == "dir":
                dst.mkdir(parents=True, exist_ok=True)
            elif kind == "file":
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(real.joinpath(*rel.split("/")), dst)

    def _path_plain(self, rel: str) -> bool:
        """Every folder from ``work/`` down to ``rel``'s parent is a plain directory — no link or
        junction (S8e.2 fix B). ``lstat`` each component; never ``scandir`` through one, so a
        swapped parent (``sub`` → a link out of ``work/``) is detected without being traversed."""
        cur = extended_path(self.work)
        for part in [None, *[p for p in rel.split("/")[:-1] if p]]:
            if part is not None:
                cur = cur.joinpath(part)
            try:
                st = os.lstat(cur)
            except OSError:
                return False
            if _is_link(cur, st) or not stat.S_ISDIR(st.st_mode):
                return False
        return True

    def _current(self, e: GitEntry) -> dict[str, tuple[str, int, str]] | None:
        """The host ``.git`` now, as a manifest. None if it is gone. The caller must have checked
        ``_path_plain`` first, so this never reads through a link in the path."""
        real = extended_path(self.work).joinpath(*e.rel.split("/"))
        if not os.path.lexists(real):
            return None
        if e.is_dir:
            return _walk_git(real)
        st = os.lstat(real)
        if _is_link(real, st) or not stat.S_ISREG(st.st_mode):
            return {"": ("other", 0, "")}
        return {"": ("file", st.st_size, _sha(real))}

    def restorable(self, rel: str) -> bool:
        e = next((x for x in self.entries if x.rel == rel), None)
        return bool(e and e.restorable)

    def changed(self) -> list[str]:
        """Task-start ``.git`` entries whose host tree is not byte-for-byte the manifest. A path
        with a link/junction above the ``.git`` counts as changed **without reading through it**
        (S8e.2); a read error counts as changed too (fail closed)."""
        out = []
        for e in self.entries:
            if not self._path_plain(e.rel):
                out.append(e.rel)
                continue
            try:
                now = self._current(e)
            except OSError:
                out.append(e.rel)
                continue
            if now is None or now != e.manifest:
                out.append(e.rel)
        return out

    def restore(self, rels: list[str], quiet: Callable[[], bool]) -> None:
        """Restore each named ``.git`` from its backup (raises ``GitError`` on any failure, before
        touching that one): the container must be quiet, the ``.git`` must be restorable, and
        every folder above it must be a plain folder (fix B backstop). Verifies byte-for-byte."""
        if not quiet():
            raise GitError("a container process may be running; refusing to restore .git")
        by_rel = {e.rel: e for e in self.entries}
        for r in rels:
            e = by_rel[r]
            if not e.restorable:
                raise GitError(f".git {r!r} not restorable ({e.reason})")
            if not self._path_plain(r):
                raise GitError(f".git {r!r} path has a link or junction; refusing to restore")
            self._restore(e)

    def review(self, quiet: Callable[[], bool]) -> tuple[str, str] | None:
        """Task-end backstop (no step to undo here): None if every task-start ``.git`` is
        unchanged. Otherwise set ``flagged``; a not-restorable change → ``("undo_unsafe", why)``;
        else restore the changed ones (``restore`` raises on a quiet / path / verify failure) and
        return ``("git_changed", detail)``."""
        changed = self.changed()
        if not changed:
            return None
        self.flagged = True
        bad = [r for r in changed if not self.restorable(r)]
        if bad:
            e = next(x for x in self.entries if x.rel == bad[0])
            return "undo_unsafe", f".git {e.rel!r} changed but was not backed up ({e.reason}); cannot restore"
        self.restore(changed, quiet)
        return "git_changed", f".git changed and restored: {sorted(changed)}"

    def _restore(self, e: GitEntry) -> None:
        real = extended_path(self.work).joinpath(*e.rel.split("/"))
        remove_tree(real)  # no-follow: a link inside goes, its target does not
        if not e.is_dir:
            os.makedirs(real.parent, exist_ok=True)
            fd = _create_new_file(real)
            with os.fdopen(fd, "wb") as out, open(e.backup / "blob", "rb") as src:
                shutil.copyfileobj(src, out)
        else:
            os.makedirs(real, exist_ok=True)
            for rel, (kind, _sz, _sha256) in sorted(e.manifest.items()):
                dst = real.joinpath(*rel.split("/"))
                if kind == "dir":
                    os.makedirs(dst, exist_ok=True)
                elif kind == "file":
                    os.makedirs(dst.parent, exist_ok=True)
                    fd = _create_new_file(dst)
                    with os.fdopen(fd, "wb") as out, open(e.backup.joinpath(*rel.split("/")), "rb") as src:
                        shutil.copyfileobj(src, out)
        now = self._current(e)
        if now != e.manifest:
            raise GitError(f".git {e.rel!r} did not verify after restore")
