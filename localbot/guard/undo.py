"""Undo: a snapshot before every bash/write/edit, and restore (docs/STAGE1_SPEC.md §5.7, S7b.1).

Content-addressed file copies, not git. On this PC ``git add -A`` with ``--work-tree=<work>``
followed a junction out of work/ (it stored ``outside/secret.txt`` as ``jx/secret.txt``) and
failed the whole snapshot on a file named ``nul`` (it read the NUL device) — BUILD_LOG S7b.1.

Store layout, under ``<store_dir>/<bot>/undo/`` (outside the workspace, never mounted)::

    meta.json            {"bot": ..., "work": ...}: the one workspace this store belongs to
    objects/ab/<sha256>  file bytes, written once, shared by every snapshot that has them
    snaps/000001.json    one manifest per snapshot, oldest first

A snapshot holds exactly what the host path layer accepts: plain files and folders whose every
name passes the name rules, paths up to ``MAX_PATH_LEN``. Never stored: symlinks / junctions,
hard-linked files (st_nlink > 1), refused names (``nul``, ``CON``, U+F0xx, ``.git``, ...);
files over the size limit are skipped with a warning and named.

Restore brings work/ back to one snapshot, byte for byte: files added since are removed,
deleted files and folders (empty ones too) come back, changed or read-only files are replaced.
It never follows a link: links, junctions, hard links and refused names are removed first with
``remove_tree`` on the ``\\\\?\\`` form, and files are created with ``_create_new_file`` (fails on
anything already at the name). It runs only when the caller's ``quiet`` check says no container
process is running, and it re-scans afterwards to verify the result.

One exception (S8e): a ``.git`` that the target snapshot also listed (it was there before the
undone steps: the user's repository) is left exactly as it is — never removed, never descended
into, never rewritten. A ``.git`` the target did not list (made since) is removed like any
other refused name.

H1 G-1 makes that the general rule for everything a snapshot skips: the snapshot records each
skipped entry's ``lstat`` identity (type, inode, volume, size, mtime, link flag), and a restore
keeps an entry the target listed whose identity is unchanged (a link, a hard link, a refused
name, an over-long path, ``.localbot/``). It removes one that is new or changed — it can't be
brought back, so ``verify`` names it ``lost`` and the restore does not verify. A skipped folder
that is the same folder but changed inside is kept (removing it would delete more of what was
never stored) and named ``changed inside, not stored``: the restore does not verify either.

A skipped folder's own ``lstat`` does not move when a file is added deeper inside it (same clock
tick, size unchanged), so the snapshot also records ``skipped_inner``: one SHA-256 over the
(path, type, size, mtime, inode, volume) of every entry under each skipped folder, up to
``INNER_CAP`` entries (BRIEF_001). File bytes are never read and links are never followed.
Known limits: a rewrite that keeps size, mtime and inode; anything past the cap; what a link
inside points to. It detects a change inside; it can't undo it (the folder is never stored).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from localbot.config import UndoConfig
from localbot.guard.workspace_fs import (
    FILE_ATTRIBUTE_REPARSE_POINT,
    MAX_PATH_LEN,
    WorkspaceError,
    _check_name,
    _create_new_file,
    extended_path,
    fold_name,
    is_git_name,
    remove_tree,
)

MIB = 1024 * 1024
INNER_CAP = 20000  # entries listed under one skipped folder; past it the digest is marked truncated
_CHUNK = 1 << 20
_BOT_PART = re.compile(r"^[A-Za-z0-9._-]+$")
FILE_ATTRIBUTE_READONLY = 0x1


class UndoError(RuntimeError):
    """Undo could not snapshot, or could not restore and verify."""


@dataclass
class Snapshot:
    seq: int
    reason: str
    time: float
    files: dict[str, tuple[str, int]]  # rel -> (sha256, size)
    dirs: list[str]
    skipped_large: dict[str, int] = field(default_factory=dict)  # rel -> size (not stored)
    skipped: dict[str, str] = field(default_factory=dict)  # rel -> why (link, hard link, refused name)
    skipped_ids: dict[str, list[int]] = field(default_factory=dict)  # rel -> _skip_id (H1 G-1)
    large_ids: dict[str, list[int]] = field(default_factory=dict)  # skipped_large rel -> _skip_id (H1 G-5)
    skipped_inner: dict[str, str] = field(default_factory=dict)  # skipped folder rel -> _inner_digest

    def as_json(self) -> dict:
        return {"seq": self.seq, "reason": self.reason, "time": self.time,
                "files": {k: list(v) for k, v in self.files.items()}, "dirs": self.dirs,
                "skipped_large": self.skipped_large, "skipped": self.skipped,
                "skipped_ids": self.skipped_ids, "large_ids": self.large_ids,
                "skipped_inner": self.skipped_inner}

    @classmethod
    def from_json(cls, d: dict) -> "Snapshot":
        return cls(seq=d["seq"], reason=d["reason"], time=d["time"],
                   files={k: (v[0], v[1]) for k, v in d["files"].items()}, dirs=list(d["dirs"]),
                   skipped_large=dict(d.get("skipped_large", {})), skipped=dict(d.get("skipped", {})),
                   skipped_ids={k: list(v) for k, v in d.get("skipped_ids", {}).items()},
                   large_ids={k: list(v) for k, v in d.get("large_ids", {}).items()},
                   skipped_inner=dict(d.get("skipped_inner", {})))


@dataclass
class _Entry:
    path: Path  # \\?\ form
    st: os.stat_result


def _bot_parts(bot: str) -> list[str]:
    parts = bot.replace("\\", "/").split("/")
    if not bot or any(p in ("", ".", "..") or not _BOT_PART.match(p) for p in parts):
        raise UndoError(f"bad bot name: {bot!r}")
    return parts


def _is_link(st: os.stat_result) -> bool:
    return bool(getattr(st, "st_file_attributes", 0) & FILE_ATTRIBUTE_REPARSE_POINT) or stat.S_ISLNK(st.st_mode)


def _read_only(st: os.stat_result) -> bool:
    return bool(getattr(st, "st_file_attributes", 0) & FILE_ATTRIBUTE_READONLY) or not st.st_mode & stat.S_IWRITE


def _inside(child: Path, parent: Path) -> bool:
    c, p = os.path.normcase(str(child)), os.path.normcase(str(parent))
    try:
        return os.path.commonpath([c, p]) == p
    except ValueError:  # different drives
        return False


def scan(work: Path) -> tuple[dict[str, _Entry], list[str], dict[str, str]]:
    """Walk ``work`` (``\\\\?\\`` form) without following anything. Returns (files, dirs, skipped):
    the plain files and folders the path layer accepts, and why everything else was left out.
    A link or refused-name folder is not descended into."""
    files, dirs, skipped, _ids = _scan(work)
    return files, dirs, skipped


def _scan(work: Path) -> tuple[dict[str, _Entry], list[str], dict[str, str], dict[str, list[int]]]:
    """``scan``, plus the ``_skip_id`` of every skipped entry (H1 G-1)."""
    files: dict[str, _Entry] = {}
    dirs: list[str] = []
    skipped: dict[str, str] = {}
    ids: dict[str, list[int]] = {}

    def skip(rel: str, why: str, st: os.stat_result) -> None:
        skipped[rel] = why
        ids[rel] = _skip_id(st)

    def walk(folder: Path, prefix: str) -> None:
        with os.scandir(folder) as it:
            entries = list(it)
        for e in entries:
            rel = prefix + e.name
            st = os.lstat(e.path)  # not e.stat(): on Windows it reports st_nlink/st_ino as 0
            if _is_link(st):
                skip(rel, "link", st)
                continue
            try:
                _check_name(fold_name(e.name), original=rel)
            except WorkspaceError:
                skip(rel, "refused name", st)
                continue
            if len(rel) > MAX_PATH_LEN:
                skip(rel, "path too long", st)
                continue
            if stat.S_ISDIR(st.st_mode):
                dirs.append(rel)
                walk(Path(e.path), rel + "/")
            elif stat.S_ISREG(st.st_mode):
                if st.st_nlink > 1:
                    skip(rel, "hard link", st)
                else:
                    files[rel] = _Entry(Path(e.path), st)
            else:
                skip(rel, "not a regular file", st)

    walk(extended_path(work), "")
    return files, sorted(dirs), skipped, ids


def _skip_id(st: os.stat_result) -> list[int]:
    """What identifies a skipped entry across a step (H1 G-1): type, inode, volume, size, mtime
    and the link flag. Not the link count: another name for a hard-linked file changes nothing."""
    return [stat.S_IFMT(st.st_mode), st.st_ino, st.st_dev, st.st_size, st.st_mtime_ns,
            getattr(st, "st_file_attributes", 0) & FILE_ATTRIBUTE_REPARSE_POINT]


def _inner_digest(folder: Path) -> str:
    """One SHA-256 over what is under ``folder`` (``\\\\?\\`` form): for every entry, sorted by name,
    its relative path, type, size, mtime, inode, volume and link flag. Never follows a link, never
    reads file bytes. After ``INNER_CAP`` entries it stops and the digest says ``+truncated``. A
    folder that can't be listed is part of the digest as ``unreadable``, so a change there shows."""
    h = hashlib.sha256()
    count = 0
    truncated = False

    def walk(path: Path, prefix: str) -> None:
        nonlocal count, truncated
        try:
            with os.scandir(path) as it:
                names = sorted(e.name for e in it)
        except OSError:
            h.update(f"{prefix}\0unreadable\n".encode("utf-8", "surrogatepass"))
            return
        for name in names:
            if count >= INNER_CAP:
                truncated = True
                return
            count += 1
            rel = prefix + name
            try:
                st = os.lstat(path / name)
            except OSError:
                h.update(f"{rel}\0gone\n".encode("utf-8", "surrogatepass"))
                continue
            h.update(f"{rel}\0{json.dumps(_skip_id(st))}\n".encode("utf-8", "surrogatepass"))
            if stat.S_ISDIR(st.st_mode) and not _is_link(st):
                walk(path / name, rel + "/")
                if truncated:
                    return

    walk(folder, "")
    return h.hexdigest() + ("+truncated" if truncated else "")


def _skip_state(target: Snapshot, rel: str, st: os.stat_result, work: Path | None = None) -> str:
    """``same``: ``rel`` is a skipped entry ``target`` listed and the step left unchanged.
    ``inside``: the same skipped folder (same inode, not a link), changed inside. ``other``: not a
    skipped entry of ``target``, or a different / changed one. With ``work`` given, a skipped folder
    whose own stat is unchanged is also compared by what is inside it (BRIEF_001): ``target``'s
    ``skipped_inner`` digest against the folder now. A snapshot without a digest compares the stat
    only. A caller that treats ``same`` and ``inside`` alike (the restore's clean step) omits ``work``."""
    want = target.skipped_ids.get(rel)
    if want is None or rel not in target.skipped:
        return "other"
    now = _skip_id(st)
    same_folder = now[0] == want[0] == stat.S_IFDIR and now[1:3] == want[1:3] and now[5] == want[5] == 0
    if now == want:
        inner = target.skipped_inner.get(rel)
        if inner is None or work is None or not same_folder:
            return "same"
        return "same" if _inner_digest(extended_path(work).joinpath(*rel.split("/"))) == inner else "inside"
    return "inside" if same_folder else "other"


def _lstat_rel(work: Path, rel: str) -> os.stat_result | None:
    try:
        return os.lstat(extended_path(work).joinpath(*rel.split("/")))
    except OSError:
        return None


def _open_checked(entry: _Entry):
    """Open a scanned regular file for reading and make sure it is the same file ``lstat`` saw."""
    fh = open(entry.path, "rb")
    try:
        if not os.path.samestat(os.fstat(fh.fileno()), entry.st):
            raise UndoError(f"file changed while reading: {entry.path.name!r}")
    except BaseException:
        fh.close()
        raise
    return fh


def file_sha(entry: _Entry) -> str:
    h = hashlib.sha256()
    with _open_checked(entry) as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def changed_since(snap: Snapshot, work: Path, scanned=None) -> tuple[list[str], list[str]]:
    """What the last step left in ``work`` relative to ``snap``, for the canary check (S7b.2b):
    files that are new, or whose bytes differ (size or content), and the entries the path
    layer refuses now (links, refused names, hard links, over-long paths, odd entries — they
    are read raw by the caller and removed by a restore) that are new or changed since ``snap``
    (H1 G-1: one ``snap`` listed and the step left alone is kept by a restore, not listed; one
    that can't be lstat'ed is listed). A file the snapshot skipped as too large (bytes not stored)
    is compared by its lstat identity — size, inode, mtime (H1 G-5: before, size only, so a write
    into it that kept the size was never seen). ``scanned``: a ``scan(work)`` the caller already
    made after the step (S8c.2 shares it with the delete check)."""
    files, _dirs, skipped = scanned or scan(work)
    changed: list[str] = []
    for rel, entry in files.items():
        want = snap.files.get(rel)
        if want is None:
            if rel in snap.skipped_large and snap.skipped_large[rel] == entry.st.st_size                     and snap.large_ids.get(rel, _skip_id(entry.st)) == _skip_id(entry.st):
                continue
            changed.append(rel)
        elif want[1] != entry.st.st_size:
            changed.append(rel)
        elif file_sha(entry) != want[0]:
            changed.append(rel)
    refused = []
    for rel in sorted(skipped):
        st = _lstat_rel(work, rel)
        if st is None or _skip_state(snap, rel, st, work) != "same":
            refused.append(rel)
    return changed, refused


class UndoStore:
    """The undo store of one bot (one workspace). ``hook`` is the loop's undo callback."""

    def __init__(self, cfg: UndoConfig, bot: str, work: str | os.PathLike[str]) -> None:
        self.cfg = cfg
        self.bot = "/".join(_bot_parts(bot))
        self.work = Path(work).resolve()
        self.root = Path(cfg.store_dir).resolve().joinpath(*_bot_parts(bot), "undo")
        if _inside(self.root, self.work) or _inside(self.work, self.root):
            raise UndoError("the undo store must be outside the workspace (and not contain it)")
        self.skip_over = int(cfg.skip_files_over_mb * MIB)
        self.cap = int(cfg.store_cap_mb * MIB)
        self.warnings: list[str] = []
        self._objects = self.root / "objects"
        self._snaps = self.root / "snaps"
        meta = self.root / "meta.json"
        if meta.exists():
            known = json.loads(meta.read_text(encoding="utf-8"))
            if os.path.normcase(known["work"]) != os.path.normcase(str(self.work)):
                raise UndoError(f"store {self.bot!r} belongs to another workspace")
        else:
            self._objects.mkdir(parents=True, exist_ok=True)
            self._snaps.mkdir(parents=True, exist_ok=True)
            _write_json(meta, {"bot": self.bot, "work": str(self.work)})

    @classmethod
    def open_existing(cls, cfg: UndoConfig, bot: str) -> "UndoStore":
        meta = Path(cfg.store_dir).resolve().joinpath(*_bot_parts(bot), "undo", "meta.json")
        if not meta.exists():
            raise UndoError(f"no undo store for bot {bot!r} under {cfg.store_dir}")
        return cls(cfg, bot, json.loads(meta.read_text(encoding="utf-8"))["work"])

    # -- snapshots ---------------------------------------------------------------------------

    def hook(self, name: str, args: dict) -> Snapshot:
        """Loop callback: called before every bash/write/edit that passed the vetoes. Returns
        the snapshot taken, so the loop can compare the tree after the step (S7b.2b's canary
        check) and undo it if a secret landed in a changed file."""
        return self.snapshot(reason=name)

    def seqs(self) -> list[int]:
        return sorted(int(p.stem) for p in self._snaps.glob("*.json"))

    def load(self, seq: int) -> Snapshot:
        return Snapshot.from_json(json.loads((self._snaps / f"{seq:06d}.json").read_text(encoding="utf-8")))

    def snapshot(self, reason: str = "") -> Snapshot:
        self._check_root()
        found, dirs, skipped, skipped_ids = _scan(self.work)
        files: dict[str, tuple[str, int]] = {}
        large: dict[str, int] = {}
        large_ids: dict[str, list[int]] = {}
        for rel, entry in found.items():
            if entry.st.st_size > self.skip_over:
                large[rel] = entry.st.st_size
                large_ids[rel] = _skip_id(entry.st)
                self.warnings.append(f"undo: not stored (over {self.cfg.skip_files_over_mb} MB): "
                                     f"{rel} ({entry.st.st_size} bytes)")
                continue
            files[rel] = (self._store_object(entry), entry.st.st_size)
        inner = {rel: _inner_digest(extended_path(self.work).joinpath(*rel.split("/")))
                 for rel, ident in skipped_ids.items()
                 if ident[0] == stat.S_IFDIR and ident[5] == 0}  # a skipped folder that is not a link
        seqs = self.seqs()
        snap = Snapshot(seq=(seqs[-1] + 1) if seqs else 1, reason=reason, time=time.time(),
                        files=files, dirs=dirs, skipped_large=large, skipped=skipped,
                        skipped_ids=skipped_ids, large_ids=large_ids, skipped_inner=inner)
        _write_json(self._snaps / f"{snap.seq:06d}.json", snap.as_json())
        self.prune()
        return snap

    def _store_object(self, entry: _Entry) -> str:
        """Copy one file into objects/ (hashing while copying); returns its sha256."""
        self._objects.mkdir(parents=True, exist_ok=True)
        tmp = self._objects / f".tmp-{os.getpid()}-{time.monotonic_ns()}"
        h = hashlib.sha256()
        with _open_checked(entry) as src, open(tmp, "xb") as dst:
            for chunk in iter(lambda: src.read(_CHUNK), b""):
                h.update(chunk)
                dst.write(chunk)
        sha = h.hexdigest()
        dest = self._objects / sha[:2] / sha
        if dest.exists():
            tmp.unlink()
        else:
            dest.parent.mkdir(exist_ok=True)
            os.replace(tmp, dest)
        return sha

    # -- restore -----------------------------------------------------------------------------

    def restore(self, steps: int = 1, *, quiet: Callable[[], bool]) -> dict:
        """Undo the last ``steps`` steps: bring work/ back to the snapshot taken before the
        ``steps``-th last step, verify it, then drop the snapshots that were undone.
        ``quiet()`` must confirm that no container process is running (checked first)."""
        seqs = self.seqs()
        if steps < 1 or steps > len(seqs):
            raise UndoError(f"can undo 1..{len(seqs)} steps, asked for {steps}")
        if not quiet():
            raise UndoError("a container process may be running; refusing to restore")
        self._check_root(allow_missing=True)
        target = self.load(seqs[-steps])
        report = self._restore_to(target)
        problems = self.verify(target)
        if problems:
            raise UndoError(f"restore did not verify: {problems[:5]}")
        for seq in seqs[-steps:]:
            (self._snaps / f"{seq:06d}.json").unlink()
        self._gc()
        report.update(restored_to=target.seq, steps=steps, verified=True,
                      not_restored_large=sorted(target.skipped_large),
                      not_stored=dict(sorted(target.skipped.items())))
        return report

    def _restore_to(self, target: Snapshot) -> dict:
        removed: list[str] = []
        kept_large: list[str] = []
        kept_git: list[str] = []
        kept_skipped: list[str] = []
        want_dirs = set(target.dirs)

        def clean(folder: Path, prefix: str) -> None:
            with os.scandir(folder) as it:
                entries = list(it)
            for e in entries:
                rel = prefix + e.name
                path = Path(e.path)
                if _kept_git(rel, target):
                    kept_git.append(rel)  # the user's repository: not touched (S8e)
                    continue
                st = os.lstat(e.path)  # not e.stat(): on Windows it reports st_nlink/st_ino as 0
                if _skip_state(target, rel, st) in ("same", "inside"):
                    kept_skipped.append(rel)  # never stored, so never removed (H1 G-1); verify judges
                    continue
                ok_name = True
                try:
                    _check_name(fold_name(e.name), original=rel)
                except WorkspaceError:
                    ok_name = False
                if _is_link(st) or not ok_name or len(rel) > MAX_PATH_LEN:
                    remove_tree(path)  # never followed: the link itself goes, not its target
                    removed.append(rel)
                elif stat.S_ISDIR(st.st_mode):
                    if rel in want_dirs:
                        clean(path, rel + "/")
                    else:
                        remove_tree(path)
                        removed.append(rel + "/")
                elif stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and rel in target.skipped_large:
                    kept_large.append(rel)  # not stored: left as it is, and named
                elif stat.S_ISREG(st.st_mode) and st.st_nlink == 1 and self._same(rel, _Entry(path, st), target):
                    continue
                else:  # changed, added, read-only, hard-linked (only this name goes), or odd
                    remove_tree(path)
                    removed.append(rel)

        root = extended_path(self.work)
        if not os.path.lexists(root):
            os.mkdir(root)
        clean(root, "")
        made_dirs: list[str] = []
        for rel in sorted(target.dirs, key=lambda r: r.count("/")):
            p = root.joinpath(*rel.split("/"))
            if os.path.lexists(p):
                st = os.lstat(p)
                if _is_link(st) or not stat.S_ISDIR(st.st_mode):
                    raise UndoError(f"not a plain folder after cleaning: {rel!r}")
                continue
            os.mkdir(p)
            made_dirs.append(rel)
        written: list[str] = []
        for rel, (sha, size) in sorted(target.files.items()):
            p = root.joinpath(*rel.split("/"))
            if os.path.lexists(p):
                continue  # kept: same bytes, writable
            self._write_new(p, sha)
            written.append(rel)
        return {"removed": removed, "dirs_made": made_dirs, "written": written, "kept_large": kept_large,
                "kept_git": kept_git, "kept_skipped": kept_skipped}

    def _same(self, rel: str, entry: _Entry, target: Snapshot) -> bool:
        want = target.files.get(rel)
        if want is None or want[1] != entry.st.st_size or _read_only(entry.st):
            return False
        return file_sha(entry) == want[0]

    def _write_new(self, path: Path, sha: str) -> None:
        """Create ``path`` fresh (never through a link: CREATE_NEW / O_EXCL|O_NOFOLLOW) and copy
        the object's bytes into it."""
        src = self._objects / sha[:2] / sha
        if not src.exists():
            raise UndoError(f"undo object missing: {sha[:12]}")
        fd = _create_new_file(path)
        with os.fdopen(fd, "wb") as dst, open(src, "rb") as fh:
            for chunk in iter(lambda: fh.read(_CHUNK), b""):
                dst.write(chunk)

    def verify(self, target: Snapshot) -> list[str]:
        """Differences between work/ now and ``target`` (empty = byte-for-byte equal). Files
        that were too large to store are not compared. H1 G-1: a skipped entry ``target`` listed
        must still be there, unchanged; one that is gone is ``lost`` (never stored, can't come
        back), and a skipped folder changed inside is ``changed inside, not stored``."""
        files, dirs, skipped = scan(self.work)
        problems = []
        same: set[str] = set()
        for rel, why in sorted(skipped.items()):
            if _kept_git(rel, target):
                continue
            st = _lstat_rel(self.work, rel)
            state = "other" if st is None else _skip_state(target, rel, st, self.work)
            if state == "same":
                same.add(rel)
            elif state == "inside":
                same.add(rel)
                problems.append(f"changed inside, not stored: {rel}")
            else:
                problems.append(f"left: {rel} ({why})")
        for rel, why in sorted(target.skipped.items()):
            if not _kept_git(rel, target) and _lstat_rel(self.work, rel) is None:
                problems.append(f"lost: {rel} (was {why}; never stored)")
        for rel in files:  # a hard link whose other name went is a plain file now: still kept
            st = files[rel].st
            if rel in target.skipped and _skip_state(target, rel, st) == "same":
                same.add(rel)
        if set(dirs) != set(target.dirs):
            problems.append(f"folders differ: +{sorted(set(dirs) - set(target.dirs))} "
                            f"-{sorted(set(target.dirs) - set(dirs))}")
        have = {r for r in files if r not in target.skipped_large and r not in same}
        if have != set(target.files):
            problems.append(f"files differ: +{sorted(have - set(target.files))} "
                            f"-{sorted(set(target.files) - have)}")
        for rel in sorted(have & set(target.files)):
            if file_sha(files[rel]) != target.files[rel][0]:
                problems.append(f"bytes differ: {rel}")
        return problems

    # -- size cap ----------------------------------------------------------------------------

    def store_bytes(self) -> int:
        return sum(p.stat().st_size for p in self._objects.glob("*/*"))

    def prune(self) -> dict:
        """Keep the store under the size cap: drop the oldest snapshots (never the newest) until
        what the rest reference fits, then delete objects nothing references."""
        total = self.store_bytes()
        if total <= self.cap:
            return {"dropped": [], "store_bytes": total}
        seqs = self.seqs()
        dropped = []
        while len(seqs) > 1 and self._referenced_bytes(seqs) > self.cap:
            (self._snaps / f"{seqs[0]:06d}.json").unlink()
            dropped.append(seqs.pop(0))
        self._gc()
        total = self.store_bytes()
        if total > self.cap:
            self.warnings.append(f"undo: store over its cap with only the newest snapshot "
                                 f"({total} > {self.cap} bytes)")
        return {"dropped": dropped, "store_bytes": total}

    def _referenced(self, seqs: list[int]) -> set[str]:
        return {sha for s in seqs for sha, _ in self.load(s).files.values()}

    def _referenced_bytes(self, seqs: list[int]) -> int:
        return sum((self._objects / sha[:2] / sha).stat().st_size for sha in self._referenced(seqs))

    def _gc(self) -> None:
        keep = self._referenced(self.seqs())
        for p in self._objects.glob("*/*"):
            if p.name not in keep:
                p.unlink()

    def _check_root(self, allow_missing: bool = False) -> None:
        root = extended_path(self.work)
        if allow_missing and not os.path.lexists(root):
            return  # restore makes it again
        try:
            st = os.lstat(root)
        except OSError as exc:
            raise UndoError(f"the workspace root is missing: {exc}") from exc
        if _is_link(st) or not stat.S_ISDIR(st.st_mode):
            raise UndoError("the workspace root is not a plain folder")


def _kept_git(rel: str, target: Snapshot) -> bool:
    """A ``.git`` the target snapshot listed: the user's repository, left as it is (S8e)."""
    return rel in target.skipped and is_git_name(rel)


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    # ensure_ascii (H1 P2): a skipped name with a lone surrogate must not crash every snapshot
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=True), encoding="utf-8")
    os.replace(tmp, path)
