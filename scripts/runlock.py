"""One run lock for the whole PC, so two long jobs never overlap (audit E-3).

Overlapping runs voided each other twice: a suite or G2 run sweeps every labeled container on the
machine (audit S-2), and a running exam's HostWatch counts other runs' folders in workspaces/_trials
as host changes (S-3). So the test suite, the G2 pack and the GPU batch take this lock, from whatever
worktree they run in, and a second job is refused instead of silently spoiling the first.

The lock is a byte-range lock on byte 0 of <main checkout>/state/run.lock (LOCALBOT_RUNLOCK_PATH
overrides the path, for tests). The OS drops it when the holding process dies, however it dies, so a
stale lock can never block. Who holds it is written next to it in run.lock.json (pid, name, cmd, start
as UTC ISO + epoch); that file is information only, the lock itself is the truth. scripts/runlock.ps1
locks the same byte from PowerShell (FileStream.Lock(0, 1)), so the two exclude each other.

    python scripts/runlock.py status                    # FREE (exit 0) or BUSY + holder json (exit 3)
    python scripts/runlock.py run <name> -- <cmd ...>   # hold the lock while cmd runs, exit with its code;
                                                        # busy -> "busy: <holder json>", exit 3

A job that holds the lock exports LOCALBOT_RUNLOCK_HELD=1 to its children. A child does not take the
lock again only when held_by_parent() is True: the flag is set, the lock is really held, AND the
holder is the child itself or one of its ancestors (checked with psutil). A stray flag (lock free, or
held by an unrelated process) is ignored, and the child tries to take the lock as usual. The standard
library only, plus psutil for that ancestor check (without psutil the check fails safe: False).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if os.name == "nt":
    import msvcrt
else:
    import fcntl

HELD_ENV = "LOCALBOT_RUNLOCK_HELD"
PATH_ENV = "LOCALBOT_RUNLOCK_PATH"
BUSY_EXIT = 3
REPO = Path(__file__).resolve().parents[1]
WIN = os.name == "nt"
TRIES = 3  # a status probe holds the lock for an instant; a couple of quick retries ride over it


class LockBusy(Exception):
    """Another process holds the lock. ``holder`` is its run.lock.json as a dict, or None."""

    def __init__(self, holder: dict | None, path: Path):
        self.holder = holder
        self.path = path
        super().__init__(f"busy: {describe(holder)}")


def describe(holder: dict | None) -> str:
    """The holder as one line of JSON (what "busy: ..." prints)."""
    return json.dumps(holder) if holder else '{"holder": "unknown: no readable run.lock.json"}'


def main_checkout(repo: Path | None = None) -> Path:
    """The main checkout of ``repo``, read from .git without running git.

    .git is a dir: the repo is the main checkout. .git is a file "gitdir: <p>" (a worktree):
    <p>/commondir names the shared .git dir (relative to <p>), whose parent is the main checkout.
    Anything else falls back to the repo itself."""
    repo = Path(repo) if repo else REPO
    dot = repo / ".git"
    if dot.is_dir():
        return repo
    try:
        first = dot.read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IndexError, UnicodeDecodeError):
        return repo
    if not first.startswith("gitdir:"):
        return repo
    gitdir = Path(first[len("gitdir:"):].strip())
    if not gitdir.is_absolute():
        gitdir = repo / gitdir
    try:
        common = Path((gitdir / "commondir").read_text(encoding="utf-8").strip())
    except OSError:
        return repo
    if not common.is_absolute():
        common = gitdir / common
    common = common.resolve()
    return common.parent if common.is_dir() else repo


def lock_path() -> Path:
    env = os.environ.get(PATH_ENV)
    return Path(env) if env else main_checkout() / "state" / "run.lock"


def _holder_file(path: Path) -> Path:
    return path.with_name(path.name + ".json")


def _try_lock(fd: int) -> bool:
    for i in range(TRIES):
        try:
            if WIN:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # byte 0, the byte runlock.ps1 locks
            else:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            if i < TRIES - 1:
                time.sleep(0.05)
    return False


def _unlock(fd: int) -> None:
    try:
        if WIN:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass


def _write_json(file: Path, data: dict) -> None:
    """Atomic: a reader sees the old file or the new one, never half of one."""
    tmp = file.with_name(f"{file.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data) + "\n", encoding="utf-8", newline="\n")
    for i in range(20):
        try:
            os.replace(tmp, file)
            return
        except PermissionError:  # Windows: a reader has the file open this instant
            time.sleep(0.05)
    os.replace(tmp, file)


def holder(path: Path | None = None) -> dict | None:
    """run.lock.json as a dict, or None. Information only: it can be stale (see is_held)."""
    file = _holder_file(Path(path) if path else lock_path())
    try:
        d = json.loads(file.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def is_held(path: Path | None = None) -> bool:
    """True if some process holds the lock right now (this one included). When it is free this
    takes it for an instant and lets it go."""
    path = Path(path) if path else lock_path()
    if not path.exists():
        return False
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return False
    try:
        if _try_lock(fd):
            _unlock(fd)
            return False
        return True
    finally:
        os.close(fd)


def _lineage() -> dict[int, float] | None:
    """{pid: create time} of this process and its ancestors, or None without psutil. psutil's
    parent() only follows a parent that is older than its child, so a reused ppid is not followed."""
    try:
        import psutil  # optional: imported here so the rest of the module stays standard library
    except ImportError:
        return None
    try:
        me = psutil.Process()
        chain = [me, *me.parents()]
    except psutil.Error:
        return None
    out = {}
    for p in chain:
        try:
            out[p.pid] = p.create_time()
        except psutil.Error:
            pass
    return out


def held_by_parent() -> bool:
    """True when this process runs under the job that holds the lock. All of these must hold:
    the flag is "1"; the lock is really held; the holder named in run.lock.json is this process or
    one of its ancestors; and that process already existed when the lock was taken (its create time
    is not later than the holder's start_epoch, so a reused pid does not count).

    Anything else gives False and the caller tries to take the lock itself (and is refused while
    someone else holds it): a stray flag with a free lock, a lock held by an unrelated process
    (a sibling, another shell, PowerShell), a missing or unreadable holder file, or no psutil."""
    if os.environ.get(HELD_ENV) != "1" or not is_held():
        return False
    h = holder()
    try:
        pid = int(h["pid"])
    except (TypeError, KeyError, ValueError):
        return False
    lineage = _lineage()
    if not lineage or pid not in lineage:
        return False
    start = h.get("start_epoch")
    if isinstance(start, (int, float)) and lineage[pid] > start + 2:
        return False
    return True


def _cmd_text(cmd) -> str:
    if cmd is None:
        cmd = sys.argv
    if isinstance(cmd, (list, tuple)):
        return " ".join(str(c) for c in cmd)
    return str(cmd)


class RunLock:
    """A held lock. release() (or leaving the ``with`` block) lets it go; so does process death."""

    def __init__(self, path: Path, fd: int, info: dict):
        self.path = path
        self.info = info
        self._fd: int | None = fd

    @property
    def held(self) -> bool:
        return self._fd is not None

    def release(self) -> None:
        if self._fd is None:
            return
        try:  # the holder file goes first, while the lock still keeps everyone else out
            h = holder(self.path)
            if h and h.get("pid") == self.info["pid"] and h.get("start") == self.info["start"]:
                try:
                    _holder_file(self.path).unlink()
                except OSError:
                    pass
        finally:
            fd, self._fd = self._fd, None
            _unlock(fd)
            os.close(fd)

    def __enter__(self) -> RunLock:
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def acquire(name: str, cmd=None, path: Path | None = None) -> RunLock:
    """Take the lock or raise LockBusy at once (never waits). ``cmd`` (list or str, default this
    process's argv) goes into the holder file with the pid, ``name`` and the start time."""
    path = Path(path) if path else lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    if not _try_lock(fd):
        os.close(fd)
        h = None
        for _ in range(10):  # the holder writes its file right after it takes the lock
            h = holder(path)
            if h:
                break
            time.sleep(0.05)
        raise LockBusy(h, path)
    now = time.time()
    info = {"pid": os.getpid(), "name": name, "cmd": _cmd_text(cmd),
            "start": datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "start_epoch": round(now, 3)}
    try:
        _write_json(_holder_file(path), info)
    except OSError:
        pass  # the holder file is information only; the lock is held either way
    return RunLock(path, fd, info)


USAGE = ("usage: runlock.py status\n"
         "       runlock.py run <name> -- <command ...>")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["status"]:
        path = lock_path()
        h = holder(path)
        if is_held(path):
            print(f"BUSY {describe(h)}\n  lock: {path}")
            return BUSY_EXIT
        print(f"FREE\n  lock: {path}")
        if h:
            print(f"  (stale holder file from a process that no longer holds it: {json.dumps(h)})")
        return 0
    if len(argv) >= 4 and argv[0] == "run" and argv[2] == "--":
        name, cmd = argv[1], argv[3:]
        lk = None
        if not held_by_parent():
            try:
                lk = acquire(name, cmd=cmd)
            except LockBusy as e:
                print(f"busy: {describe(e.holder)}", flush=True)
                return BUSY_EXIT
        env = dict(os.environ)
        env[HELD_ENV] = "1"
        try:
            return subprocess.call(cmd, env=env)
        except OSError as e:
            print(f"cannot run {cmd[0]!r}: {e}", file=sys.stderr)
            return 127
        except KeyboardInterrupt:
            return 130
        finally:
            if lk:
                lk.release()
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
