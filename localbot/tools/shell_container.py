"""Real bash in a podman/docker container (docs/STAGE1_SPEC.md §5.6, step S7a).

One container per trial, started once with the spec flags and held open by a tiny python
keepalive that is pid 1. Each tool call is a separate ``exec`` into it, with a 30 s timeout and
a 4,000-character output cap. The image is pinned by digest (config); the same digest runs on
podman and docker.

The container is locked down exactly as the spec says: ``--network none --read-only --tmpfs
/tmp --cap-drop ALL --security-opt no-new-privileges --pids-limit 256 --memory 2g --cpus 2
--user 1000:1000 -v <work>:/work``. Nothing else that changes isolation: no ``--init``
(it bind-mounts an init binary), no socket mount, no host env (``--env`` is never passed), no
extra mount. Besides those, ``run`` only gets ``-d``, ``--label`` (for the sweep),
``--entrypoint`` (the keepalive) and ``-w /``.

H1 S-1: pid 1 is never started inside the workspace. ``python3 -c`` puts its working folder first
on ``sys.path``, and the keepalive imports ``signal`` / ``ctypes`` (and through them ``enum``), so
a ``/work/enum.py`` used to run **as pid 1** at the next start, where the reaper cannot reach it.
Now the container's working folder is ``/``, pid 1 runs ``python3 -I -S`` (no working folder, no
script folder, no ``PYTHON*`` variables, no site), and each model command gets ``-w /work`` on
its own ``exec``. After every step a check confirms pid 1 is still one thread with its own
command line; if not, the container is replaced (and a fresh one that fails it raises).

Also H1 (Track C): containers carry an owner (pid + start time) and an expiry, and the sweep
removes only those nobody owns (S-2); ``--rm`` + a keepalive lifetime end a lost container, and a
container near its end is renewed before a command; podman gets ``--http-proxy=false``; each
command prints a start marker, so a failed exec is an error, not output; the scratch folders are
emptied after every step; every secret-named entry at start is masked out of the container
(decision 3), each mask checked mounted after every step; and work/ has a growth quota (S-4).

One exception (S8e): every ``.git`` that is in ``work/`` when the container first starts gets an
empty, read-only mask, so the container can neither read nor write the user's repository: a
``.git`` folder is covered by ``--tmpfs <dest>:ro``, a gitfile (worktree / submodule) by a
read-only bind of an empty host file. The masks are fixed at the first start and reused on a
restart (a ``.git`` made later is the loop's business: it vetoes and undoes the step).

After **every** exec a reaper makes sure nothing the model started is alive (S2.5 TOCTOU): it
sends ``kill -9 -1`` (every process the user may signal, except the kill itself and pid 1, which
the kernel protects inside its namespace), then lists what is left. If the listing fails — a
fork bomb holding all 256 pids makes every new exec fail — or anything survives two passes, the
container is removed and started again on the same ``work/`` (``/tmp`` is lost). If even that
cannot be confirmed, ``ContainerError`` is raised and the task ends. One exception (H1-L L2): a
fresh container whose only problem is masks it can't place (a step renamed a masked ``.git``'s
parent and linked the old name) is removed too, and ``run`` returns with no container at all; the
next command first needs a fresh container that confirms every mask. So when ``run`` returns,
no process the model started is alive, and a host-side tool may touch ``work/``.
"""
from __future__ import annotations

import base64
import json
import os
import re
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Protocol

from localbot.config import ContainerConfig
from localbot.guard import workspace_fs
from localbot.guard.secret_names import SecretNames
from localbot.guard.workspace_fs import (get_short_name, extended_path, git_entries,
                                         git_mask_spellings, is_git_name)

OUTPUT_CAP = 4000
# H1-K K6 D1 (owner decision): a mask sits on one spelling of a path, so the container starts only
# on a case-sensitive work folder (fail closed).
CASE_SENSITIVE_MSG = ("refused to start: the work folder {work} is not case-sensitive, so a masked secret "
                      "could be read through another spelling of its name. Local Bot makes a new work folder "
                      "case-sensitive before its first file (workspace_fs.new_work_folder); a folder that "
                      "already holds files can't be switched")

# pid 1. It reaps every orphan it adopts (no zombies, so no --init), and it installs no signal
# handler (python's SIGINT handler is reset), so the kernel drops every signal a process inside
# the namespace sends it: the model cannot stop the container from inside. It also makes itself
# non-dumpable (prctl PR_SET_DUMPABLE 0) and never execs (exec would reset it, S7b.2e): at
# ptrace_scope 0 a same-uid process (uid 1000, CAP_SYS_PTRACE dropped) cannot ptrace-attach to it
# or read /proc/1/mem, so code injected into pid 1 cannot keep acting on /work after a step. If
# the prctl fails (non-zero return), the keepalive exits non-zero and the container fails to
# start, rather than running pid 1 as dumpable (S8a). H1 P2: it lives ``sys.argv[1]`` seconds
# (``max_lifetime_s``) and then exits, so a container whose runner was killed stops by itself
# and ``--rm`` removes it; reaping is a non-blocking poll (1 = WNOHANG) every 0.5 s for that.
KEEPALIVE = (
    "import os, signal, time\n"
    "import ctypes, sys\n"
    "signal.signal(signal.SIGINT, signal.SIG_DFL)\n"
    "if ctypes.CDLL(None).prctl(4, 0, 0, 0, 0) != 0:\n"  # PR_SET_DUMPABLE=4 -> 0
    "    sys.exit(1)\n"
    "end = time.monotonic() + float(sys.argv[1])\n"
    "while time.monotonic() < end:\n"
    "    try:\n"
    "        while os.waitpid(-1, 1)[0]:\n"
    "            pass\n"
    "    except ChildProcessError:\n"
    "        pass\n"
    "    time.sleep(0.5)\n"
)

# H1 P2: each model command's bash first prints a one-time marker, so a failed exec (container
# gone, runtime error) can't pass its own error text off as the command's output.
EXEC_WRAP = 'printf "%s\\n" "$1"; exec bash -lc "$2"'

# H1 S-1: how pid 1 runs the keepalive. -I: no working / script folder on sys.path, PYTHON*
# variables ignored, no user site; -S: no site module. Started with -w / (run_argv).
KEEPALIVE_FLAGS = ("-I", "-S")

# H1 S-2: who started a container and until when it may live (labels; the sweep reads them).
# The owner is "<pid>:<that process's start time>", so a pid the OS has handed to another
# process does not count as the owner.
OWNER_LABEL = "localbot-owner"
EXPIRES_LABEL = "localbot-expires"


def _owner() -> str:
    try:
        import psutil
        return f"{os.getpid()}:{psutil.Process().create_time():.3f}"
    except Exception:  # noqa: BLE001 - no psutil: the owner can't be checked, so it is never "dead"
        return f"{os.getpid()}:0"


def _owner_alive(owner: str) -> bool:
    """False only when the owner process is provably gone: no such pid, a zombie, or a pid now
    held by a process that started at another time. Anything unreadable counts as alive."""
    pid_s, _, ctime_s = owner.partition(":")
    try:
        pid, ctime = int(pid_s), float(ctime_s)
        import psutil
    except Exception:  # noqa: BLE001
        return True
    if ctime == 0:
        return True
    try:
        p = psutil.Process(pid)
        if p.status() == psutil.STATUS_ZOMBIE:
            return False
        return abs(p.create_time() - ctime) < 1.0
    except psutil.NoSuchProcess:
        return False
    except Exception:  # noqa: BLE001 - access denied etc.: can't prove it is gone
        return True


_CREATED = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d)$")


def _created_ts(text: str) -> float | None:
    """``Created`` from ``inspect`` (docker: ``…Z``, podman: ``…+01:00``, nanoseconds) as epoch
    seconds; None if it can't be read."""
    m = _CREATED.match(text or "")
    if not m:
        return None
    frac = (m.group(2) or "0")[:6].ljust(6, "0")
    tz = "+00:00" if m.group(3) == "Z" else m.group(3)
    try:
        return datetime.fromisoformat(f"{m.group(1)}.{frac}{tz}").timestamp()
    except ValueError:
        return None


def _stale(c: dict, cfg: ContainerConfig, now: float) -> bool:
    labels = (c.get("Config") or {}).get("Labels") or {}
    try:
        if EXPIRES_LABEL in labels and now > float(labels[EXPIRES_LABEL]):
            return True
    except ValueError:
        pass
    if OWNER_LABEL in labels:
        return not _owner_alive(labels[OWNER_LABEL])
    created = _created_ts(c.get("Created", ""))
    return created is not None and now - created > cfg.legacy_stale_s


class Shell(Protocol):
    def run(self, command: str, timeout_s: float) -> str:
        """Run one command in the workspace; return combined output (the loop also caps it)."""
        ...


# Appended by ``run`` when it had to restart the container. The harness speaking, not tool text:
# loop.OWN_SHELL_NOTES imports this so the untrusted-output wrapper exempts it (H1-I item 9).
RESTART_NOTE = ("\n[harness: processes left by this command could not be killed; the container "
                "was restarted (work/ kept, /tmp cleared)]")


class ContainerError(RuntimeError):
    """The container runtime itself failed, or the container could not be confirmed free of
    model processes. The loop ends the task on it: no host-side tool may run after it."""


class WorkQuotaError(ContainerError):
    """H1 S-4: work/ grew past its quota since the shell's first start. The command was stopped
    and the task ends here, before the next undo snapshot copies the flood. Raised after the reap
    (the container is confirmed quiet), so the loop undoes the step first (H1-I item 7)."""


class ExecNotStartedError(ContainerError):
    """H1 P2: the runtime never started the command (container gone, CLI error). Raised after the
    reap, like ``WorkQuotaError``: the loop undoes the step, then ends ``exec_failed``."""


class MaskCheckError(ContainerError):
    """H1-L L2: a fresh container whose only problem is masks it could not place (a step renamed a
    masked ``.git``'s parent and linked the old name to it, so the mount lands through the link).
    It was removed: no container is left."""


MASKS_NOT_MOUNTED = "masks not mounted"  # the check's problem text (H1 C.4)


def work_usage(root: str | os.PathLike[str], stop_bytes: int | None = None,
               stop_entries: int | None = None) -> tuple[int, int]:
    """(bytes of regular files, entries of any kind) under ``root``, never following a link,
    on the ``\\\\?\\`` form. Stops counting as soon as either limit is passed (a 200k-file flood
    costs only the limit). Unreadable folders are skipped."""
    total = entries = 0
    stack = [extended_path(root)]
    while stack:
        try:
            it = os.scandir(stack.pop())
        except OSError:
            continue
        with it:
            for e in it:
                entries += 1
                try:
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    continue
                link = getattr(st, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT or stat.S_ISLNK(st.st_mode)
                if stat.S_ISDIR(st.st_mode) and not link:
                    stack.append(Path(e.path))
                elif stat.S_ISREG(st.st_mode):
                    total += st.st_size
                if (stop_bytes is not None and total > stop_bytes) or (stop_entries is not None and entries > stop_entries):
                    return total, entries
    return total, entries


def cap(text: str, limit: int = OUTPUT_CAP) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n[truncated {len(text) - limit} chars]"


class ContainerShell:
    """One locked-down container for one trial. ``start`` then ``run`` repeatedly, ``close`` in a
    ``finally``. Not thread-safe; the loop drives it one command at a time."""

    REAP_PASSES = 2

    def __init__(self, cfg: ContainerConfig, work_dir: Path, label: str | None = None,
                 reaper_timeout_s: float = 10.0) -> None:
        if not cfg.image:
            raise ContainerError("no container image configured")
        self.cfg = cfg
        self.exe = cfg.runtime  # "podman" | "docker", found on PATH
        self.work = Path(work_dir).resolve()
        self.label = label or f"{cfg.label_key}={uuid.uuid4().hex[:12]}"
        self.reaper_timeout_s = reaper_timeout_s
        self.cid: str | None = None
        self.restarts: list[str] = []  # why each restart happened (evidence for the trial)
        self.git_masks: list[tuple[str, bool]] | None = None  # (rel, is_dir), fixed at first start (S8e)
        self._empty_file: Path | None = None  # the host file a gitfile mask binds (S8e)
        self._expires: int | None = None  # H1 S-2: this container's expiry label (epoch s)
        self._scratch_keep: set[str] | None = None  # H1 P2: scratch entries the fresh container had
        self._notice_file: Path | None = None  # H1: the host file a secret mask binds
        self._mask_dests: list[str] | None = None  # every mask of the last run_argv
        self._mask_stats: dict[str, tuple] = {}  # masked secret rel -> host stat key at start
        self._usage_base: tuple[int, int] | None = None  # H1 S-4: work/ at the FIRST start
        self._remake: str | None = None  # H1-L L2: why the reap left no container (made before the next command)

    # -- lifecycle ---------------------------------------------------------------------------

    @classmethod
    def sweep(cls, cfg: ContainerConfig, now: float | None = None) -> list[str]:
        """Remove leftover containers carrying our label key (run start) — only those nobody
        owns any more (H1 S-2): the owner process is gone (or its pid now belongs to another
        process), or the container is past its expiry, or it predates the owner label and is
        older than ``legacy_stale_s``. A live run's container is never touched. If the runtime's
        answer can't be read, nothing is removed. Returns the removed ids."""
        ids = _run([cfg.runtime, "ps", "-aq", "--filter", f"label={cfg.label_key}"], timeout=30).stdout.split()
        if not ids:
            return []
        try:
            r = _run([cfg.runtime, "inspect", *ids], timeout=60)
            info = json.loads(r.stdout) if r.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired, ValueError):
            info = None
        if not isinstance(info, list):
            return []
        now = time.time() if now is None else now
        gone = []
        for c in info:
            if isinstance(c, dict) and c.get("Id") and _stale(c, cfg, now):
                _run([cfg.runtime, "rm", "-f", c["Id"]], timeout=60)
                gone.append(c["Id"])
        return gone

    def keepalive_args(self) -> list[str]:
        """pid 1's arguments after ``python3`` (also what its ``/proc/1/cmdline`` must show)."""
        return [*KEEPALIVE_FLAGS, "-c", KEEPALIVE, str(self.cfg.max_lifetime_s)]

    def _runtime_flags(self) -> list[str]:
        """Per-runtime options. podman copies HTTP(S)_PROXY from its own environment into the
        container by default (H1 P2): off. docker has no such option."""
        return ["--http-proxy=false"] if self.exe == "podman" else []

    def run_argv(self) -> list[str]:
        """The exact ``run`` command line (tested against the spec flags). The expiry is fixed
        until the next ``start`` (each new container gets a new one)."""
        if self._expires is None:
            self._expires = int(time.time() + self.cfg.max_lifetime_s)
        return [self.exe, "run", "-d", "--rm", "--label", self.label,
                "--label", f"{OWNER_LABEL}={_owner()}", "--label", f"{EXPIRES_LABEL}={self._expires}",
                *self.cfg.run_flags, *self._runtime_flags(),
                "-v", f"{self.work}:{self.cfg.workdir}", *self._mask_args(), "-w", "/",
                "--entrypoint", "python3", self.cfg.image, *self.keepalive_args()]

    def exec_argv(self, command: str, marker: str) -> list[str]:
        """One model command: its own exec, in ``/work`` (S-1: pid 1 is not), printing
        ``marker`` before the command runs (P2)."""
        return [self.exe, "exec", "-w", self.cfg.workdir, self.cid, "bash", "-c", EXEC_WRAP, "_", marker, command]

    def _mask_args(self) -> list[str]:
        """S8e: an empty read-only mask over each ``.git`` found at the first start that is still
        there (a mount over a missing path would make the runtime create it in work/). S8e.1:
        also its case and 8.3 spellings, each verified to resolve to the same entry."""
        if self.git_masks is None:
            masks: list[tuple[str, bool]] = []
            for rel, is_dir in git_entries(self.work):
                masks.append((rel, is_dir))
                masks.extend(git_mask_spellings(self.work, rel))
            seen: set[str] = set()
            self.git_masks = [(r, d) for r, d in masks if not (r in seen or seen.add(r))]
        root = extended_path(self.work)
        git = [(r, d) for r, d in self.git_masks if os.path.lexists(root.joinpath(*r.split("/")))]
        secret = self._secret_masks(len(git))
        args: list[str] = []
        dests: list[str] = []
        for (rel, is_dir), notice in [*((m, False) for m in git), *((m, True) for m in secret)]:
            dest = f"{self.cfg.workdir}/{rel}"
            dests.append(dest)
            if is_dir:
                # podman copies the folder underneath into a --tmpfs by default (tmpcopyup): the
                # mask would be a readable copy of the repo, not empty. docker has no copy-up.
                opts = "ro,notmpcopyup" if self.exe == "podman" else "ro"
                args += ["--tmpfs", f"{dest}:{opts}"]
            else:
                args += ["-v", f"{self._mask_file(notice)}:{dest}:ro"]
        self._mask_dests = dests
        return args

    def _secret_masks(self, used: int) -> list[tuple[str, bool]]:
        """H1 decision 3: every secret-named entry in work/ now, at its exact path (refuse to
        start if they don't all fit in ``max_masks``), plus its case / 8.3 spellings while room
        is left (NTFS opens ``.ENV`` for ``.env``; prevention, the content checks stay the
        second layer). Records each masked entry's host stat for the pre-exec check."""
        self._mask_stats = {}
        if not self.cfg.secret_masks:
            return []
        exact = secret_entries(self.work, self.cfg)
        need = used + len(exact)
        if need > self.cfg.max_masks:  # H1-K D3: the message names the total it compared, and the cap
            raise ContainerError(f"too many masks to start the container: {need} masks needed ({len(exact)} secret "
                                 f"entries + {used} for .git), max_masks {self.cfg.max_masks}. Nothing was started; "
                                 f"move secret files out of the work folder, or raise max_masks")
        spelled = [s for rel, is_dir in exact for s in mask_spellings(self.work, rel, is_dir)]
        room = self.cfg.max_masks - used - len(exact)
        self._mask_stats = {rel: _stat_key(self.work, rel) for rel, _d in exact}
        return exact + spelled[:room]

    def _mask_file(self, notice: bool) -> Path:
        """The host file a file mask binds: empty for a gitfile (S8e), the notice for a secret."""
        if notice:
            if self._notice_file is None:
                fd, name = tempfile.mkstemp(prefix="localbot-secretmask-")
                os.write(fd, MASK_NOTICE.encode())
                os.close(fd)
                self._notice_file = Path(name)
            return self._notice_file
        if self._empty_file is None:
            fd, name = tempfile.mkstemp(prefix="localbot-gitmask-")
            os.close(fd)
            self._empty_file = Path(name)
        return self._empty_file

    def _masks_changed(self) -> bool:
        """A masked secret entry changed on the host since this container started (e.g. an undo
        restore rewrote it): the mount over it may be gone, so the container is re-made."""
        return any(_stat_key(self.work, rel) != key for rel, key in self._mask_stats.items())

    def start(self) -> None:
        if not workspace_fs.is_case_sensitive(self.work):  # H1-K D1: before any runtime command
            raise ContainerError(CASE_SENSITIVE_MSG.format(work=self.work))
        self._expires = None  # a new container, a new expiry
        self._scratch_keep = None  # and a new scratch baseline (its first check lists it)
        if self._usage_base is None:  # S-4: never re-baselined by a restart
            self._usage_base = work_usage(self.work)
        try:
            r = _run(self.run_argv(), timeout=120)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ContainerError(f"{self.exe} run failed: {exc}") from exc
        if r.returncode != 0:
            raise ContainerError(f"{self.exe} run failed: {(r.stderr or r.stdout).strip()[:300]}")
        self.cid = r.stdout.strip().splitlines()[-1].strip() if r.stdout.strip() else ""
        if not self.cid:
            raise ContainerError(f"{self.exe} run returned no container id")
        problems = self.check()
        if problems:
            cid, self.cid = self.cid, None
            self._remove(cid)
            masks_only = all(p.endswith(MASKS_NOT_MOUNTED) for p in problems)
            raise (MaskCheckError if masks_only else ContainerError)(
                f"fresh container failed its check: {'; '.join(problems)}")
        self._remake = None

    def close(self) -> None:
        if self.cid:
            cid, self.cid = self.cid, None
            self._remove(cid)
        for attr in ("_empty_file", "_notice_file"):
            f = getattr(self, attr)
            if f is not None:
                f.unlink(missing_ok=True)
                setattr(self, attr, None)

    def _remove(self, cid: str) -> None:
        """``rm -f`` and confirm it is gone; raise if it can't be confirmed."""
        try:
            _run([self.exe, "rm", "-f", cid], timeout=60)
            left = _run([self.exe, "ps", "-aq", "--no-trunc", "--filter", f"id={cid}"], timeout=30).stdout.split()
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ContainerError(f"could not confirm container {cid[:12]} removed: {exc}") from exc
        if left:
            raise ContainerError(f"container {cid[:12]} still present after rm -f")

    def __enter__(self) -> "ContainerShell":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- exec --------------------------------------------------------------------------------

    def run(self, command: str, timeout_s: float) -> str:
        if not self.cid:
            if self._remake is None:
                raise ContainerError("container not started")
            self._restart(f"{self._remake}; made again before the next command")  # H1-L L2: raises if unconfirmed
        elif self._masks_changed():  # before the command: it must run with every mask in place
            self._restart("a masked secret changed on the host; re-masked before the next command")
        elif self._expires is not None and time.time() > self._expires - (timeout_s + 60):
            self._restart("container lifetime almost over; renewed before the next command")
        restarts = len(self.restarts)
        marker = f"LOCALBOT-{uuid.uuid4().hex}"
        try:
            out, timed_out = _run_capped(self.exec_argv(command, marker), max(1.0, timeout_s),
                                         self.cfg.output_cap_chars + len(marker) + 1,
                                         watch=lambda: self.over_quota() is not None, poll_s=self.cfg.quota_poll_s)
        except OSError as exc:
            raise ContainerError(f"{self.exe} exec failed: {exc}") from exc
        finally:
            self._reap()  # raises ContainerError if the container can't be confirmed clean
        started = out.startswith(marker + "\n")
        out = out[len(marker) + 1:] if started else out
        if not started and not timed_out:
            raise ExecNotStartedError(f"{self.exe} exec did not start the command: {out.strip()[:200]}")
        over = self.over_quota()  # S-4: the container is quiet now; before any next snapshot
        if over:
            raise WorkQuotaError(over)
        if timed_out:
            out += f"\n[timeout after {timeout_s:.0f}s; killed in container]"
        if len(self.restarts) > restarts:
            out += RESTART_NOTE
        return out

    def over_quota(self) -> str | None:
        """H1 S-4: why work/ is over its growth quota since the first start, or None."""
        base_b, base_e = self._usage_base or (0, 0)
        lim_b = base_b + self.cfg.work_growth_max_bytes
        lim_e = base_e + self.cfg.work_growth_max_entries
        b, e = work_usage(self.work, stop_bytes=lim_b, stop_entries=lim_e)
        if b > lim_b:
            return f"work/ grew past its quota: over {self.cfg.work_growth_max_bytes} bytes added since the start"
        if e > lim_e:
            return f"work/ grew past its quota: over {self.cfg.work_growth_max_entries} entries added since the start"
        return None

    def _reap(self) -> None:
        """Kill everything the last exec left, then confirm by listing. ``kill -9 -1`` from a
        fresh exec reaches every process of the (single) container user except the kill itself
        and pid 1. A respawning tree can race one pass, hence two. A failed listing counts as
        'not clean' (fail closed), and then the container is replaced."""
        if not self.cid:
            return
        for _ in range(self.REAP_PASSES):
            self._exec_quiet(["bash", "-c", "kill -9 -1"])  # builtin kill: the caller is the bash
            if self.alive_pids() == []:
                problems = self.check()  # H1: pid 1 still the keepalive (S-1), every mask mounted (C.4)
                if problems:
                    self._replace("; ".join(problems))
                return
        self._replace("survivors or unreadable process list after kill")

    def _replace(self, why: str) -> None:
        """The reap's restart. H1-L L2: if the fresh container fails only on masks it can't place,
        both containers are gone (each removal confirmed), so nothing the step started is running.
        The step returns, the loop's after-step checks run on the host (they undo a parent swap and
        restore ``.git``), and ``run`` makes a container again before the next command. If that one
        can't confirm its masks either, ``run`` raises and the command never runs."""
        try:
            self._restart(why)
        except MaskCheckError as exc:
            self._remake = str(exc)

    def check(self) -> list[str]:
        """The post-exec check (H1), one exec of bash + coreutils/findutils: pid 1 must be one
        thread whose ``/proc/1/cmdline`` is still ``python3`` + ``keepalive_args()`` (S-1), and
        the scratch folders (P2) are emptied of everything the fresh container did not have.
        The first check of a container (in ``start``) only lists the scratch folders: that is
        its baseline (e.g. podman's ``/run/.containerenv``). Returns what is wrong; empty =
        clean. An exec that fails or answers without its tag is a problem (fail closed)."""
        script = integrity_script(self.cfg.scratch_dirs, self._scratch_keep)
        try:
            r = _run([self.exe, "exec", self.cid, "bash", "-c", script], timeout=self.reaper_timeout_s)
        except (OSError, subprocess.TimeoutExpired):
            return ["post-exec check did not run"]
        got = parse_integrity(r.stdout) if r.returncode == 0 else None
        if got is None:
            return ["post-exec check did not run"]
        problems = []
        mounted = parse_mountinfo(_b64(got.get("MOUNTS", "")))
        missing = [d for d in self.mask_dests() if d not in mounted]
        if missing:
            problems.append(f"{len(missing)} {MASKS_NOT_MOUNTED}")
        listed = parse_list(got.get("LIST", ""))
        if self._scratch_keep is None:
            self._scratch_keep = listed
        elif listed - self._scratch_keep:
            problems.append(f"{len(listed - self._scratch_keep)} scratch entries left after the clean")
        if got.get("TASKS") != "1":
            problems.append(f"pid 1 has {got.get('TASKS', '?')} threads")
        args = _b64(got.get("CMD", "")).split(b"\0")
        if args and args[-1] == b"":
            args.pop()
        want = [a.encode() for a in self.keepalive_args()]
        if not args or os.path.basename(args[0].decode(errors="replace")) != "python3" or args[1:] != want:
            problems.append("pid 1 command line changed")
        return problems

    def mask_dests(self) -> list[str]:
        """Container paths of the masks this container was started with (``-v``/``--tmpfs``)."""
        if self._mask_dests is None:
            self._mask_args()
        return list(self._mask_dests or [])

    def _restart(self, why: str) -> None:
        self.restarts.append(why)
        cid, self.cid = self.cid, None
        if cid:
            self._remove(cid)
        self.start()
        if self.alive_pids() != []:
            raise ContainerError("fresh container is not clean")

    def _exec_quiet(self, argv: list[str]) -> bool:
        try:
            return _run([self.exe, "exec", self.cid, *argv], timeout=self.reaper_timeout_s).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def processes(self) -> tuple[int, list[tuple[int, int, str, str]]] | None:
        """Every process in the container as (pid, ppid, state, name), read from
        ``/proc/<pid>/stat`` (the image has no ``ps``, S7b.0), plus the pid of the measuring
        bash itself. ``PROC_LIST`` uses bash builtins only, so the measuring exec has no child.
        None if the list could not be read (exec failed or timed out)."""
        if not self.cid:
            return None
        try:
            r = _run([self.exe, "exec", self.cid, "bash", "-c", PROC_LIST], timeout=self.reaper_timeout_s)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return parse_proc_list(r.stdout) if r.returncode == 0 else None

    def alive_pids(self) -> list[int] | None:
        """Pids of processes a model command left running: everything except pid 1 (the
        keepalive) and the measuring bash (``$$``). Excluded by pid, not by name, so a model's
        own backgrounded ``bash`` still counts. Zombies are skipped (already dead; pid 1 reaps
        them). **None if the list could not be read**: callers must treat that as not clean."""
        if not self.cid:
            return []
        listed = self.processes()
        if listed is None:
            return None
        self_pid, procs = listed
        return [pid for pid, _ppid, state, _name in procs if pid not in (1, self_pid) and state != "Z"]


# Lists every process from /proc with bash builtins only (no ps in the image, and no child
# process of its own): "SELF=<its pid>", then "<pid> <ppid> <state> <name>" per process. In
# /proc/<pid>/stat the name sits between the first "(" and the LAST ")" and may contain spaces,
# ")" or a newline (prctl allows any byte), so the WHOLE file is read (`-d ''`: up to EOF, not
# the first newline), the name is cut out first, every control character in it becomes "?",
# and the numeric fields are read after it (S7b.0.7). A process that exits during the loop
# leaves an empty read and is skipped.
PROC_LIST = (
    'echo "SELF=$$"; '
    'for d in /proc/[0-9]*; do '
    "l=; IFS= read -r -d '' l < \"$d/stat\" 2>/dev/null; [ -n \"$l\" ] || continue; "
    'n=${l#*\\(}; n=${n%\\)*}; n=${n//[[:cntrl:]]/?}; r=${l##*\\) }; set -- $r; '
    'echo "${l%% *} $2 $1 $n"; '
    'done'
)


INTEGRITY_TAG = "LOCALBOT-CHECK"


def _sq(text: str) -> str:
    """A bash single-quoted word (any byte but NUL)."""
    return "'" + text.replace("'", "'\\''") + "'"


def scratch_script(dirs: tuple[str, ...] | list[str], keep: set[str] | None) -> str:
    """H1 P2: the scratch folders (``/tmp``, ``/var/tmp``, ``/run``, ``/dev/shm``,
    ``/dev/mqueue``) persist across execs, so a secret could be staged there in parts. With
    ``keep`` (the fresh container's own entries), delete every other entry, depth first, after
    giving the owner rwx (a chmod-000 folder); with ``keep=None`` delete nothing. Then print
    ``LIST=`` = base64 of every entry left, NUL-separated (names may hold any byte)."""
    d = " ".join(_sq(x) for x in dirs)
    listing = f'echo "LIST=$(find {d} -mindepth 1 -print0 2>/dev/null | base64 -w0)"'
    if keep is None:
        return listing
    k = " ".join(_sq(x) for x in sorted(keep))
    return ("declare -A K; for k in " + k + '; do K["$k"]=1; done; '
            f'for d in {d}; do [ -d "$d" ] || continue; '
            "while IFS= read -r -d '' p; do [ -n \"${K[\"$p\"]+x}\" ] && continue; "
            'chmod -R u+rwx -- "$p" 2>/dev/null; rm -rf -- "$p" 2>/dev/null; '
            "done < <(find \"$d\" -mindepth 1 -depth -print0 2>/dev/null); done; " + listing)


def parse_list(b64: str) -> set[str]:
    return {p.decode("utf-8", "surrogateescape") for p in _b64(b64).split(b"\0") if p}


def integrity_script(scratch_dirs: tuple[str, ...] = (), keep: set[str] | None = None) -> str:
    """The post-exec check's bash (H1). Prints its tag, then ``KEY=value`` lines; values that
    may hold any byte are base64. ``nullglob``: an unreadable ``/proc/1/task`` counts 0
    threads, never the literal pattern as 1."""
    return (f"echo {INTEGRITY_TAG}; shopt -s nullglob; "
            + scratch_script(scratch_dirs, keep) + "; "
            'set -- /proc/1/task/*; echo "TASKS=$#"; '
            'echo "CMD=$(base64 -w0 /proc/1/cmdline 2>/dev/null)"; '
            'echo "MOUNTS=$(base64 -w0 /proc/self/mountinfo 2>/dev/null)"')


def parse_mountinfo(raw: bytes) -> set[str]:
    """Mount points (field 5) of ``/proc/self/mountinfo``, octal escapes (``\\040``) undone."""
    out = set()
    for line in raw.split(b"\n"):
        fields = line.split(b" ")
        if len(fields) > 4:
            point = re.sub(rb"\\([0-7]{3})", lambda m: bytes([int(m.group(1), 8)]), fields[4])
            out.add(point.decode("utf-8", "surrogateescape"))
    return out


# --- secret masks (H1 decision 3) ---------------------------------------------------------------

MASK_NOTICE = "localbot: this secret file is hidden from the shell; the read tool can open it.\n"
SHORT_NAME_LETTERS = 4  # a name with at most this many letters gets all its case spellings (<= 16)
MAX_SPELLINGS_LONG = 4  # a longer name: exact, lower, upper, title


def secret_names_of(cfg: ContainerConfig) -> SecretNames:
    """The guard's matcher (H1 G-3) over this config's lists: one rule for masks and guard."""
    return SecretNames(globs=cfg.secret_globs, templates=cfg.secret_templates)


def is_secret_entry(name: str, cfg: ContainerConfig) -> bool:
    return secret_names_of(cfg).matches(name)


def secret_entries(root: str | os.PathLike[str], cfg: ContainerConfig) -> list[tuple[str, bool]]:
    """Every secret-named file or folder under ``root`` as ``(rel, is_dir)``, walked on the
    ``\\\\?\\`` form without following or listing any link; a secret folder is masked whole (not
    descended), ``.git`` is skipped (masked on its own). Refused-name folders are descended: the
    container sees inside them even though the host tools don't."""
    out: list[tuple[str, bool]] = []
    names = secret_names_of(cfg)

    def walk(folder: Path, prefix: str) -> None:
        try:
            with os.scandir(folder) as it:
                entries = list(it)
        except OSError:
            return
        for e in entries:
            rel = prefix + e.name
            try:
                st = os.lstat(e.path)
            except OSError:
                continue
            if getattr(st, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT or stat.S_ISLNK(st.st_mode):
                continue
            if is_git_name(rel):
                continue
            is_dir = stat.S_ISDIR(st.st_mode)
            if names.matches(e.name) and (is_dir or stat.S_ISREG(st.st_mode)):
                out.append((rel, is_dir))
            elif is_dir:
                walk(Path(e.path), rel + "/")

    walk(extended_path(root), "")
    return sorted(out)


def _spellings(name: str) -> set[str]:
    letters = [i for i, ch in enumerate(name) if ch.isalpha()]
    if len(letters) > SHORT_NAME_LETTERS:
        return {name, name.lower(), name.upper(), name.title()}
    out = set()
    for bits in range(1 << len(letters)):
        chars = list(name)
        for j, i in enumerate(letters):
            chars[i] = chars[i].upper() if (bits >> j) & 1 else chars[i].lower()
        out.add("".join(chars))
    return out


def mask_spellings(work: str | os.PathLike[str], rel: str, is_dir: bool) -> list[tuple[str, bool]]:
    """Other spellings of ``rel``'s last component that reach the same host entry (case, and
    the real 8.3 short name), each verified with ``samefile`` so a mask never creates an entry.
    Bounded (see ``_spellings``); a parent folder's spellings are not covered (as S8e.1)."""
    root = extended_path(work)
    real = root.joinpath(*rel.split("/"))
    parent = rel.rsplit("/", 1)[0] + "/" if "/" in rel else ""
    leaf = rel.rsplit("/", 1)[-1]
    names = _spellings(leaf)
    short = get_short_name(real)
    if short:
        short_leaf = os.path.basename(short.rstrip("\\/"))
        if short_leaf and short_leaf.casefold() != leaf.casefold():
            names |= _spellings(short_leaf)
    out = []
    for n in sorted(names - {leaf}):
        cand = root.joinpath(*(parent + n).split("/"))
        try:
            if os.path.lexists(cand) and os.path.samefile(cand, real):
                out.append((parent + n, is_dir))
        except OSError:
            continue
    return out


def _stat_key(work: Path, rel: str) -> tuple | None:
    try:
        st = os.lstat(extended_path(work).joinpath(*rel.split("/")))
    except OSError:
        return None
    return (st.st_ino, st.st_size, st.st_mtime_ns, st.st_mode)


def parse_integrity(text: str) -> dict[str, str] | None:
    """``KEY=value`` lines after the tag; None if the tag is missing (the check did not run)."""
    lines = text.split("\n")
    if INTEGRITY_TAG not in lines:
        return None
    out: dict[str, str] = {}
    for line in lines[lines.index(INTEGRITY_TAG) + 1:]:
        key, sep, val = line.partition("=")
        if sep and key.isupper():
            out[key] = val.strip()
    return out


def _b64(text: str) -> bytes:
    try:
        return base64.b64decode(text, validate=True)
    except ValueError:
        return b""


def parse_proc_list(text: str) -> tuple[int, list[tuple[int, int, str, str]]] | None:
    """Parse ``PROC_LIST`` output. None (= the list is unusable, so not clean) if the SELF line
    is missing or ANY other line does not parse: a process that can't be read is never just
    dropped (S7b.0.7: a name with a newline used to split into two lines and vanish)."""
    self_pid = None
    procs = []
    for line in text.split("\n"):  # not splitlines(): that also splits on \r, \x0b, \x1c, ...
        if not line:
            continue
        if line.startswith("SELF="):
            self_pid = int(line[5:].strip())
            continue
        parts = line.split(" ", 3)
        if len(parts) != 4 or not (parts[0].isdigit() and parts[1].isdigit()) or not parts[2].isalpha():
            return None
        procs.append((int(parts[0]), int(parts[1]), parts[2], parts[3]))
    return None if self_pid is None else (self_pid, procs)


def _run_capped(argv: list[str], timeout: float, limit: int, watch=None,
                poll_s: float = 2.0) -> tuple[str, bool]:
    """Run ``argv`` with stdout+stderr merged, keeping only the first ``limit`` characters in
    memory (the rest is read and dropped, so ``yes`` for 30 s cannot fill the host's RAM).
    Returns (capped text, timed_out). On timeout the client is killed; the reaper kills the
    in-container process. H1 S-4: while it runs, ``watch()`` is asked every ``poll_s``; True
    kills the client the same way (the caller then reads why from its own check)."""
    keep_bytes = limit * 4 + 4  # enough bytes for ``limit`` utf-8 characters
    kept = bytearray()
    dropped = [0]
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def pump() -> None:
        while True:
            chunk = proc.stdout.read1(65536)
            if not chunk:
                return
            room = keep_bytes - len(kept)
            if room > 0:
                kept.extend(chunk[:room])
            dropped[0] += max(0, len(chunk) - max(room, 0))

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    timed_out = False
    deadline = time.monotonic() + timeout
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            timed_out = True
            break
        try:
            proc.wait(timeout=min(left, poll_s) if watch else left)
            break
        except subprocess.TimeoutExpired:
            if watch is not None and time.monotonic() < deadline and watch():
                break
    if proc.poll() is None:
        proc.kill()
        proc.wait()
    reader.join(timeout=5)
    text = bytes(kept).decode("utf-8", errors="replace")
    if len(text) > limit or dropped[0]:
        extra = len(text[limit:].encode("utf-8")) + dropped[0]
        text = f"{text[:limit]}\n[truncated {extra} bytes]"
    return text, timed_out


def _run(argv: list[str], timeout: float, check: bool = False) -> subprocess.CompletedProcess:
    """Run a runtime command. No shell, so Git Bash never mangles the ``:/work`` mount; stdin
    closed so a command can't wait on input."""
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=check,
                          stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace")
