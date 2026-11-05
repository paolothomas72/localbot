"""``python -m localbot undo <bot> [steps]`` (docs/STAGE1_SPEC.md §5.7, S7b.1).

Restores a bot's workspace to the snapshot taken before its ``steps``-th last bash/write/edit
(default 1). ``--list`` shows the snapshots instead. A restore runs only when no Local Bot
container is running on podman or docker (a runtime that is not running has none); if that
can't be confirmed, it refuses.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from typing import Callable

from localbot.config import ContainerConfig, UndoConfig
from localbot.guard.undo import UndoError, UndoStore

# What the CLIs print when their engine isn't running: then no container of ours can be running.
_ENGINE_DOWN = ("cannot connect", "error during connect", "unable to connect", "is the docker daemon running")


def running_localbot_containers(runtimes: tuple[str, ...] = ("podman", "docker"),
                                run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> list[str]:
    """Ids of running containers with our label key, on every installed runtime. Raises
    UndoError if a runtime's answer can't be read (fail closed)."""
    key = ContainerConfig().label_key
    found: list[str] = []
    for exe in runtimes:
        if not shutil.which(exe):
            continue
        try:
            r = run([exe, "ps", "-q", "--filter", f"label={key}"], capture_output=True, text=True,
                    timeout=60, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise UndoError(f"could not ask {exe} for running containers: {exc}") from exc
        if r.returncode != 0:
            if any(s in (r.stderr or "").lower() for s in _ENGINE_DOWN):
                continue
            raise UndoError(f"could not ask {exe} for running containers: {(r.stderr or '').strip()[:200]}")
        found += [f"{exe}:{cid}" for cid in r.stdout.split()]
    return found


def main(argv: list[str] | None = None, *, quiet: Callable[[], bool] | None = None,
         cfg: UndoConfig | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m localbot")
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("undo", help="restore a bot's workspace to an earlier snapshot")
    u.add_argument("bot")
    u.add_argument("steps", nargs="?", type=int, default=1)
    u.add_argument("--list", action="store_true", help="list the snapshots, change nothing")
    a = ap.parse_args(argv)
    cfg = cfg or UndoConfig()
    try:
        store = UndoStore.open_existing(cfg, a.bot)
        if a.list:
            for seq in store.seqs():
                s = store.load(seq)
                print(f"{seq:6d}  before {s.reason or '?':5}  files={len(s.files)} dirs={len(s.dirs)} "
                      f"not_stored={len(s.skipped) + len(s.skipped_large)}")
            return 0
        report = store.restore(a.steps, quiet=quiet or (lambda: running_localbot_containers() == []))
    except UndoError as exc:
        print(f"undo: {exc}", file=sys.stderr)
        return 2
    print(f"undo {a.bot}: restored snapshot {report['restored_to']} ({a.steps} step(s)), verified; "
          f"removed {len(report['removed'])}, folders made {len(report['dirs_made'])}, "
          f"files written {len(report['written'])}")
    for rel in report["not_restored_large"]:
        print(f"  not restored (was over the size limit, never stored): {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
