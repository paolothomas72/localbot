"""Run a long command (the test suite, the G2 pack) with a watchdog, so a stall never goes unseen.

An agent cannot interrupt itself while one shell call is waiting, so long runs go in the
background and the agent checks a plain status line between calls:

    python scripts/watch_run.py start suite -- <python> -m pytest -v -rs -p no:cacheprovider
    python scripts/watch_run.py wait suite          # blocks <= 90 s, returns early when done
    python scripts/watch_run.py status suite        # instant
    python scripts/watch_run.py stop suite          # kill the whole process tree + leftover report
    python scripts/watch_run.py run suite -- ...    # foreground (blocking), same watchdog

Every status ends in one verdict: RUNNING, STALLED, TIMED_OUT, FINISHED (exit code), STOPPED
(killed by the watchdog or by hand), BUSY (the run lock was taken, the command never ran) or DIED,
with the reason in plain words and the exact next command. FINISHED prints passed / failed /
skipped / errors from pytest's last line (the "=====" form and the -q form), and says so when an
exit 0 run skipped tests (G1 needs 0 skipped).

Logs (state/runs/, gitignored; LOCALBOT_RUNS_DIR overrides it): each run writes
<name>_<stamp>.log and <name>_<stamp>.json (stamp = UTC time, e.g. 20261031T190906Z) and never
overwrites an old one. <name>.json is the latest run's meta; <name>.log is a copy of the latest
run's log once it ends (finished, killed or stopped) and a one-line pointer to the live log while
it runs. A <name>.log / .json from before stamped logs is renamed to <name>_<stamp>, not lost.

Run lock (scripts/runlock.py, audit E-3): every start/run job takes the one machine-wide lock for
its whole life, container, GPU and host jobs alike (host tests add folders to workspaces/_trials,
which a running exam counts as host changes). A busy lock refuses at once: "busy: <holder>",
exit 3, and the command never runs; the refusal is recorded as its own stamped run, so status /
wait say BUSY (exit 3) instead of showing the previous run's verdict. The command gets
LOCALBOT_RUNLOCK_HELD=1. Under a parent that holds the lock (the GPU batch; the holder must be an
ancestor, see runlock.held_by_parent) run does not take it again, and start is refused (exit 2:
a detached job would outlive the parent's lock). --no-lock opts out: only for a job that touches
no container, no GPU, no model and no workspaces/ folder.

Processes are recorded with their create time. stop kills only while the run has no exit code,
and only processes that are still the recorded ones, so a pid reused by an unrelated process (or
an old meta after a reboot) is never killed; the same check decides DIED and "still going".

Stall = no new output for --stall seconds (default 300). Timeout = --timeout seconds (default
5400 = 90 min; the full suite outgrew the old 1800). Python standard library + psutil (already in
requirements-dev.txt).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    import psutil
except ImportError:  # status still works without it; stop/kill needs it
    psutil = None

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/, for runlock (also under python -I)
import runlock  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("LOCALBOT_RUNS_DIR") or (REPO / "state" / "runs"))
WIN = os.name == "nt"
TIMEOUT = 5400
SLACK = 2.0  # s: for a meta without create times, a run's process is created at most this long after its start
PROGRESS = re.compile(r"\[\s*\d+%\]")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
# pytest's last line, "===== 3 failed, 867 passed in 49.07s =====" or (-q) "3 failed, 867 passed in 49.07s"
FINAL = re.compile(r"^=*\s*(?P<items>\d+ [a-z]+(?:, \d+ [a-z]+)*|no tests ran) in [\d.]+s(?: \([\d:]+\))?\s*=*$")
ITEM = re.compile(r"(\d+) ([a-z]+)")
WORDS = {"passed": "passed", "failed": "failed", "skipped": "skipped", "error": "errors", "errors": "errors",
         "deselected": "deselected", "xfailed": "xfailed", "xpassed": "xpassed", "warning": "warnings",
         "warnings": "warnings", "rerun": "rerun"}
BASE = ("passed", "failed", "skipped", "errors", "deselected")


def paths(name: str) -> tuple[Path, Path]:
    """The latest-run files: <name>.json (meta) and <name>.log (copy of the latest log)."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,60}", name):
        sys.exit(f"bad run name {name!r}: letters, digits, _ . - only")
    RUNS.mkdir(parents=True, exist_ok=True)
    return RUNS / f"{name}.json", RUNS / f"{name}.log"


def live_log(name: str, d: dict) -> Path:
    """The log the run writes: its stamped log (recorded in the meta), else the old single log."""
    return Path(d["log"]) if d.get("log") else paths(name)[1]


def load(meta: Path) -> dict:
    try:
        return json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8", newline="\n")
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows: a status call is reading it this instant
            time.sleep(0.05)
    os.replace(tmp, path)


def save(meta: Path, data: dict) -> None:
    """Write <name>.json and, for a stamped run, its own <name>_<stamp>.json."""
    _write(meta, data)
    if data.get("meta"):
        _write(Path(data["meta"]), data)


def create_time(pid: int | None) -> float | None:
    """A process's create time, recorded next to its pid so the pid can be checked later."""
    if not pid or not psutil:
        return None
    try:
        return psutil.Process(pid).create_time()
    except psutil.Error:
        return None


def proc_of(pid: int | None, ct: float | None = None, started: float | None = None):
    """The live process a meta recorded as ``pid``, or None.

    A pid is reused once its process ends, and an old meta outlives a reboot, so a bare pid may name
    an unrelated process by now (llama-server, say). The process counts only if it is the one that
    was recorded: its create time equals the recorded one (``ct``), or, for a meta without it, it is
    not later than the run's start + SLACK. Without psutil nothing can be checked: None."""
    if not pid or not psutil:
        return None
    try:
        p = psutil.Process(pid)
        if not p.is_running() or p.status() == psutil.STATUS_ZOMBIE:
            return None
        made = p.create_time()
    except psutil.Error:
        return None
    if ct is not None:
        return p if abs(made - ct) <= 0.5 else None
    if started is not None:
        return p if made <= started + SLACK else None
    return None


def alive(pid: int | None, ct: float | None = None, started: float | None = None) -> bool:
    if psutil:
        return proc_of(pid, ct, started) is not None
    if not pid:
        return False
    if WIN:
        return True  # cannot tell without psutil (and os.kill(pid, 0) would terminate it on Windows)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_tree(target) -> list[int]:
    """Kill a process and all its children. ``target`` is a psutil.Process from proc_of(), or the pid
    of this script's own child (a Popen not waited yet, so that pid cannot have been reused). Never
    a bare pid read back from a meta."""
    if not target or not psutil:
        return []
    try:
        root = target if isinstance(target, psutil.Process) else psutil.Process(target)
        if not root.is_running():  # psutil compares the create time: False once the pid is reused
            return []
        procs = root.children(recursive=True) + [root]
    except psutil.Error:
        return []
    for p in procs:
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(procs, timeout=10)
    return [p.pid for p in procs]


def fmt(sec: float) -> str:
    sec = int(max(0, sec))
    return f"{sec // 60}m{sec % 60:02d}s"


def tail(log: Path, n: int = 4000) -> list[str]:
    try:
        with open(log, "rb") as f:
            f.seek(max(0, log.stat().st_size - n))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    return [ln.rstrip("\r") for ln in text.split("\n")]


def parse_counts(line: str) -> dict | None:
    """Counts from pytest's final summary line, or None if ``line`` is not one. passed, failed,
    skipped, errors and deselected are always there (0 if absent); xfailed etc. only when shown."""
    m = FINAL.match(ANSI.sub("", line).strip())
    if not m:
        return None
    counts = dict.fromkeys(BASE, 0)
    if m["items"] == "no tests ran":
        return counts
    known = False
    for n, word in ITEM.findall(m["items"]):
        key = WORDS.get(word)
        if key:
            counts[key] = counts.get(key, 0) + int(n)
            known = True
    return counts if known else None


def find_summary(lines: list[str]) -> tuple[str | None, dict | None]:
    for ln in reversed(lines):
        c = parse_counts(ln)
        if c is not None:
            return ANSI.sub("", ln).strip(), c
    return None, None


def counts_text(c: dict) -> str:
    out = [f"{c['passed']} passed", f"{c['failed']} failed", f"{c['skipped']} skipped", f"{c['errors']} errors"]
    out += [f"{c[k]} {k}" for k in ("deselected", "xfailed", "xpassed") if c.get(k)]
    return ", ".join(out)


def leftovers() -> str:
    """Containers / trial folders a killed test run may have left behind (fixtures never ran)."""
    out = []
    for exe in ("podman", "docker"):
        try:
            r = subprocess.run([exe, "ps", "-aq", "--filter", "label=localbot-container"],
                               capture_output=True, text=True, timeout=60)
            ids = r.stdout.split() if r.returncode == 0 else None
            out.append(f"{exe} labeled containers: {len(ids) if ids is not None else 'unknown (runtime not answering)'}")
        except (OSError, subprocess.TimeoutExpired):
            out.append(f"{exe}: not available")
    trials = REPO / "workspaces" / "_trials"
    n = len(list(trials.iterdir())) if trials.is_dir() else 0
    out.append(f"workspaces/_trials entries: {n}")
    return "; ".join(out)


def verdict(name: str, stall: float | None = None, timeout: float | None = None) -> tuple[str, str]:
    meta, latest = paths(name)
    d = load(meta)
    if not d:
        return "UNKNOWN", f"no run named {name!r} (nothing in {meta}). Start it with: start {name} -- <command>"
    log = live_log(name, d)
    stall = stall or d.get("stall", 300)
    timeout = timeout or d.get("timeout", TIMEOUT)
    now = time.time()
    elapsed = (d.get("ended") or now) - d["started"]
    try:
        last_out = log.stat().st_mtime
    except OSError:
        last_out = d["started"]
    quiet = now - last_out
    lines = [ln for ln in tail(log) if ln.strip()]
    summary, counts = find_summary(lines)
    prog = next((m.group(0) for ln in reversed(lines) for m in [PROGRESS.search(ln)] if m), "")
    last = lines[-1][:200] if lines else "(no output yet)"
    head = f"run {name}: elapsed {fmt(elapsed)}, last output {fmt(quiet)} ago {prog}".rstrip()
    where = f"  log: {log}" + (f"  (copy: {latest})" if log != latest else "")

    if "exit_code" in d:
        code = d["exit_code"]
        why = d.get("killed_reason")
        if why and why.startswith("busy"):
            return "BUSY", (f"{head}\n  not started: the run lock was {why}\n  the command never ran.\n"
                            f"  next: wait for that job to end (python scripts/runlock.py status says FREE), "
                            f"then start {name} again")
        if why:
            return "STOPPED", f"{head}\n  killed by the watchdog: {why}\n  last line: {last}\n{where}"
        plain = "all passed" if code == 0 else ("tests FAILED" if code == 1 else f"exit code {code}")
        if code == 0 and counts and counts["skipped"]:
            plain += f", but {counts['skipped']} SKIPPED (G1 needs 0 skipped)"
        if code == 0 and not counts:
            plain += " (no pytest summary line in the log, so no counts)"
        res = summary or last
        cnt = f"  counts: {counts_text(counts)}\n" if counts else ""
        return "FINISHED", f"{head}\n  exit {code} = {plain}\n{cnt}  result: {res}\n{where}"
    if not alive(d.get("pid"), d.get("pid_ct"), d.get("started")):
        return "DIED", (f"{head}\n  the run's process is gone but never wrote an exit code: it was killed from "
                        f"outside (e.g. the agent's shell call ended and took it along) or crashed.\n"
                        f"  last line: {last}\n  next: check leftovers (stop {name}), then use the foreground "
                        f"fallback: run {name} -- <same command>")
    if elapsed > timeout:
        return "TIMED_OUT", (f"{head}\n  over the {fmt(timeout)} limit and still running.\n  last line: {last}\n"
                             f"  next: stop {name}, then re-run only the file named in the last line")
    if quiet > stall:
        return "STALLED", (f"{head}\n  alive, but no new output for {fmt(quiet)} (limit {fmt(stall)}): most likely a hung "
                           f"test or a container that stopped answering.\n  stuck at: {last}\n"
                           f"  next: stop {name}, check leftovers, re-run that one test file once; if it hangs "
                           f"again, report it to the owner instead of retrying")
    return "RUNNING", f"{head}\n  last line: {last}"


def lock_mode(no_lock: bool) -> str:
    """take = this job takes the run lock; parent = a parent job holds it; off = --no-lock."""
    if no_lock:
        return "off"
    return "parent" if runlock.held_by_parent() else "take"


def child_env(mode: str) -> dict:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"  # without this, output to a file is buffered and looks like a stall
    env["PYTHONIOENCODING"] = "utf-8"
    if mode in ("take", "parent"):
        env[runlock.HELD_ENV] = "1"  # the command (pytest, g2, ...) must not take the lock again
    else:
        env.pop(runlock.HELD_ENV, None)  # --no-lock: the command must not think it is covered
    return env


def finish(name: str, d: dict) -> None:
    """The run is over: counts into the meta, the stamped log copied to <name>.log, meta saved."""
    meta, latest = paths(name)
    log = live_log(name, d)
    summary, counts = find_summary([ln for ln in tail(log) if ln.strip()])
    if counts:
        d.update(counts=counts, result=summary)
    if log != latest:
        try:
            shutil.copyfile(log, latest)
        except OSError:
            pass
    save(meta, d)


def supervise(name: str, cmd: list[str], stall: float, timeout: float, heartbeat: float | None,
              mode: str = "take") -> int:
    """Run cmd with output to the log; kill it on stall/timeout. Records the exit code."""
    meta, _ = paths(name)
    d = load(meta)
    killed = None
    with open(live_log(name, d), "ab") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                env=child_env(mode), cwd=d.get("cwd") or os.getcwd())
        d = load(meta)  # fresh: `start` may have recorded the supervisor's pid meanwhile
        d.update(cmd_pid=proc.pid, cmd_ct=create_time(proc.pid))
        save(meta, d)
        last_beat = time.time()
        while True:
            try:
                code = proc.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                pass
            v, msg = verdict(name, stall, timeout)
            if v in ("STALLED", "TIMED_OUT"):
                if not kill_tree(proc.pid):
                    proc.kill()
                proc.wait()
                code, killed = -1, (v, msg)
                break
            if heartbeat and time.time() - last_beat >= heartbeat:
                print(f"[watchdog] {v} — {msg.splitlines()[0]}", flush=True)
                last_beat = time.time()
    d = load(meta)
    d.update(exit_code=code, ended=time.time())
    if killed:
        v, msg = killed
        d.update(killed_reason=f"{v}: {msg.splitlines()[1].strip()}")
    finish(name, d)
    if killed and heartbeat:
        print(f"[watchdog] {v}\n{msg}\n[watchdog] leftovers: {leftovers()}", flush=True)
    return code


def not_started(name: str, holder: dict | None) -> int:
    """The detached supervisor lost the race for the lock: record it, never run the command."""
    meta, _ = paths(name)
    d = load(meta)
    why = f"busy: {runlock.describe(holder)}"
    with open(live_log(name, d), "a", encoding="utf-8", newline="\n") as f:
        f.write(f"[watchdog] not started: the run lock is {why}\n")
    d.update(exit_code=runlock.BUSY_EXIT, ended=time.time(), killed_reason=why)
    finish(name, d)
    return runlock.BUSY_EXIT


def _stamp(t: float) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(t))


def _free_stamp(name: str, base: str) -> str:
    stamp, i = base, 1
    while (RUNS / f"{name}_{stamp}.log").exists() or (RUNS / f"{name}_{stamp}.json").exists():
        i += 1
        stamp = f"{base}-{i}"
    return stamp


def keep_legacy(name: str, old: dict) -> None:
    """<name>.log / .json from before stamped logs are the only copy of that run: rename them to
    <name>_<stamp>.log / .json instead of overwriting them."""
    if old.get("stamp"):
        return  # <name>.log is only a copy of a stamped log
    meta, log = paths(name)
    files = [p for p in (log, meta) if p.exists()]
    if not files:
        return
    stamp = _free_stamp(name, _stamp(old.get("started") or files[0].stat().st_mtime))
    for p in files:
        os.replace(p, RUNS / f"{name}_{stamp}{p.suffix}")


def still_going(old: dict) -> bool:
    """The meta's run has no exit code and its recorded process is alive and really that process."""
    return bool(old) and "exit_code" not in old and alive(old.get("pid"), old.get("pid_ct"), old.get("started"))


def new_run(name: str, cmd: list[str], stall: float, timeout: float, mode: str = "take") -> None:
    meta, latest = paths(name)
    old = load(meta)
    if still_going(old):
        sys.exit(f"run {name!r} is still going (pid {old['pid']}). status {name} / stop {name} first.")
    keep_legacy(name, old)
    now = time.time()
    stamp = _free_stamp(name, _stamp(now))
    log, smeta = RUNS / f"{name}_{stamp}.log", RUNS / f"{name}_{stamp}.json"
    log.write_bytes(b"")
    latest.write_text(f"(run {name} in progress: live log {log})\n", encoding="utf-8", newline="\n")
    save(meta, {"name": name, "cmd": cmd, "cwd": os.getcwd(), "started": now, "stall": stall,
                "timeout": timeout, "pid": os.getpid(), "pid_ct": create_time(os.getpid()), "stamp": stamp,
                "log": str(log), "meta": str(smeta), "lock": mode})


def record_busy(name: str, cmd: list[str], stall: float, timeout: float, holder: dict | None) -> None:
    """A refused start/run is recorded as its own stamped run (exit 3, "busy: ..."), so status / wait
    say BUSY, the command never ran, instead of showing the previous run's verdict. The previous
    run's stamped files are untouched. Not while a run of this name is still going: <name>.json
    keeps showing that run."""
    if still_going(load(paths(name)[0])):
        print(f"  (run {name!r} is still going; status {name} shows that run)")
        return
    new_run(name, cmd, stall, timeout, "take")
    not_started(name, holder)


def rc(name: str, v: str) -> int:
    """0 = still running or finished with every test passing; 3 = busy; 1 = anything else."""
    if v == "RUNNING":
        return 0
    if v == "BUSY":
        return runlock.BUSY_EXIT
    return 0 if v == "FINISHED" and load(paths(name)[0]).get("exit_code") == 0 else 1


def busy_line(holder: dict | None) -> str:
    return f"busy: {runlock.describe(holder)}\n  the command did not run. python scripts/runlock.py status shows the holder"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["start", "run", "status", "wait", "stop", "_child"])
    ap.add_argument("name")
    ap.add_argument("--stall", type=float, default=300, help="seconds without output = STALLED (default 300)")
    ap.add_argument("--timeout", type=float, default=TIMEOUT, help=f"seconds in all = TIMED_OUT (default {TIMEOUT})")
    ap.add_argument("--max", type=float, default=90, help="wait: longest block in seconds")
    ap.add_argument("--no-lock", action="store_true",
                    help="start/run without the run lock: only for a job that touches no container, GPU, "
                         "model or workspaces/ folder")
    ap.add_argument("--lock-mode", choices=["take", "parent", "off"], default="take", help=argparse.SUPPRESS)
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    a = ap.parse_args(argv[:cut])  # options before "--", the command after it, untouched
    cmd = argv[cut + 1:]

    if a.action in ("start", "run") and not cmd:
        sys.exit("give the command after --")
    if a.action == "start":
        mode = lock_mode(a.no_lock)
        if mode == "parent":
            print("start refused: this runs under a job that holds the run lock (the GPU batch, runlock.py "
                  "run ...). A detached job would outlive that lock, and its command could not show that it "
                  f"is covered (its starter leaves its ancestry). Inside such a job use run (foreground): "
                  f"watch_run.py run {a.name} -- ...", flush=True)
            return 2
        if mode == "take" and runlock.is_held():  # refuse at once; the supervisor takes it for real
            h = runlock.holder()
            print(busy_line(h), flush=True)
            record_busy(a.name, cmd, a.stall, a.timeout, h)
            return runlock.BUSY_EXIT
        new_run(a.name, cmd, a.stall, a.timeout, mode)
        meta, log = paths(a.name)
        args = [sys.executable, str(Path(__file__).resolve()), "_child", a.name, "--stall", str(a.stall),
                "--timeout", str(a.timeout), "--lock-mode", mode, "--", *cmd]
        kw = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
        if WIN:
            base = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            try:  # leave the agent's job object so the run survives the shell call that started it
                p = subprocess.Popen(args, creationflags=base | subprocess.CREATE_BREAKAWAY_FROM_JOB, **kw)
            except OSError:
                p = subprocess.Popen(args, creationflags=base, **kw)
        else:
            p = subprocess.Popen(args, start_new_session=True, **kw)
        d = load(meta)
        d.update(pid=p.pid, pid_ct=create_time(p.pid))
        save(meta, d)
        time.sleep(3)
        v, msg = verdict(a.name)
        print(f"{v}\n{msg}\nnext: wait {a.name}   (repeat until FINISHED; each call blocks <= {int(a.max)} s)")
        if v == "BUSY":
            return runlock.BUSY_EXIT
        return 0 if v in ("RUNNING", "FINISHED") else 1
    if a.action == "_child":
        lk = None
        if a.lock_mode == "take":  # the detached supervisor holds the lock for the job's whole life
            try:
                lk = runlock.acquire(a.name, cmd=cmd)
            except runlock.LockBusy as e:
                return not_started(a.name, e.holder)
        try:
            return supervise(a.name, cmd, a.stall, a.timeout, heartbeat=None, mode=a.lock_mode)
        finally:
            if lk:
                lk.release()
    if a.action == "run":
        mode = lock_mode(a.no_lock)
        lk = None
        if mode == "take":  # the foreground process holds the lock while the command runs
            try:
                lk = runlock.acquire(a.name, cmd=cmd)
            except runlock.LockBusy as e:
                print(busy_line(e.holder), flush=True)
                record_busy(a.name, cmd, a.stall, a.timeout, e.holder)
                return runlock.BUSY_EXIT
        try:
            new_run(a.name, cmd, a.stall, a.timeout, mode)
            lock = {"take": "run lock held", "parent": "run lock held by the parent job",
                    "off": "NO run lock (--no-lock)"}[mode]
            print(f"[watchdog] {a.name}: foreground, stall {fmt(a.stall)}, timeout {fmt(a.timeout)}, {lock}, "
                  f"log {live_log(a.name, load(paths(a.name)[0]))}", flush=True)
            code = supervise(a.name, cmd, a.stall, a.timeout, heartbeat=60, mode=mode)
        finally:
            if lk:
                lk.release()
        v, msg = verdict(a.name)
        print(f"{v}\n{msg}", flush=True)
        return code
    if a.action == "wait":
        end = time.time() + a.max
        while True:
            v, msg = verdict(a.name)
            if v != "RUNNING" or time.time() >= end:
                break
            time.sleep(5)
        print(f"{v}\n{msg}")
        if v == "RUNNING":
            print(f"next: wait {a.name}")
        return rc(a.name, v)
    if a.action == "status":
        v, msg = verdict(a.name)
        print(f"{v}\n{msg}")
        return rc(a.name, v)
    if a.action == "stop":
        meta, _ = paths(a.name)
        d = load(meta)
        killed = []
        if d and "exit_code" not in d:
            # Only processes that are still the ones this run recorded (pid + create time): a pid
            # reused by an unrelated process (llama-server on :8090, say) is never touched.
            for key, ct in (("pid", "pid_ct"), ("cmd_pid", "cmd_ct")):
                p = proc_of(d.get(key), d.get(ct), d.get("started"))
                if p:
                    killed += kill_tree(p)
            d.update(exit_code=-1, ended=time.time(), killed_reason="stopped by hand (stop)")
            finish(a.name, d)  # copy the stamped log to <name>.log (also after DIED)
        elif d:
            print(f"run {a.name} already ended (exit {d['exit_code']}): nothing to kill, its record is left as it is")
        print(f"stopped {a.name}: killed {len(set(killed))} process(es)")
        print(f"leftovers: {leftovers()}")
        print("if any labeled containers or _trials entries are left: remove them before the next run "
              "(they are test leftovers, never the user's)")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
