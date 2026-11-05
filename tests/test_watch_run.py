"""H1 Track T (audit E-3, E-4, E-6): scripts/watch_run.py stamps its logs, counts the results and
takes the run lock.

- E-6: each run writes <name>_<stamp>.log and <name>_<stamp>.json and never overwrites an old one;
  <name>.json is the latest run's meta and <name>.log a copy of the latest run's log once it ends.
  The default timeout is 5400 s (the suite outgrew 1800 s and was killed at 72%).
- E-4: the FINISHED verdict prints passed / failed / skipped / errors from pytest's last line (the
  "=====" form and the -q form); exit 0 with skipped tests says so, because G1 needs 0 skipped.
- E-3: start/run take the run lock (busy: exit 3, the command never runs); --no-lock opts out; the
  command sees LOCALBOT_RUNLOCK_HELD=1; under a parent that holds the lock it is not taken again.

The commands are tiny python one-liners that print a fake pytest summary. LOCALBOT_RUNS_DIR and
LOCALBOT_RUNLOCK_PATH point at tmp_path, and PATH holds only python's own folder, so the leftover
check (podman/docker ps) can never reach a runtime. No container, no model, no GPU.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "watch_run.py"
RUNLOCK = REPO / "scripts" / "runlock.py"
HELD = "LOCALBOT_RUNLOCK_HELD"
PY = sys.executable
STAMPED = re.compile(r"^(?P<name>.+)_(?P<stamp>\d{8}T\d{6}Z(?:-\d+)?)\.(?P<ext>log|json)$")


@pytest.fixture(scope="module")
def rl():
    spec = importlib.util.spec_from_file_location("runlock_h1w", RUNLOCK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def box(tmp_path, monkeypatch):
    runs, lock = tmp_path / "runs", tmp_path / "state" / "run.lock"
    monkeypatch.setenv("LOCALBOT_RUNS_DIR", str(runs))
    monkeypatch.setenv("LOCALBOT_RUNLOCK_PATH", str(lock))
    monkeypatch.delenv(HELD, raising=False)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PATH=str(Path(PY).parent),
               NoDefaultCurrentDirectoryInExePath="1")
    env.pop(HELD, None)

    class Box:
        pass

    b = Box()
    b.runs, b.lock, b.env, b.cwd = runs, lock, env, tmp_path
    return b


def wr(box, *argv: str, **extra: str) -> subprocess.CompletedProcess:
    env = dict(box.env, **extra)
    return subprocess.run([PY, str(SCRIPT), *argv], capture_output=True, text=True, encoding="utf-8",
                          env=env, cwd=box.cwd, timeout=180)


def say(*lines: str, code: int = 0) -> list[str]:
    body = "; ".join(f"print({ln!r})" for ln in lines)
    return [PY, "-c", f"import sys; {body}; sys.exit({code})"]


def stamped(box, name: str, ext: str) -> list[Path]:
    out = []
    for p in sorted(box.runs.glob(f"{name}_*.{ext}")):
        m = STAMPED.match(p.name)
        if m and m["name"] == name:
            out.append(p)
    return out


def _kill_tree(root) -> None:
    """Kill ``root`` and its children. ``root`` is the pid of our own Popen (not waited yet, so the
    pid cannot be reused) or a psutil.Process captured while it was known to be the job's: a pid read
    back from a meta later may belong to another process by then (F8), so it is never used bare."""
    try:
        if not isinstance(root, psutil.Process):
            root = psutil.Process(root)
        if not root.is_running():  # psutil compares the create time: False once the pid is reused
            return
        procs = root.children(recursive=True) + [root]
    except psutil.Error:
        return
    for p in procs:
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(procs, timeout=15)


def _job_procs(box, name: str) -> list:
    """The job's supervisor and command as psutil.Process objects, captured now, while the meta's
    recorded create times prove they are the job's own processes. Waits (<= 15 s) until the
    supervisor has recorded its command, or the job has ended."""
    meta = box.runs / f"{name}.json"

    def read():
        try:
            return json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    _until(lambda: "cmd_ct" in read() or "exit_code" in read(), 15)
    d = read()
    out = []
    for key, ct in (("pid", "pid_ct"), ("cmd_pid", "cmd_ct")):
        try:
            p = psutil.Process(d[key])
            if d.get(ct) is not None and abs(p.create_time() - d[ct]) <= 0.5:
                out.append(p)
        except (KeyError, TypeError, psutil.Error):
            pass
    return out


class Victim:
    """An unrelated process (a sleeper) that a stale meta points at; it must survive."""

    def __init__(self):
        self.proc = subprocess.Popen([PY, "-c", "import time; time.sleep(120)"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.pid = self.proc.pid
        self.ct = psutil.Process(self.pid).create_time()

    def alive(self) -> bool:
        return self.proc.poll() is None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        _kill_tree(self.proc.pid)
        self.proc.wait(timeout=30)


def _meta(box, name: str, **d) -> None:
    box.runs.mkdir(parents=True, exist_ok=True)
    (box.runs / f"{name}.json").write_text(json.dumps(dict({"name": name, "cmd": ["x"], "stall": 300,
                                                            "timeout": 5400}, **d)), encoding="utf-8")


def _until(pred, timeout: float = 30.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.1)
    return pred()


class Holder:
    def __init__(self, rl, box, name: str = "holder"):
        self.proc = subprocess.Popen([PY, str(RUNLOCK), "run", name, "--", PY, "-c", "import time; time.sleep(90)"],
                                     env=box.env, cwd=box.cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not _until(lambda: (rl.holder(box.lock) or {}).get("name") == name and rl.is_held(box.lock)):
            self.kill()
            pytest.fail("the holder process never took the lock")
        self.pid = rl.holder(box.lock)["pid"]

    def kill(self):
        if self.proc.returncode is not None:  # already killed and waited: the pid may be reused now
            return
        _kill_tree(self.proc.pid)
        self.proc.wait(timeout=30)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.kill()


# --- counts in the verdict (E-4) ------------------------------------------------------------


@pytest.mark.parametrize("line, code, counts", [
    ("===== 3 failed, 867 passed in 49.07s =====", 1, "867 passed, 3 failed, 0 skipped, 0 errors"),
    ("3 failed, 867 passed in 49.07s", 1, "867 passed, 3 failed, 0 skipped, 0 errors"),
    ("== 1 passed in 0.01s ==", 0, "1 passed, 0 failed, 0 skipped, 0 errors"),
    ("=== 2 failed, 10 passed, 1 skipped, 4 deselected, 3 errors in 1625.32s (0:27:05) ===", 1,
     "10 passed, 2 failed, 1 skipped, 3 errors, 4 deselected"),
    ("1 failed, 1 error in 0.50s", 1, "0 passed, 1 failed, 0 skipped, 1 errors"),
])
def test_finished_verdict_shows_the_counts(box, line, code, counts):
    r = wr(box, "run", "cnt", "--", *say("collected 870 items", "tests/test_x.py ....  [100%]", line, code=code))
    assert r.returncode == code, r.stdout + r.stderr
    assert "FINISHED" in r.stdout
    assert f"counts: {counts}" in r.stdout, r.stdout
    assert f"result: {line}" in r.stdout
    meta = json.loads((box.runs / "cnt.json").read_text(encoding="utf-8"))
    assert meta["exit_code"] == code and meta["result"] == line
    assert "SKIPPED" not in r.stdout


@pytest.mark.parametrize("line", ["===== 867 passed, 2 skipped in 49.07s =====", "867 passed, 2 skipped in 49.07s"])
def test_exit_0_with_skips_is_flagged(box, line):
    r = wr(box, "run", "sk", "--", *say(line))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "exit 0 = all passed, but 2 SKIPPED (G1 needs 0 skipped)" in r.stdout, r.stdout
    assert "counts: 867 passed, 0 failed, 2 skipped, 0 errors" in r.stdout
    meta = json.loads((box.runs / "sk.json").read_text(encoding="utf-8"))
    assert meta["counts"]["skipped"] == 2 and meta["counts"]["passed"] == 867


def test_no_summary_line_says_so(box):
    r = wr(box, "run", "plain", "--", *say("[podman] pass=True -> results/stage1/g2_podman_x.json"))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "no pytest summary line" in r.stdout


def test_parse_counts_unit():
    spec = importlib.util.spec_from_file_location("watch_run_h1", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.parse_counts("= 5 passed, 1 xfailed, 2 warnings in 3.00s =") == {
        "passed": 5, "failed": 0, "skipped": 0, "errors": 0, "deselected": 0, "xfailed": 1, "warnings": 2}
    assert mod.parse_counts("no tests ran in 0.01s") == {
        "passed": 0, "failed": 0, "skipped": 0, "errors": 0, "deselected": 0}
    assert mod.parse_counts("\x1b[32m===== 4 passed in 1.2s =====\x1b[0m")["passed"] == 4
    assert mod.parse_counts("870 tests collected in 2.10s") is None
    assert mod.parse_counts("tests/test_x.py::test_a PASSED  [ 50%]") is None
    assert mod.parse_counts("3 passed") is None  # not a final line: no "in Ns"


# --- stamped logs, default timeout (E-6) ---------------------------------------------------


def test_default_timeout_is_5400(box):
    r = wr(box, "run", "demo", "--", *say("== 1 passed in 0.01s =="))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "timeout 90m00s" in r.stdout
    assert json.loads((box.runs / "demo.json").read_text(encoding="utf-8"))["timeout"] == 5400
    h = subprocess.run([PY, str(SCRIPT), "--help"], capture_output=True, text=True, encoding="utf-8",
                       env=box.env, timeout=60)
    assert "5400" in h.stdout and "--no-lock" in h.stdout


def test_stamped_log_meta_and_latest_copy(box):
    r = wr(box, "run", "demo", "--", *say("first run", "== 1 passed in 0.01s =="))
    assert r.returncode == 0, r.stdout + r.stderr
    logs, metas = stamped(box, "demo", "log"), stamped(box, "demo", "json")
    assert len(logs) == 1 and len(metas) == 1
    assert STAMPED.match(logs[0].name)["stamp"] == STAMPED.match(metas[0].name)["stamp"]
    latest = box.runs / "demo.log"
    assert latest.read_bytes() == logs[0].read_bytes() and b"first run" in latest.read_bytes()
    meta = json.loads((box.runs / "demo.json").read_text(encoding="utf-8"))
    assert meta == json.loads(metas[0].read_text(encoding="utf-8"))
    assert Path(meta["log"]) == logs[0] and meta["exit_code"] == 0
    assert str(logs[0]) in r.stdout  # the verdict names the stamped log


def test_second_run_keeps_the_first_stamped_log(box):
    assert wr(box, "run", "demo", "--", *say("run one", "== 1 passed in 0.01s ==")).returncode == 0
    assert wr(box, "run", "demo", "--", *say("run two", "== 1 failed in 0.01s ==", code=1)).returncode == 1
    logs = stamped(box, "demo", "log")
    assert len(logs) == 2 and len(stamped(box, "demo", "json")) == 2
    texts = sorted(p.read_text(encoding="utf-8") for p in logs)
    assert any("run one" in t for t in texts) and any("run two" in t for t in texts)
    assert "run two" in (box.runs / "demo.log").read_text(encoding="utf-8")
    meta = json.loads((box.runs / "demo.json").read_text(encoding="utf-8"))
    assert meta["exit_code"] == 1 and "run two" in Path(meta["log"]).read_text(encoding="utf-8")
    first = next(p for p in logs if "run one" in p.read_text(encoding="utf-8"))
    first_meta = json.loads(first.with_suffix(".json").read_text(encoding="utf-8"))
    assert first_meta["exit_code"] == 0 and Path(first_meta["log"]) == first


def test_a_legacy_unstamped_log_is_kept(box):
    box.runs.mkdir(parents=True)
    (box.runs / "suite.log").write_text("old green log\n", encoding="utf-8")
    (box.runs / "suite.json").write_text(json.dumps({"name": "suite", "started": 1761935941.0, "exit_code": 0,
                                                     "pid": 1}), encoding="utf-8")
    assert wr(box, "run", "suite", "--", *say("== 1 passed in 0.01s ==")).returncode == 0
    kept = [p for p in stamped(box, "suite", "log") if p.read_text(encoding="utf-8") == "old green log\n"]
    assert len(kept) == 1
    assert any(json.loads(p.read_text(encoding="utf-8")).get("pid") == 1 for p in stamped(box, "suite", "json"))


def test_start_holds_the_lock_and_status_reads_the_live_log(rl, box):
    cmd = [PY, "-c", "import time; print('working', flush=True); time.sleep(6); print('== 2 passed in 6.00s ==')"]
    r = wr(box, "start", "bg", "--", *cmd)
    procs = _job_procs(box, "bg")
    try:
        assert r.returncode == 0, r.stdout + r.stderr
        assert "RUNNING" in r.stdout
        assert procs, "no supervisor process recorded with its create time"
        assert rl.is_held(box.lock) and rl.holder(box.lock)["name"] == "bg"
        assert "in progress" in (box.runs / "bg.log").read_text(encoding="utf-8")  # pointer while running
        st = wr(box, "status", "bg")
        assert "RUNNING" in st.stdout and "working" in st.stdout, st.stdout
        busy = wr(box, "start", "other", "--", *say("never"))
        assert busy.returncode == 3 and "busy:" in busy.stdout, busy.stdout + busy.stderr
        assert json.loads((box.runs / "other.json").read_text(encoding="utf-8"))["exit_code"] == 3  # F4
        # The same name again while it runs: refused, and bg.json still shows the running job.
        again = wr(box, "start", "bg", "--", *say("never"))
        assert again.returncode == 3 and "busy:" in again.stdout, again.stdout + again.stderr
        assert "RUNNING" in wr(box, "status", "bg").stdout
        w = wr(box, "wait", "bg", "--max", "60")
        assert "FINISHED" in w.stdout and "counts: 2 passed, 0 failed, 0 skipped, 0 errors" in w.stdout, w.stdout
    finally:
        for p in procs:
            _kill_tree(p)  # captured while they were the job's: a reused pid is skipped (F8)
    assert _until(lambda: not rl.is_held(box.lock), 15)
    assert "2 passed" in (box.runs / "bg.log").read_text(encoding="utf-8")


def test_stop_copies_the_log_kills_the_job_and_frees_the_lock(rl, box):
    cmd = [PY, "-c", "import time; print('long job', flush=True); time.sleep(120)"]
    r = wr(box, "start", "long", "--", *cmd)
    procs = _job_procs(box, "long")
    try:
        assert r.returncode == 0, r.stdout + r.stderr
        assert len(procs) == 2, "supervisor and command should both be recorded with their create times"
        assert rl.is_held(box.lock)
        s = wr(box, "stop", "long")
        assert s.returncode == 0 and "stopped long" in s.stdout, s.stdout + s.stderr
        assert "killed 0 process" not in s.stdout
        assert "not available" in s.stdout  # PATH has no podman/docker: the leftover check reached none
        assert _until(lambda: not any(p.is_running() for p in procs), 15)  # the real job is gone
    finally:
        for p in procs:
            _kill_tree(p)
    assert _until(lambda: not rl.is_held(box.lock), 15)
    assert "long job" in (box.runs / "long.log").read_text(encoding="utf-8")
    st = wr(box, "status", "long")
    assert "STOPPED" in st.stdout and "stopped by hand" in st.stdout


# --- recorded pids are checked before anything is killed (F1) -------------------------------


def test_stop_never_kills_for_a_run_that_already_ended(box):
    """A FINISHED meta whose pid now belongs to a live, unrelated process (pids are reused; old metas
    outlive reboots): stop must not touch it, e.g. llama-server on :8090."""
    with Victim() as v:
        # started after the victim was made, so only the exit_code rule protects it here
        _meta(box, "oldsuite", started=v.ct + 10, ended=v.ct + 20, exit_code=0, pid=v.pid, cmd_pid=v.pid)
        s = wr(box, "stop", "oldsuite")
        assert s.returncode == 0, s.stdout + s.stderr
        assert v.alive(), "stop killed an unrelated process"
        assert "killed 0 process" in s.stdout and "already ended" in s.stdout, s.stdout


@pytest.mark.parametrize("ct_recorded", [False, True])
def test_a_reused_pid_is_not_the_job(box, ct_recorded):
    """A RUNNING meta (no exit_code) whose pid was reused by a younger process: not killed by stop,
    not "still going" for a new run, and status says DIED, not RUNNING."""
    with Victim() as v:
        if ct_recorded:  # recorded create time does not match the process that has the pid now
            d = dict(started=v.ct + 10, pid=v.pid, pid_ct=v.ct - 100, cmd_pid=v.pid, cmd_ct=v.ct - 100)
        else:  # an old meta without create times: the process is younger than the run's start
            d = dict(started=time.time() - 3600, pid=v.pid, cmd_pid=v.pid)
        _meta(box, "ghost", **d)
        st = wr(box, "status", "ghost")
        assert "DIED" in st.stdout, st.stdout
        s = wr(box, "stop", "ghost")
        assert s.returncode == 0 and "killed 0 process" in s.stdout, s.stdout + s.stderr
        assert v.alive(), "stop killed a process that only reused the job's pid"
        _meta(box, "ghost2", **d)
        r = wr(box, "run", "ghost2", "--", *say("== 1 passed in 0.01s =="))
        assert r.returncode == 0 and "still going" not in r.stdout + r.stderr, r.stdout + r.stderr
        assert v.alive()


# --- the run lock (E-3) ----------------------------------------------------------------------


def test_run_refuses_when_busy_and_never_runs_the_command(rl, box):
    """A refusal is recorded as its own stamped BUSY run, so status / wait never show the previous
    run's "exit 0 = all passed" for a run that never happened (F4). The previous run's stamped
    files are untouched."""
    assert wr(box, "run", "job", "--", *say("earlier green", "== 1 passed in 0.01s ==")).returncode == 0
    before = {p: p.read_bytes() for p in stamped(box, "job", "log") + stamped(box, "job", "json")}
    assert len(before) == 2
    marker = box.cwd / "ran.txt"
    write_marker = [PY, "-c", f"open({str(marker)!r}, 'w').write('x')"]
    with Holder(rl, box) as h:
        for action in ("run", "start"):
            r = wr(box, action, "job", "--", *write_marker)
            assert r.returncode == 3, action + r.stdout + r.stderr
            assert "busy:" in r.stdout and str(h.pid) in r.stdout
            d = json.loads((box.runs / "job.json").read_text(encoding="utf-8"))
            assert d["exit_code"] == 3 and d["killed_reason"].startswith("busy: ") and str(h.pid) in d["killed_reason"]
            for check in ("status", "wait"):
                st = wr(box, check, "job")
                assert st.returncode == 3, check + st.stdout
                assert "BUSY" in st.stdout and "never ran" in st.stdout and "FINISHED" not in st.stdout, st.stdout
        time.sleep(1)
    assert not marker.exists()
    assert all(p.read_bytes() == b for p, b in before.items())  # the earlier run's record is intact
    assert len(stamped(box, "job", "json")) == 3  # earlier run + two refusals, each its own stamp
    assert "busy:" in (box.runs / "job.log").read_text(encoding="utf-8")


def test_a_stray_flag_with_an_unrelated_holder_is_refused(rl, box):
    """F2: LOCALBOT_RUNLOCK_HELD=1 is not enough; the holder must be this process or an ancestor.
    Here a sibling process holds the lock, so watch_run must refuse, not run "under the parent"."""
    marker = box.cwd / "ran.txt"
    with Holder(rl, box) as h:
        r = wr(box, "run", "unrelated", "--", PY, "-c", f"open({str(marker)!r}, 'w').write('x')",
               LOCALBOT_RUNLOCK_HELD="1")
        assert r.returncode == 3, r.stdout + r.stderr
        assert "busy:" in r.stdout and str(h.pid) in r.stdout and "held by the parent" not in r.stdout
        s = wr(box, "start", "unrelated", "--", PY, "-c", f"open({str(marker)!r}, 'w').write('x')",
               LOCALBOT_RUNLOCK_HELD="1")
        assert s.returncode == 3 and "busy:" in s.stdout, s.stdout + s.stderr
        time.sleep(1)
    assert not marker.exists()


def test_a_command_under_a_start_supervisor_counts_as_covered(box):
    """The conftest case: pytest started by a detached `start` supervisor. The supervisor holds the
    lock and is the command's ancestor, so held_by_parent() is True in the command."""
    r = wr(box, "start", "cov", "--", *_probe())
    procs = _job_procs(box, "cov")
    try:
        w = wr(box, "wait", "cov", "--max", "60")
        assert "FINISHED" in w.stdout, r.stdout + w.stdout
    finally:
        for p in procs:
            _kill_tree(p)
    assert "HELD=1 parent=True holder=cov" in (box.runs / "cov.log").read_text(encoding="utf-8")


def test_pytest_and_its_xdist_workers_under_start_count_as_covered(box):
    """The same, literally: an inner pytest (with -n 2 workers when pytest-xdist is installed) run
    by a `start` supervisor asserts held_by_parent() inside its test, the conftest check."""
    tdir = box.cwd / "inner_tests"
    tdir.mkdir()
    (tdir / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tdir / "test_cov.py").write_text(
        "import importlib.util, os\n\n\n"
        "def test_covered():\n"
        f"    s = importlib.util.spec_from_file_location('rl', {str(RUNLOCK)!r})\n"
        "    m = importlib.util.module_from_spec(s)\n"
        "    s.loader.exec_module(m)\n"
        "    assert os.environ.get('LOCALBOT_RUNLOCK_HELD') == '1'\n"
        "    if os.environ.get('INNER_XDIST') == '1':  # really inside an xdist worker\n"
        "        assert os.environ.get('PYTEST_XDIST_WORKER', '').startswith('gw')\n"
        "    assert m.held_by_parent(), m.holder()\n", encoding="utf-8")
    args = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-c", str(tdir / "pytest.ini"),
            "--rootdir", str(tdir), str(tdir)]
    xdist = importlib.util.find_spec("xdist") is not None
    if xdist:
        args += ["-n", "2"]
    r = wr(box, "start", "pyt", "--", *args, INNER_XDIST="1" if xdist else "0")
    procs = _job_procs(box, "pyt")
    try:
        w = wr(box, "wait", "pyt", "--max", "120")
        assert "FINISHED" in w.stdout and "exit 0" in w.stdout, r.stdout + w.stdout + \
            (box.runs / "pyt.log").read_text(encoding="utf-8", errors="replace")
        assert "counts: 1 passed, 0 failed, 0 skipped, 0 errors" in w.stdout
    finally:
        for p in procs:
            _kill_tree(p)


def test_start_under_a_parent_that_holds_the_lock_is_refused(rl, box):
    """A detached job under a lock-holding parent would outlive that lock, and its command could not
    prove it is covered (the starter is gone from its ancestry): use run (foreground) there."""
    marker = box.cwd / "ran.txt"
    r = subprocess.run([PY, str(RUNLOCK), "run", "gpu_batch", "--", PY, str(SCRIPT), "start", "inner", "--",
                        PY, "-c", f"open({str(marker)!r}, 'w').write('x')"],
                       capture_output=True, text=True, encoding="utf-8", env=box.env, cwd=box.cwd, timeout=180)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "use run" in r.stdout + r.stderr
    time.sleep(1)
    assert not marker.exists() and not (box.runs / "inner.json").exists()


def test_child_that_loses_the_race_records_busy(rl, box, monkeypatch):
    """start pre-checks the lock; if another job takes it before the detached supervisor does, the
    supervisor records exit 3 + "busy: ..." and never runs the command."""
    spec = importlib.util.spec_from_file_location("watch_run_h1r", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # RUNS comes from LOCALBOT_RUNS_DIR (set by box)
    marker = box.cwd / "ran.txt"
    cmd = [PY, "-c", f"open({str(marker)!r}, 'w').write('x')"]
    mod.new_run("race", cmd, 300, 60)
    with Holder(rl, box) as h:
        r = wr(box, "_child", "race", "--stall", "300", "--timeout", "60", "--", *cmd)
        assert r.returncode == 3, r.stdout + r.stderr
    assert not marker.exists()
    d = json.loads((box.runs / "race.json").read_text(encoding="utf-8"))
    assert d["exit_code"] == 3 and d["killed_reason"].startswith("busy: ") and str(h.pid) in d["killed_reason"]
    st = wr(box, "status", "race")
    assert "BUSY" in st.stdout and "never ran" in st.stdout, st.stdout
    assert "busy:" in (box.runs / "race.log").read_text(encoding="utf-8")


def test_no_lock_runs_while_busy_without_the_flag(rl, box):
    marker = box.cwd / "ran.txt"
    probe = (f"import os; open({str(marker)!r}, 'w').write('x'); "
             "print('HELD=' + os.environ.get('LOCALBOT_RUNLOCK_HELD', '-'))")
    with Holder(rl, box):
        r = wr(box, "run", "free", "--no-lock", "--", PY, "-c", probe)
        assert r.returncode == 0, r.stdout + r.stderr
    assert marker.exists()
    assert "HELD=-" in (box.runs / "free.log").read_text(encoding="utf-8")


def _probe() -> list[str]:
    return [PY, "-c", "import importlib.util, os; "
            f"s = importlib.util.spec_from_file_location('rl', {str(RUNLOCK)!r}); m = importlib.util.module_from_spec(s); "
            "s.loader.exec_module(m); "
            "print('HELD=' + os.environ.get('LOCALBOT_RUNLOCK_HELD', '-'), 'parent=' + str(m.held_by_parent()), "
            "'holder=' + str((m.holder() or {}).get('name')))"]


def test_the_command_sees_the_held_flag(box):
    r = wr(box, "run", "job", "--", *_probe())
    assert r.returncode == 0, r.stdout + r.stderr
    assert "HELD=1 parent=True holder=job" in (box.runs / "job.log").read_text(encoding="utf-8")


def test_stale_flag_still_takes_the_lock(box):
    r = wr(box, "run", "job", "--", *_probe(), LOCALBOT_RUNLOCK_HELD="1")  # flag set, lock free
    assert r.returncode == 0, r.stdout + r.stderr
    assert "HELD=1 parent=True holder=job" in (box.runs / "job.log").read_text(encoding="utf-8")


def test_under_a_parent_that_holds_the_lock_it_is_not_taken_again(rl, box):
    """The GPU batch holds the lock and exports the flag; watch_run inside it must not fight it."""
    r = subprocess.run([PY, str(RUNLOCK), "run", "gpu_batch", "--", PY, str(SCRIPT), "run", "inner", "--", *_probe()],
                       capture_output=True, text=True, encoding="utf-8", env=box.env, cwd=box.cwd, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "busy" not in r.stdout + r.stderr
    assert "HELD=1 parent=True holder=gpu_batch" in (box.runs / "inner.log").read_text(encoding="utf-8")
