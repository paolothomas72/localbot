"""H1 Track T (audit E-3): scripts/runlock.py, one run lock for the whole PC.

Overlapping runs voided each other twice (a suite or G2 sweeps every labeled container, S-2; a running
exam's HostWatch counts other runs' _trials folders, S-3). The lock is a byte-range lock on byte 0 of
<main checkout>/state/run.lock, so it holds across worktrees; the OS drops it when the holder dies.
scripts/runlock.ps1 takes the same byte from PowerShell, and g2_run.py takes it before anything else.

Every test points LOCALBOT_RUNLOCK_PATH at tmp_path: the real state/run.lock is never opened. The g2
test also strips PATH down to python's own folder, so even a broken lock cannot reach podman/docker
(g2 then stops with "no podman/docker runtime available", exit 2). No container, no model, no GPU.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "runlock.py"
PS1 = REPO / "scripts" / "runlock.ps1"
G2 = REPO / "scripts" / "g2_run.py"
HELD = "LOCALBOT_RUNLOCK_HELD"
WIN = os.name == "nt"
POWERSHELL = shutil.which("powershell") if WIN else None
SLEEP = "import time; time.sleep(90)"
KEYS = {"pid", "name", "cmd", "start", "start_epoch"}


@pytest.fixture(scope="module")
def rl():
    spec = importlib.util.spec_from_file_location("runlock_h1", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def lock(tmp_path, monkeypatch) -> Path:
    """A private lock path for this test, also seen by in-process calls; no inherited HELD flag."""
    p = tmp_path / "state" / "run.lock"
    monkeypatch.setenv("LOCALBOT_RUNLOCK_PATH", str(p))
    monkeypatch.delenv(HELD, raising=False)
    return p


def _env(lock: Path, **extra: str) -> dict:
    env = dict(os.environ, LOCALBOT_RUNLOCK_PATH=str(lock), PYTHONDONTWRITEBYTECODE="1",
               PATH=str(Path(sys.executable).parent), NoDefaultCurrentDirectoryInExePath="1")
    env.pop(HELD, None)
    env.update(extra)
    return env


def _cli(lock: Path, *argv: str, **extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *argv], capture_output=True, text=True,
                          encoding="utf-8", env=_env(lock, **extra), cwd=lock.parent.parent, timeout=120)


def _tree(pid: int) -> list[int]:
    try:
        root = psutil.Process(pid)
        return [pid] + [c.pid for c in root.children(recursive=True)]
    except psutil.Error:
        return [pid]


def _kill_tree(pid: int) -> None:
    """Kill the tree of one of our own Popen children. Only ever called with ``Popen.pid`` before the
    Popen is waited, so the pid cannot have been reused (never with a pid read back from a file)."""
    try:
        root = psutil.Process(pid)
        if not root.is_running():
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


def _until(pred, timeout: float = 30.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.1)
    return pred()


class Holder:
    """``runlock.py run <name> -- python -c <sleep>`` in the background, holding ``lock``."""

    def __init__(self, rl, lock: Path, name: str = "holder"):
        self.proc = subprocess.Popen([sys.executable, str(SCRIPT), "run", name, "--", sys.executable, "-c", SLEEP],
                                     env=_env(lock), cwd=lock.parent.parent,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Holder file first: it is written only after the lock is taken, so is_held() (which takes the
        # lock for an instant when it is free) never races the holder's own attempt.
        ok = _until(lambda: (rl.holder(lock) or {}).get("name") == name and rl.is_held(lock))
        if not ok:
            self.kill()
            pytest.fail("the holder process never took the lock")
        self.pid = rl.holder(lock)["pid"]

    def kill(self) -> None:
        if self.proc.returncode is not None:  # already killed and waited: the pid may be reused now
            return
        _kill_tree(self.proc.pid)
        self.proc.wait(timeout=30)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.kill()


def test_second_process_is_refused_until_the_holder_dies(rl, lock):
    with Holder(rl, lock) as h:
        assert h.pid in _tree(h.proc.pid)  # the venv python.exe is a launcher: the holder may be its child
        with pytest.raises(rl.LockBusy) as busy:
            rl.acquire("second", path=lock)
        assert busy.value.holder["pid"] == h.pid
        assert busy.value.holder["name"] == "holder"
        assert set(busy.value.holder) == KEYS  # no hostname / PC name
        assert "busy:" in str(busy.value)
        st = _cli(lock, "status")
        assert st.returncode == 3, st.stdout + st.stderr
        assert st.stdout.startswith("BUSY") and str(h.pid) in st.stdout
        assert rl.is_held(lock)
        h.kill()
        # Killed hard: it never removed its holder file, but the OS dropped the lock.
        assert rl.holder(lock)["pid"] == h.pid
        assert _until(lambda: not rl.is_held(lock), 15)
        st = _cli(lock, "status")
        assert st.returncode == 0, st.stdout + st.stderr
        assert st.stdout.startswith("FREE") and "stale" in st.stdout
        with rl.acquire("after", path=lock) as mine:
            assert rl.holder(lock)["pid"] == os.getpid() and rl.holder(lock)["name"] == "after"
            assert mine.path == lock


def test_status_free_and_release_removes_only_our_holder_file(rl, lock):
    st = _cli(lock, "status")
    assert st.returncode == 0 and st.stdout.startswith("FREE"), st.stdout + st.stderr
    assert not rl.is_held(lock)
    held = rl.acquire("mine", cmd=["python", "-m", "pytest"])
    info = rl.holder(lock)
    assert set(info) == KEYS and info["cmd"] == "python -m pytest" and info["start"].endswith("Z")
    assert rl.is_held(lock)  # a second handle in the same process is refused too
    held.release()
    held.release()  # twice is harmless
    assert not rl.is_held(lock)
    assert rl.holder(lock) is None and not lock.with_name("run.lock.json").exists()
    # A holder file that is no longer ours (someone else's after a race) is left alone.
    held = rl.acquire("mine2", path=lock)
    other = dict(rl.holder(lock), pid=1, name="someone else")
    lock.with_name("run.lock.json").write_text(json.dumps(other), encoding="utf-8")
    held.release()
    assert rl.holder(lock)["name"] == "someone else"


def test_cli_run_busy_exits_3_and_never_runs_the_command(rl, lock, tmp_path):
    marker = tmp_path / "ran.txt"
    with Holder(rl, lock) as h:
        r = _cli(lock, "run", "second", "--", sys.executable, "-c", f"open({str(marker)!r}, 'w').write('x')")
        assert r.returncode == 3, r.stdout + r.stderr
        assert "busy:" in r.stdout and str(h.pid) in r.stdout
    assert not marker.exists()


def test_cli_run_exports_the_flag_and_returns_the_child_code(rl, lock):
    probe = ("import importlib.util, os, sys; "
             f"s = importlib.util.spec_from_file_location('rl', {str(SCRIPT)!r}); m = importlib.util.module_from_spec(s); "
             "s.loader.exec_module(m); "
             "print('HELD=' + os.environ.get('LOCALBOT_RUNLOCK_HELD', ''), 'parent=' + str(m.held_by_parent()), "
             "'name=' + m.holder()['name']); sys.exit(7)")
    r = _cli(lock, "run", "job", "--", sys.executable, "-c", probe)
    assert r.returncode == 7, r.stdout + r.stderr
    assert "HELD=1 parent=True name=job" in r.stdout
    assert not rl.is_held(lock) and rl.holder(lock) is None  # released and cleaned up


def test_stale_held_flag_cannot_bypass_the_lock(rl, lock, monkeypatch):
    monkeypatch.setenv(HELD, "1")
    assert not rl.is_held(lock)
    assert rl.held_by_parent() is False  # flag set, lock free = a stray flag: take the lock as usual
    monkeypatch.setenv(HELD, "0")
    with rl.acquire("x"):
        assert rl.held_by_parent() is False  # flag not "1"
        monkeypatch.setenv(HELD, "1")
        assert rl.held_by_parent() is True
    assert rl.held_by_parent() is False


def test_flag_with_an_unrelated_holder_is_not_covered(rl, lock, monkeypatch):
    """F2: the flag counts only if the holder is this process or one of its ancestors."""
    monkeypatch.setenv(HELD, "1")
    with Holder(rl, lock) as h:  # a sibling process holds it, not a parent
        assert rl.is_held(lock) and rl.holder(lock)["pid"] == h.pid
        assert rl.held_by_parent() is False
    assert _until(lambda: not rl.is_held(lock), 15)
    with rl.acquire("me", path=lock):
        assert rl.held_by_parent() is True  # this process holds it
        hf = lock.with_name("run.lock.json")
        mine = hf.read_text(encoding="utf-8")
        sleeper = subprocess.Popen([sys.executable, "-c", SLEEP], env=_env(lock))
        try:  # the holder file names a live process that is not an ancestor
            hf.write_text(json.dumps(dict(json.loads(mine), pid=sleeper.pid)), encoding="utf-8")
            assert rl.held_by_parent() is False
        finally:
            _kill_tree(sleeper.pid)
            sleeper.wait(timeout=30)
        hf.write_text(json.dumps(dict(json.loads(mine), pid="x")), encoding="utf-8")
        assert rl.held_by_parent() is False  # unreadable pid
        hf.unlink()
        assert rl.held_by_parent() is False  # no holder file: cannot prove it, take the lock
        hf.write_text(mine, encoding="utf-8")
        assert rl.held_by_parent() is True
        monkeypatch.setitem(sys.modules, "psutil", None)  # psutil missing: fail safe
        assert rl.held_by_parent() is False


def test_cli_run_under_a_held_lock_does_not_take_it_again(rl, lock):
    inner = [sys.executable, str(SCRIPT), "run", "inner", "--", sys.executable, "-c", "print('inner ran')"]
    r = _cli(lock, "run", "outer", "--", *inner)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "inner ran" in r.stdout and "busy" not in r.stdout + r.stderr


def _worktree(tmp_path: Path, gitdir_text: str | None = None, commondir: str | None = "../..") -> tuple[Path, Path]:
    main = tmp_path / "main"
    wt_admin = main / ".git" / "worktrees" / "wt"
    wt_admin.mkdir(parents=True)
    if commondir is not None:
        (wt_admin / "commondir").write_text(commondir + "\n", encoding="utf-8")
    wt = tmp_path / "elsewhere" / "wt"
    wt.mkdir(parents=True)
    (wt / ".git").write_text((gitdir_text or f"gitdir: {wt_admin.as_posix()}") + "\n", encoding="utf-8")
    return main, wt


def test_main_checkout_from_a_worktree_layout(rl, tmp_path):
    main, wt = _worktree(tmp_path)
    assert rl.main_checkout(wt).resolve() == main.resolve()
    assert rl.main_checkout(main).resolve() == main.resolve()  # .git is a dir: the repo itself
    lone = tmp_path / "no_git"
    lone.mkdir()
    assert rl.main_checkout(lone) == lone  # no .git at all: fall back to the repo


def test_main_checkout_relative_gitdir_and_fallbacks(rl, tmp_path):
    main, wt = _worktree(tmp_path, gitdir_text="gitdir: ../../main/.git/worktrees/wt")
    assert rl.main_checkout(wt).resolve() == main.resolve()
    _, wt2 = _worktree(tmp_path / "b", commondir=None)  # no commondir file
    assert rl.main_checkout(wt2) == wt2
    odd = tmp_path / "odd"
    odd.mkdir()
    (odd / ".git").write_text("not a gitdir line\n", encoding="utf-8")
    assert rl.main_checkout(odd) == odd


def test_lock_path_env_and_default(rl, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALBOT_RUNLOCK_PATH", str(tmp_path / "x.lock"))
    assert rl.lock_path() == tmp_path / "x.lock"
    monkeypatch.delenv("LOCALBOT_RUNLOCK_PATH")
    assert rl.lock_path() == rl.main_checkout() / "state" / "run.lock"  # computed only, never opened here
    git = shutil.which("git")
    if git:
        out = subprocess.run([git, "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=REPO,
                             capture_output=True, text=True, timeout=60)
        if out.returncode == 0:
            assert rl.main_checkout().resolve() == Path(out.stdout.strip()).resolve().parent


def test_g2_refuses_when_the_lock_is_busy(rl, lock, tmp_path):
    """g2 takes the lock first. PATH holds only python's folder, so a broken lock would still stop
    at "no podman/docker runtime available" (exit 2) and never touch a container."""
    def g2(*argv):
        return subprocess.run([sys.executable, str(G2), *argv], capture_output=True, text=True, encoding="utf-8",
                              env=_env(lock), cwd=tmp_path, timeout=180)

    with Holder(rl, lock) as h:
        r = g2("podman")
        assert r.returncode == 3, r.stdout + r.stderr
        assert "busy:" in r.stderr and str(h.pid) in r.stderr
        assert "no podman/docker runtime" not in r.stderr
    # Free lock: g2 takes it, finds no runtime on this PATH, stops with exit 2 and lets the lock go.
    assert _until(lambda: not rl.is_held(lock), 15)
    r = g2("podman")
    assert r.returncode == 2, r.stdout + r.stderr
    assert "no podman/docker runtime available" in r.stderr
    assert not rl.is_held(lock) and rl.holder(lock) is None


# --- PowerShell interop: runlock.ps1 locks the same byte ------------------------------------

needs_ps = pytest.mark.skipif(not POWERSHELL, reason="Windows PowerShell only (runlock.ps1)")


def _ps(script: str, lock: Path | None, **popen) -> subprocess.Popen:
    env = dict(os.environ)
    env.pop(HELD, None)
    if lock is None:
        env.pop("LOCALBOT_RUNLOCK_PATH", None)
    else:
        env["LOCALBOT_RUNLOCK_PATH"] = str(lock)
    full = f". '{PS1}'; {script}"
    return subprocess.Popen([POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                             "-Command", full], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace", **popen)


@needs_ps
def test_powershell_holder_refuses_python_and_releases(rl, lock, tmp_path, monkeypatch):
    p = _ps("$l = Enter-RunLock -Name ps_holder; if ($l) { 'HELD'; Start-Sleep -Seconds 60 } else { 'BUSY' }",
            lock)
    try:
        assert _until(lambda: (rl.holder(lock) or {}).get("name") == "ps_holder" and rl.is_held(lock), 60), \
            "PowerShell never took the lock"
        info = rl.holder(lock)
        assert info["pid"] == p.pid and set(info) == KEYS and info["start"].endswith("Z")
        with pytest.raises(rl.LockBusy) as busy:
            rl.acquire("python", path=lock)
        assert busy.value.holder["name"] == "ps_holder"
        # F2: a stray flag does not make an unrelated PowerShell holder "the parent".
        monkeypatch.setenv(HELD, "1")
        assert rl.held_by_parent() is False
        marker = tmp_path / "ran.txt"
        r = subprocess.run([sys.executable, str(REPO / "scripts" / "watch_run.py"), "run", "unrelated", "--",
                            sys.executable, "-c", f"open({str(marker)!r}, 'w').write('x')"],
                           capture_output=True, text=True, encoding="utf-8", cwd=tmp_path, timeout=180,
                           env=_env(lock, LOCALBOT_RUNLOCK_HELD="1", LOCALBOT_RUNS_DIR=str(tmp_path / "runs")))
        assert r.returncode == 3 and "busy:" in r.stdout and "ps_holder" in r.stdout, r.stdout + r.stderr
        assert not marker.exists()
    finally:
        _kill_tree(p.pid)
        p.communicate(timeout=30)
    assert _until(lambda: not rl.is_held(lock), 15)
    # Exit-RunLock releases and removes its own holder file.
    p = _ps("$l = Enter-RunLock -Name ps_once; if ($l) { Exit-RunLock $l; 'RELEASED' } else { 'BUSY' }", lock)
    out, err = p.communicate(timeout=120)
    assert "RELEASED" in out, out + err
    assert not rl.is_held(lock) and rl.holder(lock) is None
    with rl.acquire("python", path=lock):
        pass


@needs_ps
def test_python_holder_refuses_powershell(rl, lock):
    with rl.acquire("py_holder", path=lock):
        p = _ps("$l = Enter-RunLock -Name ps_second; if ($l) { 'HELD'; Exit-RunLock $l } "
                "else { 'BUSY ' + (Get-RunLockHolder) }", lock)
        out, err = p.communicate(timeout=120)
        assert "BUSY" in out and "HELD" not in out, out + err
        assert str(os.getpid()) in out and "py_holder" in out
        assert rl.holder(lock)["name"] == "py_holder"  # PowerShell did not touch our holder file


@needs_ps
def test_powershell_and_python_agree_on_the_default_path(rl, monkeypatch):
    monkeypatch.delenv("LOCALBOT_RUNLOCK_PATH", raising=False)
    p = _ps("Get-RunLockPath", None, cwd=str(REPO))
    out, err = p.communicate(timeout=120)
    assert p.returncode == 0, out + err
    assert Path(out.strip()).resolve() == rl.lock_path().resolve()  # path only; the real lock is not opened
