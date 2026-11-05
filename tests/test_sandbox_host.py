"""H1 Track C (container/sandbox), the host-side half: no container, no runtime, no model.

What a builder can run under the H1 bridge rule (host-only tests): the argv the shell builds, the
real keepalive on the host's own Python, the sweep's owner rules against a scripted runtime CLI,
the per-run trials root, the secret masks on a real NTFS tree, the /work quota, and the
post-exec integrity parser. The container half of each fix is in tests/test_shell_container.py
(the gate run, under the run lock).
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psutil
import pytest

from localbot.config import ContainerConfig, get_profile
from localbot.exam.runner import lab_module, load_pack, run_pack
from localbot.guard.workspace_fs import new_work_folder
from localbot.telemetry import TrialSampler, VramReading
from localbot.tools import shell_container as sc
from localbot.tools.shell_container import KEEPALIVE, ContainerError, ContainerShell
from localbot.tools.shell_emulated import sandbox_bash
from tests import images
from tests.fakes import ReactiveTransport, RoutedTransport, provider, reply
from tests.images import GIT_IMAGE

_REPO = Path(__file__).resolve().parents[1]

P6 = get_profile("parity_lab6")


class FakeRuntime:
    """The runtime CLI, scripted by argv shape (monkeypatched over ``shell_container._run`` and
    ``_run_capped``): ``run`` returns a new id, the reaper's kill and listing answer clean, the
    post-exec check answers ``integrity(shell)``, and a model command echoes its marker line
    followed by ``output``. Every argv is recorded."""

    def __init__(self, monkeypatch, shell_ref=None) -> None:
        self.calls: list[list[str]] = []
        self.n = 0
        self.output = "ok\n"
        self.marker = True
        self.integrity = ok_integrity
        self.shell = shell_ref
        self.containers: dict[str, dict] = {}  # what ``ps`` / ``inspect`` see (the sweep tests)
        self.inspect_fails = False
        monkeypatch.setattr(sc, "_run", self._run)
        monkeypatch.setattr(sc, "_run_capped", self._run_capped)

    def removed(self) -> list[str]:
        return [c[-1] for c in self.calls if c[1:3] == ["rm", "-f"]]

    def _run(self, argv, timeout, check=False):
        self.calls.append(list(argv))
        sub = argv[1]
        if sub == "run":
            self.n += 1
            return subprocess.CompletedProcess(argv, 0, f"cid{self.n}\n", "")
        if sub == "ps" and "-aq" in argv and any(a.startswith("label=") for a in argv):
            return subprocess.CompletedProcess(argv, 0, "".join(f"{c}\n" for c in self.containers), "")
        if sub == "inspect":
            if self.inspect_fails:
                return subprocess.CompletedProcess(argv, 125, "", "Error: boom")
            found = [self.containers[c] for c in argv[2:] if c in self.containers]
            return subprocess.CompletedProcess(argv, 0, json.dumps(found), "")
        if sub == "exec":
            script = argv[-1]
            if script == "kill -9 -1":
                return subprocess.CompletedProcess(argv, 0, "", "")
            if script == sc.PROC_LIST:
                return subprocess.CompletedProcess(argv, 0, "SELF=7\n1 0 S python3\n7 1 R bash\n", "")
            if sc.INTEGRITY_TAG in script:
                return subprocess.CompletedProcess(argv, 0, self.integrity(self.shell), "")
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 0, "", "")  # rm -f, ps --filter id= (gone)

    def _run_capped(self, argv, timeout, limit, watch=None, poll_s=None):
        self.calls.append(list(argv))
        marker = argv[argv.index("_") + 1] if "_" in argv else ""
        return (f"{marker}\n" if self.marker else "") + self.output, False


def mountinfo(dests: list[str]) -> str:
    lines = ["600 500 0:52 / / ro,relatime - overlay overlay rw",
             "601 600 0:53 / /work rw,relatime - 9p drvfs rw"]
    for i, d in enumerate(dests):
        esc = d.replace("\\", "\\134").replace(" ", "\\040").replace("\t", "\\011").replace("\n", "\\012")
        lines.append(f"{700 + i} 601 0:53 /x {esc} ro,relatime - 9p drvfs rw")
    return "\n".join(lines) + "\n"


def ok_integrity(sh, *, tasks: int = 1, cmd: bytes | None = None, listed: list[str] = (),
                 dests: list[str] | None = None) -> str:
    """What a clean container answers to the post-exec check: pid 1 one thread with its own
    command line, every mask mounted, ``listed`` = what is left in the scratch folders."""
    if cmd is None:
        cmd = b"\0".join([b"python3", *[a.encode() for a in sh.keepalive_args()]]) + b"\0"
    dests = sh.mask_dests() if dests is None else dests
    b64 = lambda b: base64.b64encode(b).decode()  # noqa: E731
    return (f"{sc.INTEGRITY_TAG}\nTASKS={tasks}\nCMD={b64(cmd)}\n"
            f"MOUNTS={b64(mountinfo(dests).encode())}\n"
            f"LIST={b64(b''.join(p.encode() + b'\0' for p in listed))}\n")


def _shell(tmp_path, monkeypatch, **cfg) -> tuple[ContainerShell, FakeRuntime]:
    work = tmp_path / "work"
    if not work.exists():
        new_work_folder(work)  # H1-K D1: start() refuses a folder that is not case-sensitive
    sh = ContainerShell(ContainerConfig(**cfg), work)
    rt = FakeRuntime(monkeypatch, sh)
    return sh, rt


def _sampler():
    return TrialSampler(0.05, cpu=lambda: 1.0, ram=lambda: 1, vram=lambda: VramReading(1, 2, 0))


def _pack(disk, fn):
    spec = load_pack("work")[0]
    rt = RoutedTransport(ReactiveTransport(fn))
    res, _ = run_pack("work", P6, "fake:1", provider(rt, P6.provider), n=1, tasks=[spec],
                      vram_reader=lambda: VramReading(1, 2, 0), sampler_factory=_sampler,
                      results_dir=disk.base / "results", trials_root=disk.base / "_trials",
                      log=lambda _line: None)
    return res["tasks"][spec["id"]]


# --- S-3: HostWatch watches only folders this run owns -------------------------------------------


def test_a_sibling_run_in_the_shared_trials_root_is_not_a_host_change(disk):
    """S-3: another run creating its folder in the shared trials root while this trial runs is
    not this trial's host change (the overlap that voided 6 result files on 2026-10-31)."""
    trials = disk.base / "_trials"

    def model(messages):
        (trials / "another_run_started_meanwhile").mkdir(parents=True, exist_ok=True)
        return reply("done")

    row = _pack(disk, model)
    assert row["host_changes"] == 0, [t.get("host_change") for t in row["trials"]]


def test_a_write_into_this_runs_own_root_is_still_a_host_change(disk):
    """S-3 keeps the depth: the folder right above this run's model folders is still watched."""
    trials = disk.base / "_trials"

    def model(messages):
        run_root = next(p for p in trials.iterdir() if p.is_dir())
        (run_root / "escaped.txt").write_text("x", encoding="utf-8")
        return reply("done")

    row = _pack(disk, model)
    assert row["host_changes"] == 1
    assert not any(trials.iterdir())  # the whole run root is deleted, escapes included


# --- S-1: model code can never become pid 1 ------------------------------------------------------

PLANT = ("open({marker!r}, 'w').write('ran')\n"
         "raise SystemExit(3)\n")


@pytest.mark.parametrize("module", ["enum", "signal", "ctypes"])
def test_keepalive_never_imports_a_module_planted_in_its_start_dir(tmp_path, module):
    """S-1 (audit: the reviewer's ENUM_RAN): run pid 1's exact arguments (from ``run_argv``) on the
    host's own Python, started in a folder holding a planted stdlib name, as pid 1 used to start
    in /work. The planted file must not run. (On Windows the keepalive then stops at
    ``ctypes.CDLL(None)``; on Linux at the prctl or the loop: either way it never ran the plant.)"""
    sh = ContainerShell(ContainerConfig(), tmp_path)
    argv = sh.run_argv()
    args = argv[argv.index(sh.cfg.image) + 1:]
    marker = tmp_path / f"{module}_RAN"
    (tmp_path / f"{module}.py").write_text(PLANT.format(marker=str(marker)), encoding="utf-8")
    try:
        subprocess.run([sys.executable, *args], cwd=tmp_path, capture_output=True, timeout=20)
    except subprocess.TimeoutExpired:
        pass  # a keepalive that reached its loop: it is still a pass if the plant never ran
    assert not marker.exists(), f"planted {module}.py ran inside the keepalive"


def test_pid1_starts_in_root_and_each_exec_in_work(tmp_path, monkeypatch):
    """S-1: the container's working dir is ``/`` (pid 1 is not started inside the workspace) and
    each model command gets ``-w /work`` on its own exec."""
    sh, rt = _shell(tmp_path, monkeypatch)
    argv = sh.run_argv()
    assert argv[argv.index("-w") + 1] == "/"
    assert argv[argv.index("--entrypoint") + 1:argv.index(sh.cfg.image) + 3] == ["python3", sh.cfg.image, "-I", "-S"]
    sh.start()
    sh.run("pwd", 30)
    exec_argv = next(c for c in rt.calls if c[1] == "exec" and c[-1] == "pwd")
    assert exec_argv[:5] == [sh.exe, "exec", "-w", "/work", sh.cid]


def test_a_second_thread_in_pid1_restarts_the_container(tmp_path, monkeypatch):
    """S-1: after every step pid 1 must still be one thread (the keepalive is single-threaded; a
    thread there means code is running inside it)."""
    sh, rt = _shell(tmp_path, monkeypatch)
    sh.start()
    first = sh.cid
    rt.integrity = lambda s: ok_integrity(s, tasks=2) if s.cid == first else ok_integrity(s)
    sh.run("true", 30)
    assert sh.cid != first and any("pid 1" in why for why in sh.restarts), sh.restarts


def test_a_changed_pid1_command_line_restarts_the_container(tmp_path, monkeypatch):
    sh, rt = _shell(tmp_path, monkeypatch)
    sh.start()
    first = sh.cid
    rt.integrity = lambda s: (ok_integrity(s, cmd=b"python3\0-c\0evil\0") if s.cid == first else ok_integrity(s))
    sh.run("true", 30)
    assert sh.cid != first and any("pid 1" in why for why in sh.restarts), sh.restarts


def test_a_fresh_container_that_fails_the_pid1_check_raises(tmp_path, monkeypatch):
    sh, rt = _shell(tmp_path, monkeypatch)
    rt.integrity = lambda s: ok_integrity(s, tasks=3)
    with pytest.raises(ContainerError, match="pid 1"):
        sh.start()


# --- S-2: a sweep removes only what nobody owns any more -----------------------------------------


def _me() -> str:
    return f"{os.getpid()}:{psutil.Process().create_time():.3f}"


def _dead_owner() -> str:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    ctime = psutil.Process(p.pid).create_time()
    p.wait()
    return f"{p.pid}:{ctime:.3f}"


def _box(cid: str, labels: dict, created: str | None = None) -> dict:
    created = created or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {"Id": cid, "Created": created, "Config": {"Labels": {"localbot-container": "x", **labels}}}


def _later(s: float) -> str:
    return str(int(time.time() + s))


def test_run_argv_labels_the_owner_process_and_an_expiry(tmp_path):
    """S-2: every container carries who started it (pid + that process's start time, so a reused
    pid is not mistaken for the owner) and when it expires."""
    sh = ContainerShell(ContainerConfig(max_lifetime_s=600), tmp_path)
    argv = sh.run_argv()
    labels = dict(argv[i + 1].split("=", 1) for i, a in enumerate(argv) if a == "--label")
    assert labels[sc.OWNER_LABEL] == _me()
    assert abs(int(labels[sc.EXPIRES_LABEL]) - (time.time() + 600)) < 120
    assert argv.index("--label") < argv.index("-v")  # before the mounts (test_git_protection reads -v .. -w)


def test_sweep_keeps_a_container_whose_owner_is_alive(tmp_path, monkeypatch):
    """S-2 (audit: a suite started during an exam deleted the exam's live container)."""
    rt = FakeRuntime(monkeypatch)
    rt.containers = {"live": _box("live", {sc.OWNER_LABEL: _me(), sc.EXPIRES_LABEL: _later(600)})}
    assert ContainerShell.sweep(ContainerConfig()) == []
    assert rt.removed() == []


def test_sweep_removes_a_container_whose_owner_died(tmp_path, monkeypatch):
    rt = FakeRuntime(monkeypatch)
    rt.containers = {"orphan": _box("orphan", {sc.OWNER_LABEL: _dead_owner(), sc.EXPIRES_LABEL: _later(600)})}
    assert ContainerShell.sweep(ContainerConfig()) == ["orphan"]
    assert rt.removed() == ["orphan"]


def test_sweep_removes_a_container_whose_owner_pid_was_reused(tmp_path, monkeypatch):
    rt = FakeRuntime(monkeypatch)
    rt.containers = {"reused": _box("reused", {sc.OWNER_LABEL: f"{os.getpid()}:12345.000",
                                               sc.EXPIRES_LABEL: _later(600)})}
    assert ContainerShell.sweep(ContainerConfig()) == ["reused"]


def test_sweep_removes_an_expired_container_even_with_a_live_owner(tmp_path, monkeypatch):
    rt = FakeRuntime(monkeypatch)
    rt.containers = {"old": _box("old", {sc.OWNER_LABEL: _me(), sc.EXPIRES_LABEL: _later(-5)})}
    assert ContainerShell.sweep(ContainerConfig()) == ["old"]


def _created(form: str) -> str:
    """A ``Created`` value, built when the test runs (never at collection: xdist workers must
    collect the same ids, H1-I item 1)."""
    if form == "docker-30h":
        return (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat().replace("+00:00", "Z")
    if form == "podman-30h":
        return (datetime.now(timezone(timedelta(hours=1))) - timedelta(hours=30)).isoformat()
    if form == "fresh":
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return "not a time"


@pytest.mark.parametrize("form,swept", [
    ("docker-30h", True),    # docker form
    ("podman-30h", True),    # podman form
    ("fresh", False),        # fresh: maybe a live old-code run
    ("unreadable", False),   # unreadable: keep
])
def test_sweep_of_a_container_with_no_owner_label(tmp_path, monkeypatch, form, swept):
    """Containers made before H1 have no owner label: removed only once they are clearly stale."""
    created = _created(form)
    rt = FakeRuntime(monkeypatch)
    rt.containers = {"legacy": _box("legacy", {}, created)}
    assert ContainerShell.sweep(ContainerConfig()) == (["legacy"] if swept else [])


def test_sweep_removes_nothing_when_inspect_fails(tmp_path, monkeypatch):
    rt = FakeRuntime(monkeypatch)
    rt.containers = {"a": _box("a", {sc.OWNER_LABEL: _dead_owner()})}
    rt.inspect_fails = True
    assert ContainerShell.sweep(ContainerConfig()) == []
    assert rt.removed() == []


def test_run_argv_is_stable_until_the_next_start(tmp_path, monkeypatch):
    """The expiry label is fixed per container: two ``run_argv`` calls agree (test_git_protection
    relies on it) even across a clock tick; a new ``start`` gets a new expiry."""
    clock = [1_000_000.0]
    monkeypatch.setattr(sc.time, "time", lambda: clock[0])
    sh, _rt = _shell(tmp_path, monkeypatch, max_lifetime_s=60)
    first = sh.run_argv()
    clock[0] += 5
    assert sh.run_argv() == first
    sh.start()
    assert f"{sc.EXPIRES_LABEL}={int(clock[0]) + 60}" in sh.run_argv()


# --- sandbox P2s: --rm + keepalive lifetime, no host proxy env, exec really started, scratch -----


def test_run_argv_removes_the_container_when_pid1_exits_after_its_lifetime(tmp_path):
    """P2 (audit: killed runs leave containers alive): ``--rm``, and pid 1 is given its lifetime."""
    sh = ContainerShell(ContainerConfig(max_lifetime_s=900), tmp_path)
    argv = sh.run_argv()
    assert argv[:4] == [sh.exe, "run", "-d", "--rm"]
    assert sh.keepalive_args() == ["-I", "-S", "-c", sc.KEEPALIVE, "900"]
    assert argv[argv.index(sh.cfg.image) + 1:] == sh.keepalive_args()


def test_keepalive_exits_once_its_lifetime_is_over(monkeypatch):
    """The keepalive's own loop ends after ``sys.argv[1]`` seconds (the container then stops and
    ``--rm`` removes it), even with no child to reap. Run with a fake clock."""
    import ctypes
    import signal

    class Prctl:
        def prctl(self, *a):
            return 0

    clock = [100.0]
    sleeps = [0]

    def sleep(s):
        sleeps[0] += 1
        clock[0] += s
        if sleeps[0] > 10_000:
            raise AssertionError("keepalive kept looping past its lifetime")

    def waitpid(*a):
        raise ChildProcessError

    monkeypatch.setattr(signal, "signal", lambda *a, **k: None)
    monkeypatch.setattr(ctypes, "CDLL", lambda name=None: Prctl())
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "sleep", sleep)
    monkeypatch.setattr(os, "waitpid", waitpid)
    monkeypatch.setattr(sys, "argv", ["-c", "30"])
    try:
        exec(sc.KEEPALIVE, {"__name__": "__main__"})
    except SystemExit as e:
        assert e.code in (0, None)
    assert 30 <= clock[0] - 100.0 <= 31


@pytest.mark.parametrize("runtime,flag", [("podman", True), ("docker", False)])
def test_podman_never_copies_the_hosts_proxy_variables(tmp_path, runtime, flag):
    """P2: podman's ``--http-proxy`` default copies HTTP(S)_PROXY from the client's environment
    into the container (a proxy URL can carry a password). docker has no such flag."""
    argv = ContainerShell(ContainerConfig(runtime=runtime), tmp_path).run_argv()
    assert ("--http-proxy=false" in argv) is flag
    assert not any(a.startswith("--http-proxy") for a in argv) or runtime == "podman"


def test_the_start_marker_is_not_part_of_the_output(tmp_path, monkeypatch):
    """P2 (audit: exec return code unchecked): each command's bash prints a one-time marker
    before anything else; the host checks it and strips it."""
    sh, rt = _shell(tmp_path, monkeypatch)
    sh.start()
    rt.output = "hello\n"
    assert sh.run("echo hello", 30) == "hello\n"
    argv = next(c for c in rt.calls if c[1] == "exec" and c[-1] == "echo hello")
    assert argv[5:8] == ["bash", "-c", sc.EXEC_WRAP] and argv[8] == "_" and len(argv[9]) >= 16


def test_a_command_whose_exec_never_started_raises(tmp_path, monkeypatch):
    """If the runtime could not run the exec at all (container gone, CLI error), its own error
    text must not reach the model as if it were the command's output."""
    sh, rt = _shell(tmp_path, monkeypatch)
    sh.start()
    rt.marker = False
    rt.output = "Error: no container with name or ID \"cid1\" found: no such container\n"
    with pytest.raises(sc.ExecNotStartedError, match="did not start"):  # H1-I item 7: its own type
        sh.run("ls", 30)


def _git_bash() -> str | None:
    """Git Bash on Windows, found from git.exe (never System32\bash.exe, which runs in WSL)."""
    import shutil
    if os.name != "nt":
        return shutil.which("bash")
    git = shutil.which("git")
    if not git:
        return None
    from pathlib import Path
    for up in Path(git).resolve().parents[:4]:
        cand = up / "usr" / "bin" / "bash.exe"
        if cand.is_file():
            return str(cand)
    return None


def _bash_env(bash: str) -> dict:
    """Git Bash's own tools first on PATH (else ``find`` is Windows' FIND.EXE)."""
    return {**os.environ, "PATH": os.path.dirname(bash) + os.pathsep + os.environ.get("PATH", "")}


def _posix(p) -> str:
    s = str(p).replace("\\", "/")
    return f"/{s[0].lower()}{s[2:]}" if os.name == "nt" and s[1:2] == ":" else s


@pytest.mark.skipif(_git_bash() is None, reason="no Git Bash / bash to run the scratch script")
def test_the_scratch_script_empties_new_entries_and_keeps_the_baseline(tmp_path):
    """P2 (audit: /tmp, /var/tmp, /dev/shm persist across steps; a secret can be staged in
    halves): the post-exec check deletes every scratch entry that was not there when the
    container started (a name with a space, a nested folder) and lists what is left. Run with
    Git Bash on Windows (bash 5, coreutils, findutils; the container's bash is 5.2). Newline
    names and a chmod-000 folder can't exist on NTFS: tests/test_shell_container.py has them."""
    bash = _git_bash()
    tmp_a, run_b = tmp_path / "tmp", tmp_path / "run"
    tmp_a.mkdir()
    run_b.mkdir()
    (run_b / ".containerenv").write_text("engine", encoding="utf-8")  # made by the runtime
    (tmp_a / "half1").write_text("sk-", encoding="utf-8")
    (tmp_a / "a b.txt").write_text("x", encoding="utf-8")
    (tmp_a / "deep" / "er").mkdir(parents=True)
    (tmp_a / "deep" / "er" / "half2").write_text("123", encoding="utf-8")
    (run_b / "new").write_text("x", encoding="utf-8")
    dirs = [_posix(tmp_a), _posix(run_b), _posix(tmp_path / "missing")]
    keep = {f"{_posix(run_b)}/.containerenv"}
    script = sc.scratch_script(dirs, keep)
    r = subprocess.run([bash, "-c", script], capture_output=True, text=True, timeout=60, env=_bash_env(bash))
    assert r.returncode == 0, r.stderr
    listed = sc.parse_list(sc.parse_integrity(f"{sc.INTEGRITY_TAG}\n{r.stdout}")["LIST"])
    assert listed == keep, listed
    assert sorted(os.listdir(tmp_a)) == [] and sorted(os.listdir(run_b)) == [".containerenv"]


@pytest.mark.skipif(_git_bash() is None, reason="no Git Bash / bash to run the scratch script")
def test_the_scratch_script_in_list_mode_deletes_nothing(tmp_path):
    bash = _git_bash()
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "x").write_text("x", encoding="utf-8")
    r = subprocess.run([bash, "-c", sc.scratch_script([_posix(tmp_path / "t")], None)],
                       capture_output=True, text=True, timeout=60, env=_bash_env(bash))
    listed = sc.parse_list(sc.parse_integrity(f"{sc.INTEGRITY_TAG}\n{r.stdout}")["LIST"])
    assert listed == {f"{_posix(tmp_path / 't')}/x"} and (tmp_path / "t" / "x").exists()


def test_a_scratch_entry_left_after_cleaning_restarts_the_container(tmp_path, monkeypatch):
    """The baseline is what the fresh container had; anything else still there after the clean
    (the model made it undeletable, or the clean failed) replaces the container."""
    sh, rt = _shell(tmp_path, monkeypatch)
    rt.integrity = lambda s: ok_integrity(s, listed=["/run/.containerenv"])
    sh.start()
    first = sh.cid
    rt.integrity = lambda s: (ok_integrity(s, listed=["/run/.containerenv", "/tmp/stash"]) if s.cid == first
                              else ok_integrity(s, listed=["/run/.containerenv"]))
    sh.run("true", 30)
    assert sh.cid != first and any("scratch" in why for why in sh.restarts), sh.restarts
    # and the clean of the next check keeps the new container's own baseline
    assert any("/run/.containerenv" in c[-1] for c in rt.calls if c[1] == "exec" and sc.INTEGRITY_TAG in c[-1])


# --- Secret masks (decision 3: never trust the model; audit G-4) ---------------------------------

SEED = {".env": "API_KEY=sk-" + "mask-test-0123456789\n", "config/.env.local": "DB=pw-mask-test-9876\n",
        "secret.txt": "canary-mask-test\n", "secrets/k": "s\n", "id_rsa": "KEY\n",
        "notes.txt": "hello\n", "docs/secrets_management.md": "# how we keep secrets\n",
        ".env.example": "API_KEY=\n"}


def _seed(work, files=SEED):
    for rel, text in files.items():
        p = work / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


def _masks(argv) -> dict[str, tuple[str, str]]:
    """dest -> (kind, source/options) for every mask after the /work bind."""
    i = argv.index("-v")
    out = {}
    rest = argv[i + 2:argv.index("-w")]
    for flag, val in zip(rest[::2], rest[1::2]):
        if flag == "--tmpfs":
            dest, opts = val.split(":", 1)
            out[dest] = ("tmpfs", opts)
        else:
            src, dest, mode = val.rsplit(":", 2)
            out[dest] = ("bind:" + mode, src)
    return out


def test_secret_files_present_at_start_get_read_only_masks(tmp_path):
    """Decision 3 / G-4: every secret-named entry in work/ at start is covered inside the
    container — a file by a read-only bind of a notice file, a folder by an empty read-only
    tmpfs — so bash can't read it whatever the model does. Normal files are not masked."""
    work = tmp_path / "work"
    _seed(work)
    sh = ContainerShell(ContainerConfig(), work)
    masks = _masks(sh.run_argv())
    for rel in (".env", "config/.env.local", "secret.txt", "id_rsa"):
        kind, src = masks[f"/work/{rel}"]
        assert kind == "bind:ro", rel
        body = open(src, encoding="utf-8").read()
        assert body == sc.MASK_NOTICE and "mask-test" not in body
        assert not os.path.abspath(src).startswith(str(work))
    assert masks["/work/secrets"] == ("tmpfs", "ro,notmpcopyup")  # podman: no copy-up of the folder
    for rel in ("notes.txt", "docs/secrets_management.md", "secrets/k"):
        assert f"/work/{rel}" not in masks, rel   # secrets/k is under the masked folder
    sh.close()


@pytest.mark.parametrize("profile", ["core4", "core4_product"])
def test_masks_are_on_in_both_the_exam_and_the_product_profile(tmp_path, profile):
    """H1-K K6 D2 (owner decision): secret masks are on in BOTH container profiles, exam and
    product, so the exam measures the sandbox the product ships. Checked on the run argv each
    profile really builds, not only on the flag. (G4 records any privacy drop this causes by name.)"""
    prof = get_profile(profile)
    assert prof.tools.shell_mode == "container" and prof.container.secret_masks is True
    work = tmp_path / "work"
    _seed(work)
    sh = ContainerShell(prof.container, work)
    masks = _masks(sh.run_argv())
    assert masks["/work/.env"][0] == "bind:ro" and masks["/work/secrets"][0] == "tmpfs", masks
    sh.close()


SHARED_NAME_CASES = [".env", ".env.local", "PROD.ENV", "secret.txt", "secrets", "credentials.json",
                     "id_ed25519", "server.pem", "site.key", ".netrc", ".git-credentials",
                     ".env.example", ".env.sample", "id_rsa.pub", "config.template",  # templates: not masked
                     "secrets_management.md", "notes.txt", "env.py", "keys.md"]


def test_secret_masks_use_the_shared_name_list(tmp_path):
    """H1-I item 5: the masks use Track G's list (``localbot.guard.secret_names``), the same one
    the guard indexes with; templates (``.env.example``) are not secrets, so not masked."""
    import localbot.config as config
    from localbot.guard import secret_names as sn
    assert not hasattr(config, "SECRET_NAME_GLOBS_STUB")
    cfg = ContainerConfig()
    assert cfg.secret_globs == sn.SECRET_GLOBS and cfg.secret_templates == sn.TEMPLATE_GLOBS
    for name in SHARED_NAME_CASES:
        assert sc.is_secret_entry(name, cfg) == sn.DEFAULT.matches(name), name
    assert sc.is_secret_entry("server.pem", cfg) and not sc.is_secret_entry(".env.example", cfg)


def test_a_template_next_to_a_secret_is_not_masked(tmp_path):
    work = tmp_path / "work"
    _seed(work, {".env": "K=v\n", ".env.example": "K=\n", "deploy/server.pem": "-----\n"})
    rels = [rel for rel, _d in sc.secret_entries(work, ContainerConfig())]
    assert rels == [".env", "deploy/server.pem"]


@pytest.mark.skipif(os.name != "nt", reason="NTFS case-insensitive names: Windows only")
def test_case_spellings_of_a_short_secret_name_are_masked_too(tmp_path):
    """On NTFS ``.ENV`` opens ``.env``: a mask on the exact name alone is bypassed by a case
    spelling (S8e.1 found this for .git). A short name gets all its spellings."""
    work = tmp_path / "work"
    _seed(work, {".env": "K=v\n"})
    masks = _masks(ContainerShell(ContainerConfig(), work).run_argv())
    assert {"/work/.env", "/work/.ENV", "/work/.Env", "/work/.eNv"} <= set(masks)
    for dest in masks:  # every mask lands on the real file, never creates an entry
        assert os.path.samefile(work / dest[len("/work/"):], work / ".env")


def test_spellings_are_bounded_for_long_names(tmp_path):
    work = tmp_path / "work"
    _seed(work, {"credentials.json": "{}\n"})
    masks = _masks(ContainerShell(ContainerConfig(), work).run_argv())
    assert "/work/credentials.json" in masks
    assert len(masks) <= sc.MAX_SPELLINGS_LONG + 2 * sc.MAX_SPELLINGS_LONG  # name + its 8.3 name


def test_secret_masks_can_be_turned_off(tmp_path):
    work = tmp_path / "work"
    _seed(work)
    assert _masks(ContainerShell(ContainerConfig(secret_masks=False), work).run_argv()) == {}


def test_too_many_secret_entries_fail_closed(tmp_path):
    work = tmp_path / "work"
    _seed(work, {f"s{i}/.env": "K=v\n" for i in range(5)})
    with pytest.raises(ContainerError, match="too many"):
        ContainerShell(ContainerConfig(max_masks=3), work).run_argv()


def test_d4_secrets_and_credentials_with_any_extension_are_masked_a_public_key_is_not(tmp_path):
    """H1-K K6 D4 (owner decision, fail toward hiding): ``secrets.*`` and ``credentials.*`` are
    masked whatever the extension; ``id_rsa.pub`` (a public key) stays readable."""
    work = tmp_path / "work"
    _seed(work, {"secrets.cfg": "k=v\n", "conf/credentials.ini": "k=v\n", "secrets.yaml": "k: v\n",
                 "credentials.py": "TOKEN = 'x'\n", "id_rsa.pub": "ssh-ed25519 AAAA public\n", "app.py": "x\n"})
    masks = _masks(ContainerShell(ContainerConfig(), work).run_argv())
    for rel in ("secrets.cfg", "conf/credentials.ini", "secrets.yaml", "credentials.py"):
        assert masks.get(f"/work/{rel}", ("",))[0] == "bind:ro", (rel, sorted(masks))
    for rel in ("id_rsa.pub", "app.py"):
        assert f"/work/{rel}" not in masks, rel


def test_c_defaults_are_kept():
    """H1-K K6 D3 (owner decision): Track C's defaults stay: 2 h lifetime, 4 h legacy sweep,
    +2 GiB / +50,000 entries of growth, 256 masks."""
    c = ContainerConfig()
    assert (c.max_lifetime_s, c.legacy_stale_s, c.work_growth_max_bytes, c.work_growth_max_entries, c.max_masks) == (
        2 * 3600, 4 * 3600, 2 * 1024**3, 50_000, 256)


def test_300_cert_and_key_files_stop_the_container_before_it_starts(tmp_path, monkeypatch):
    """D3: a cert-heavy project (*.pem / *.key are secret names) needs more masks than the cap:
    the container is not started, and the message gives the count and the cap."""
    sh, rt = _shell(tmp_path, monkeypatch)
    for i in range(150):
        _seed(sh.work, {f"certs/c{i}.pem": "x\n", f"keys/k{i}.key": "x\n"})
    with pytest.raises(ContainerError) as e:
        sh.start()
    assert re.search(r"\b300 masks needed\b", str(e.value)) and "max_masks 256" in str(e.value), e.value
    assert rt.calls == [], rt.calls  # nothing reached the runtime: no container was made


def test_the_refused_mask_count_includes_the_git_masks(tmp_path):
    """The cap counts .git masks too; the message names the total it compared, not only the
    secret entries (256 secrets + a .git used to read '256 (max_masks 256)')."""
    work = tmp_path / "work"
    _seed(work, {f"s{i}.pem": "x\n" for i in range(256)})
    (work / ".git").mkdir()
    with pytest.raises(ContainerError) as e:
        ContainerShell(ContainerConfig(), work).run_argv()
    m = re.search(r"\b(\d+) masks needed \((\d+) secret entries \+ (\d+) for \.git\), max_masks 256", str(e.value))
    assert m, e.value
    total, secrets, git = map(int, m.groups())
    assert secrets == 256 and git >= 1 and total == secrets + git > 256


def test_a_secret_file_added_later_is_masked_at_the_next_start(tmp_path, monkeypatch):
    sh, rt = _shell(tmp_path, monkeypatch)
    sh.start()
    assert "/work/.env" not in sh.mask_dests()
    _seed(sh.work, {".env": "K=v\n"})
    sh._restart("test")
    assert "/work/.env" in sh.mask_dests()


def test_a_mask_missing_from_the_container_restarts_it(tmp_path, monkeypatch):
    """The post-exec check reads /proc/self/mountinfo: every mask must still be mounted (a mount
    on a 9p dentry can be dropped when the host file changes under it)."""
    sh, rt = _shell(tmp_path, monkeypatch)
    _seed(sh.work, {".env": "K=v\n"})
    sh.start()
    first = sh.cid
    rt.integrity = lambda s: (ok_integrity(s, dests=[d for d in s.mask_dests() if d != "/work/.env"])
                              if s.cid == first else ok_integrity(s))
    sh.run("true", 30)
    assert sh.cid != first and any("mask" in why for why in sh.restarts), sh.restarts


def test_a_masked_secret_changed_on_the_host_restarts_before_the_next_command(tmp_path, monkeypatch):
    """A host-side change to a masked file (an undo restore rewrites it) may drop the mount: the
    container is replaced before the next command runs, so that command is masked again."""
    sh, rt = _shell(tmp_path, monkeypatch)
    _seed(sh.work, {".env": "K=v\n"})
    sh.start()
    first = sh.cid
    (sh.work / ".env").unlink()
    (sh.work / ".env").write_text("K=v2\n", encoding="utf-8")
    sh.run("cat .env", 30)
    cmd_exec = next(c for c in rt.calls if c[1] == "exec" and c[-1] == "cat .env")
    assert cmd_exec[4] != first and any("masked" in why for why in sh.restarts), sh.restarts


def _nested_repos(work):
    """The F3 tree: a top-level repo and a nested one (``sub/.git``), both masked at start."""
    (work / ".git").mkdir()
    (work / "sub" / ".git").mkdir(parents=True)


def _swapped(sh, **kw):
    """What a container answers once ``mv sub sub2 && ln -s sub2 sub`` ran: the old one's masks under
    ``sub/`` moved with the folder, a fresh one mounts them through the link (H1-L L2)."""
    return ok_integrity(sh, dests=[d for d in sh.mask_dests() if not d.startswith("/work/sub/")], **kw)


def test_l2_a_restart_that_cannot_place_its_masks_leaves_no_container_and_runs_nothing(tmp_path, monkeypatch):
    """H1-L L2 (F3): a parent swap moves the nested ``.git`` masks; the reap restarts the container,
    and the fresh one fails its own check ("2 masks not mounted"). That raised out of the step, so the
    loop never undid it. Now the step returns with no container left (nothing runs, so the host may
    undo the step and restore ``.git``), and no command runs until a fresh container confirms every
    mask (fail closed)."""
    sh, rt = _shell(tmp_path, monkeypatch)
    _nested_repos(sh.work)
    sh.start()
    assert any(d.startswith("/work/sub/") for d in sh.mask_dests())  # precondition
    rt.integrity = _swapped
    out = sh.run("mv sub sub2 && ln -s sub2 sub", 30)
    assert out.startswith("ok\n") and sh.cid is None and sh.alive_pids() == []
    assert any("masks not mounted" in why for why in sh.restarts), sh.restarts
    with pytest.raises(ContainerError, match="masks not mounted"):
        sh.run("cat sub/.git/config", 30)  # still swapped: the next fresh container fails too
    assert not any(c[1] == "exec" and c[-1] == "cat sub/.git/config" for c in rt.calls)  # never ran
    rt.integrity = ok_integrity  # the undo put sub/ back: every mask mounts again
    assert sh.run("ls", 30) == "ok\n"
    assert next(c for c in rt.calls if c[1] == "exec" and c[-1] == "ls")[4] == sh.cid


def test_l2_survivors_then_masks_it_cannot_place_also_leave_no_container(tmp_path, monkeypatch):
    """The reap's other restart (processes survived the kill) takes the same way: the old container
    is removed with its survivors, the fresh one can't place its masks, the step returns."""
    sh, rt = _shell(tmp_path, monkeypatch)
    _nested_repos(sh.work)
    sh.start()
    first, real = sh.cid, sh.alive_pids
    monkeypatch.setattr(sh, "alive_pids", lambda: [99] if sh.cid == first else real())
    rt.integrity = _swapped
    sh.run("mv sub sub2 && ln -s sub2 sub && sleep 99 &", 30)
    assert sh.cid is None and real() == [] and first in rt.removed()
    assert any("survivors" in why for why in sh.restarts), sh.restarts


def test_l2_a_fresh_container_that_fails_another_check_still_raises(tmp_path, monkeypatch):
    """Only a fresh container whose sole problem is its masks leaves the step standing for the
    loop's undo; a pid-1 problem as well still raises out of the step, as before."""
    sh, rt = _shell(tmp_path, monkeypatch)
    _nested_repos(sh.work)
    sh.start()
    rt.integrity = lambda s: _swapped(s, tasks=2)
    with pytest.raises(ContainerError, match="pid 1"):
        sh.run("true", 30)


# --- S-4: a /work quota, checked before the next snapshot ----------------------------------------


def test_work_usage_counts_bytes_and_entries_and_stops_early(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "f.bin").write_bytes(b"x" * 1000)
    (tmp_path / "g.txt").write_bytes(b"y" * 10)
    assert sc.work_usage(tmp_path) == (1010, 3)
    for i in range(50):
        (tmp_path / f"n{i}").write_bytes(b"")
    _b, entries = sc.work_usage(tmp_path, stop_entries=10)
    assert 10 < entries < 53  # stopped once over the limit, not after the whole tree


def _grow(rt, work, files: int, size: int):
    """Make the scripted exec write into work/ (what a model command would do)."""
    def run_capped(argv, timeout, limit, watch=None, poll_s=None):
        rt.calls.append(list(argv))
        for i in range(files):
            (work / f"flood_{len(rt.calls)}_{i}.bin").write_bytes(b"z" * size)
        return f"{argv[argv.index('_') + 1]}\n", False
    return run_capped


def test_a_step_past_the_byte_quota_ends_the_task(tmp_path, monkeypatch):
    """S-4 (audit: no size quota on /work; the container can fill the host disk, and the next
    snapshot copies it all): growth since the shell's first start is capped; over it, the
    command's run raises (the task ends) before any snapshot of the flood is taken."""
    sh, rt = _shell(tmp_path, monkeypatch, work_growth_max_bytes=5000, work_growth_max_entries=1000)
    sh.start()
    monkeypatch.setattr(sc, "_run_capped", _grow(rt, sh.work, 3, 4000))
    with pytest.raises(sc.WorkQuotaError, match="bytes"):
        sh.run("dd if=/dev/zero of=big", 30)


def test_a_step_past_the_entry_quota_ends_the_task(tmp_path, monkeypatch):
    sh, rt = _shell(tmp_path, monkeypatch, work_growth_max_bytes=10**9, work_growth_max_entries=20)
    sh.start()
    monkeypatch.setattr(sc, "_run_capped", _grow(rt, sh.work, 40, 1))
    with pytest.raises(sc.WorkQuotaError, match="entries"):
        sh.run("touch many", 30)


def test_a_step_within_the_quota_runs_normally(tmp_path, monkeypatch):
    sh, rt = _shell(tmp_path, monkeypatch, work_growth_max_bytes=5000, work_growth_max_entries=20)
    sh.start()
    monkeypatch.setattr(sc, "_run_capped", _grow(rt, sh.work, 2, 100))
    assert sh.run("small", 30) == ""


def test_the_quota_baseline_is_the_first_start_not_a_restart(tmp_path, monkeypatch):
    """A restart (forced by a fork bomb, say) must not re-baseline the quota at the flood."""
    sh, rt = _shell(tmp_path, monkeypatch, work_growth_max_bytes=5000, work_growth_max_entries=1000)
    sh.start()
    (sh.work / "flood.bin").write_bytes(b"z" * 4000)
    sh._restart("test")
    monkeypatch.setattr(sc, "_run_capped", _grow(rt, sh.work, 1, 2000))
    with pytest.raises(sc.WorkQuotaError):
        sh.run("more", 30)


def test_the_watcher_stops_a_running_command_once_work_is_over_quota(tmp_path):
    """The check after the command is not enough on its own: in 30 s a command can write tens of
    GB. ``_run_capped`` polls ``watch`` while it waits and kills the command when it says stop.
    A real host process writing files, a real watch on the folder."""
    script = ("import sys, time\n"
              "for i in range(100000):\n"
              "    open(f'{sys.argv[1]}/f{i}', 'w').write('x' * 1000)\n"
              "    time.sleep(0.01)\n")
    t0 = time.monotonic()
    out, timed_out = sc._run_capped([sys.executable, "-c", script, str(tmp_path)], 60, 4000,
                                    watch=lambda: sc.work_usage(tmp_path, stop_entries=50)[1] > 50,
                                    poll_s=0.2)
    took = time.monotonic() - t0
    assert not timed_out and took < 20, took
    assert 50 < len(os.listdir(tmp_path)) < 2000


# --- G2: the new cases are registered (they run only in the gate run, under the lock) ------------


def test_g2_registers_the_h1_cases():
    """Import-level check only (no runtime is called at import): every H1 case is in CASES."""
    from scripts import g2_run
    names = [n for n, _ in g2_run.CASES]
    assert len(names) == len(set(names)) == 16 + 13
    for n in ("pid1_ignores_planted_modules", "user_signal_py_project_starts", "big_same_size_secret_masked",
              "big_same_size_secret_unmasked", "disk_fill_stopped_by_quota", "file_flood_stopped_by_quota",
              "killed_runner_leaves_nothing", "two_concurrent_runs", "var_tmp_split_secret",
              "nested_git_case_variant", "ipv6_and_host_gateway", "proxy_env_not_copied",
              "secret_masked_g4_encodings"):
        assert n in names, n


def test_a_container_near_the_end_of_its_lifetime_is_renewed_before_the_next_command(tmp_path, monkeypatch):
    """P2 follow-through: pid 1 now exits at max_lifetime_s, so a long session must not hit a
    dead container mid-task: with less than the command's timeout + 60 s left, the container is
    replaced first and the command runs on the new one."""
    clock = [1_000_000.0]
    monkeypatch.setattr(sc.time, "time", lambda: clock[0])
    sh, rt = _shell(tmp_path, monkeypatch, max_lifetime_s=600)
    sh.start()
    first = sh.cid
    clock[0] += 520  # 80 s left < 30 + 60
    sh.run("echo hi", 30)
    cmd = next(c for c in rt.calls if c[1] == "exec" and c[-1] == "echo hi")
    assert cmd[4] != first and any("lifetime" in why for why in sh.restarts), sh.restarts
    clock[0] += 10
    sh.run("echo again", 30)  # the renewed container has its full lifetime again
    assert len(sh.restarts) == 1


def test_shell_container_imports_no_private_name_from_workspace_fs():
    """H1-I item 5 (C req 4): the short-name helper is public, so a rename in workspace_fs can't
    silently break the masks."""
    import ast
    from pathlib import Path
    tree = ast.parse(Path(sc.__file__).read_text(encoding="utf-8"))
    private = [a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == "localbot.guard.workspace_fs"
               for a in n.names if a.name.startswith("_")]
    assert private == []


# --- H1-I item 8: moved here from the container files (no runtime needed) ----------------------


def test_parse_proc_list():
    from localbot.tools.shell_container import parse_proc_list
    text = "SELF=40\n1 0 S python3\n40 0 R bash\n77 1 Z sleep\n81 1 S x) y (z\n82 1 S a?b) c\n"
    assert parse_proc_list(text) == (40, [(1, 0, "S", "python3"), (40, 0, "R", "bash"), (77, 1, "Z", "sleep"),
                                          (81, 1, "S", "x) y (z"), (82, 1, "S", "a?b) c")])
    assert parse_proc_list("1 0 S python3\n") is None  # no SELF line: unusable, not "clean"
    # S7b.0.7: ONE line that doesn't parse (e.g. the second half of a name split by a newline)
    # makes the whole list None -> not clean -> the reaper restarts the container.
    assert parse_proc_list("SELF=40\n1 0 S python3\n83 1 S a\nb) c\n") is None
    assert parse_proc_list("SELF=40\n1 0 S python3\ngarbage\n") is None
    assert parse_proc_list("SELF=40\n1 0 S py\rthon\n") == (40, [(1, 0, "S", "py\rthon")])  # \r stays in the name
    assert parse_proc_list("SELF=40\n1 0 9 x\n") is None  # a state must be a letter


def test_keepalive_exits_nonzero_when_prctl_fails(monkeypatch):
    """S8a (unit): the KEEPALIVE script checks the prctl return value — a non-zero result must
    exit non-zero (so the container fails to start) instead of looping on as dumpable pid 1."""
    import ctypes
    import signal

    class FakePrctl:
        def prctl(self, *args):
            return -1

    monkeypatch.setattr(signal, "signal", lambda *a, **k: None)   # don't reset SIGINT in the test
    monkeypatch.setattr(ctypes, "CDLL", lambda name=None: FakePrctl())
    with pytest.raises(SystemExit) as exc:
        exec(KEEPALIVE, {"__name__": "__main__"})
    assert exc.value.code == 1


def test_bot_image_is_not_the_git_test_image():
    assert get_profile("core4").container.image != GIT_IMAGE
    assert "@sha256:" in get_profile("core4").container.image


# --- H1-J J3: the emulated shell's `cat` is the lab's, byte for byte ------------------------------

EMU_SEEDS = {
    "plain": {"a.txt": "alpha\n", "dir/b.txt": "beta\n"},
    "env_and_dotenv": {".env": "API_KEY=dot\n", "env": "OTHER=plain\n"},  # lstrip("./"): .env -> env
    "dotenv_only": {".env": "API_KEY=dot\n"},
    "empty_file": {"a.txt": "", "b.txt": "x\n"},                            # the lab's `or` falls through
    "hidden_dir": {".cfg/x.txt": "x\n", "cfg/x.txt": "y\n"},
}
EMU_CMDS = ["ls", "ls -la", "ls *", "cat a.txt", "cat ./a.txt", "cat .//a.txt", "cat .\\a.txt", "cat .env",
            "cat ./.env", "cat env", "cat ...env", "cat .cfg/x.txt", "cat ./cfg/x.txt", "cat dir/b.txt",
            "cat dir\\b.txt", "cat missing.txt", "cat ../x", "cat /etc/passwd", "cat C:/x", "cat ~/x",
            "curl http://x", "", "   ", "echo hi", "cat a.txt b.txt"]


@pytest.mark.parametrize("seed", sorted(EMU_SEEDS))
def test_emulated_bash_answers_exactly_as_the_lab(disk, seed):
    """J3 (parity): ``shell_emulated.py`` keeps the lab's ``lstrip("./")`` only because the parity
    profile must answer as the lab did. Every command here, on a disk workspace, must give the
    same (handled, text) as the lab's own ``sandbox_bash`` (compiled from lab_ref) on a dict."""
    lab = lab_module("sandbox_bash").sandbox_bash
    files = EMU_SEEDS[seed]
    ws = disk.ws(files)
    for cmd in EMU_CMDS:
        assert sandbox_bash(cmd, ws) == lab(cmd, dict(files)), (seed, cmd)


def test_emulated_cat_dotenv_reads_env_when_both_exist_as_the_lab_does(disk):
    """J3 probe, stated: ``lstrip`` strips characters, so ``cat .env`` looks up ``env`` first and,
    when both exist, shows ``env``. A false match, not a bypass (the guard sees the raw command
    before this; the shell reads only inside work/), and the lab's own answer: kept for parity."""
    ws = disk.ws({".env": "A=1\n", "env": "B=2\n"})
    assert sandbox_bash("cat .env", ws) == (True, "B=2\n")
    assert sandbox_bash("cat .nothere", ws) == (True, "ERROR: not found: nothere")  # the stripped name


# --- H1-J J6: the git test image is built from a pinned base and run by its id -------------------


def test_git_test_image_base_is_pinned_by_digest():
    """Already true before J6 (a guard, not a red): every FROM names its base by digest."""
    froms = [ln.split()[1] for ln in images.CONTAINERFILE.read_text(encoding="utf-8").splitlines()
             if ln.upper().startswith("FROM ")]
    assert froms and all(re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", f) for f in froms), froms


@pytest.mark.parametrize("runtime,pull", [("podman", "--pull=never"), ("docker", "--pull=false")])
def test_git_test_image_build_is_offline(runtime, pull):
    argv = images.build_argv(runtime)
    assert argv[:5] == [runtime, "build", pull, "--network", "none"] and "-q" in argv
    assert argv[-3:] == ["-f", str(images.CONTAINERFILE), str(images.CONTAINERFILE.parent)]


@pytest.mark.parametrize("out,want", [
    ("sha256:" + "a" * 64 + "\n", "sha256:" + "a" * 64),   # docker build -q
    ("b" * 64 + "\n", "b" * 64),                           # podman build -q
    ("note\n" + "c" * 64 + "\n", "c" * 64),
    ("c" * 12 + "\n", None),                               # a short id is not a pin
    ("", None),
    ("localhost/localbot-test-git:s8d\n", None),           # a tag is not a pin
])
def test_the_built_image_is_named_by_its_full_id(out, want):
    assert images.image_id(out) == want


def test_build_git_image_returns_the_id_not_the_tag():
    """J6 (audit P2: test image by tag): the container tests run the image the build just made,
    by its id, so a re-tag or another build under the same tag can't change what they test."""
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "sha256:" + "d" * 64 + "\n", "")

    assert images.build_git_image("docker", run=run) == "sha256:" + "d" * 64
    assert calls == [images.build_argv("docker")]


@pytest.mark.parametrize("rc,out,built", [(0, "", True), (0, "localhost/x:tag\n", True), (1, "", False)])
def test_build_git_image_without_a_full_id_raises(rc, out, built):
    def run(argv, **kw):
        return subprocess.CompletedProcess(argv, rc, out, "Error: base image missing")

    with pytest.raises(images.ImageBuildError) as e:
        images.build_git_image("podman", run=run)
    assert e.value.built is built


# --- H1-J J7a: no personal name in Track C's files ------------------------------------------------

C_FILES = ["localbot/tools/shell_container.py", "localbot/tools/shell_emulated.py", "tests/images/__init__.py",
           "tests/images/git/Containerfile", "tests/test_sandbox_host.py", "tests/test_shell_container.py",
           "tests/test_undo_container.py", "tests/test_delete_container.py"]
_OWNER = "pao" + "lo"  # in two parts, so this file does not match itself


def test_track_c_files_name_no_person():
    hits = [f"{rel}:{i}" for rel in C_FILES
            for i, line in enumerate((_REPO / rel).read_text(encoding="utf-8").splitlines(), 1)
            if _OWNER in line.lower()]
    assert hits == []
