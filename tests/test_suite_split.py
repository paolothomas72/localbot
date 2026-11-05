"""S8i A2: the suite split (tests/conftest.py + tests/runtimes.py), with no real podman / docker.

G1 runs as three jobs: ``-m "not container"`` (host, may use ``-n auto``), then ``-m container``
with ``LOCALBOT_TEST_RUNTIMES=podman`` and again with ``=docker``. The three job counts must add
up to the unfiltered total. With ``LOCALBOT_REQUIRE_RUNTIMES`` set, a runtime that is down stops
the session and a container test that would skip fails instead (audit E-4).

The unit tests patch the probe. The end-to-end tests run a small inner pytest in a subprocess
over a temp folder: a fake container file shaped like the real ones (its own probe, fixture
params, parametrize, one test on ``RUNTIMES[0]``), a host file, and a conftest that re-exports the
real hooks and swaps ``tests.runtimes.runtime_up`` for a fake that logs every probe.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import runlock
from tests import runtimes

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def probes(monkeypatch):
    """A fake runtime_up (podman and docker up unless the test changes ``up``) and fresh caches."""
    seen: list[str] = []
    state = SimpleNamespace(up={"podman", "docker"}, calls=seen)

    def fake(exe):
        seen.append(exe)
        return exe in state.up

    monkeypatch.delenv(runtimes.TEST_ENV, raising=False)
    monkeypatch.delenv(runtimes.REQUIRE_ENV, raising=False)
    monkeypatch.setattr(runtimes, "runtime_up", fake)
    runtimes.reset()
    yield state
    runtimes.reset()  # nothing faked stays cached for later tests


# --- tests/runtimes.py: which runtimes ---------------------------------------------------------


@pytest.mark.parametrize("raw,want", [
    (None, ("podman", "docker")), ("", ("podman", "docker")), ("  ", ("podman", "docker")),
    ("podman", ("podman",)), ("docker", ("docker",)),
    ("podman,docker", ("podman", "docker")), ("docker|podman", ("podman", "docker")),
    (" Docker , podman ", ("podman", "docker")), ("podman,podman", ("podman",)),
])
def test_selected_runtimes(monkeypatch, raw, want):
    if raw is None:
        monkeypatch.delenv(runtimes.TEST_ENV, raising=False)
    else:
        monkeypatch.setenv(runtimes.TEST_ENV, raw)
    assert runtimes.selected_runtimes() == want


@pytest.mark.parametrize("raw,want", [(None, ()), ("", ()), ("podman,docker", ("podman", "docker")),
                                      ("docker", ("docker",)), ("podman|docker", ("podman", "docker"))])
def test_required_runtimes(monkeypatch, raw, want):
    if raw is None:
        monkeypatch.delenv(runtimes.REQUIRE_ENV, raising=False)
    else:
        monkeypatch.setenv(runtimes.REQUIRE_ENV, raw)
    assert runtimes.required_runtimes() == want


@pytest.mark.parametrize("env", ["LOCALBOT_TEST_RUNTIMES", "LOCALBOT_REQUIRE_RUNTIMES"])
@pytest.mark.parametrize("raw", ["podmn", "podman,dokcer", "podman docker", ",|"])
def test_an_unknown_runtime_name_is_an_error(monkeypatch, env, raw):
    monkeypatch.setenv(env, raw)
    with pytest.raises(ValueError, match=env):
        runtimes.selected_runtimes() if env == runtimes.TEST_ENV else runtimes.required_runtimes()


# --- tests/runtimes.py: the probe ----------------------------------------------------------------


def test_runtime_up_is_the_container_files_probe(monkeypatch):
    runs = []

    def run(argv, **kw):
        runs.append((argv, kw))
        return SimpleNamespace(returncode=0 if argv[0] == "podman" else 1)

    monkeypatch.setattr(runtimes.shutil, "which", lambda exe: f"C:/bin/{exe}.exe")
    monkeypatch.setattr(runtimes.subprocess, "run", run)
    assert runtimes.runtime_up("podman") is True
    assert runtimes.runtime_up("docker") is False
    assert [a for a, _ in runs] == [["podman", "info"], ["docker", "info"]]
    assert all(kw["timeout"] == 60 and kw["capture_output"] for _, kw in runs)


def test_runtime_up_is_false_when_missing_slow_or_broken(monkeypatch):
    def boom(exc):
        def run(argv, **kw):
            raise exc
        return run

    monkeypatch.setattr(runtimes.shutil, "which", lambda exe: None)
    monkeypatch.setattr(runtimes.subprocess, "run", boom(AssertionError("must not run without the exe")))
    assert runtimes.runtime_up("podman") is False
    monkeypatch.setattr(runtimes.shutil, "which", lambda exe: f"C:/bin/{exe}.exe")
    monkeypatch.setattr(runtimes.subprocess, "run", boom(subprocess.TimeoutExpired("podman info", 60)))
    assert runtimes.runtime_up("podman") is False
    monkeypatch.setattr(runtimes.subprocess, "run", boom(OSError("cannot start")))
    assert runtimes.runtime_up("podman") is False


def test_runtimes_is_lazy_and_computed_once(probes):
    assert probes.calls == []  # the fixture's reset() asked nothing
    probes.up = {"docker"}
    assert runtimes.RUNTIMES == ["docker"]
    assert runtimes.RUNTIMES == ["docker"]
    from tests.runtimes import RUNTIMES  # the import the container files will use

    assert RUNTIMES == ["docker"]
    assert probes.calls == ["podman", "docker"]  # once per runtime, not once per use
    RUNTIMES.append("podman")
    assert runtimes.RUNTIMES == ["docker"]  # callers get a copy


def test_runtimes_probes_only_the_selected_runtimes(probes, monkeypatch):
    monkeypatch.setenv(runtimes.TEST_ENV, "docker")
    assert runtimes.RUNTIMES == ["docker"]
    assert probes.calls == ["docker"]
    assert runtimes.is_up("docker") and probes.calls == ["docker"]  # the cached answer


def test_an_unknown_module_attribute_is_still_an_error():
    with pytest.raises(AttributeError):
        runtimes.NOT_A_THING  # noqa: B018


def test_importing_runtimes_runs_no_probe():
    code = ("import shutil, subprocess, sys\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "def no(*a, **k):\n"
            "    raise AssertionError('probe ran at import')\n"
            "shutil.which = subprocess.run = no\n"
            "import tests.runtimes\n"
            "print('imported', tests.runtimes.CANDIDATES)\n")
    r = subprocess.run([sys.executable, "-I", "-c", code, str(REPO)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "imported ('podman', 'docker')" in r.stdout


# --- tests/runtimes.py: the helpers the conftest uses -------------------------------------------


NOTSET = object()  # what pytest puts in an empty parameter set


@pytest.mark.parametrize("params,module_runtimes,want", [
    ({"runtime": "docker"}, ["podman", "docker"], "docker"),           # @parametrize("runtime", RUNTIMES)
    ({"shell": "podman", "case": "x"}, ["podman", "docker"], "podman"),  # @fixture(params=RUNTIMES)
    ({"case": "docker-ish", "runtime": "docker"}, ["podman"], "docker"),
    ({}, ["podman", "docker"], "podman"),                               # not parametrized: RUNTIMES[0]
    ({}, ["docker"], "docker"),
    ({"case": "a"}, ["docker"], "docker"),
    ({"runtime": NOTSET}, [], None),                                    # empty parameter set
    ({}, [], None), ({}, None, None), ({"n": 3, "t": ("podman",)}, None, None),
])
def test_item_runtime(params, module_runtimes, want):
    assert runtimes.item_runtime(params, module_runtimes) == want


def test_an_unknown_item_belongs_to_the_podman_job_only():
    assert runtimes.job_runtime("docker") == "docker" and runtimes.job_runtime("podman") == "podman"
    assert runtimes.job_runtime(None) == "podman"  # counted once: in the P job, never in the D job


def test_missing_by_module():
    present = {"tests/test_a_container.py": {"podman", "docker"}, "tests/test_b_container.py": {"docker"},
               "tests/test_c_container.py": set()}
    assert runtimes.missing_by_module(present, ("podman",)) == {
        "tests/test_b_container.py": ["podman"], "tests/test_c_container.py": ["podman"]}
    assert runtimes.missing_by_module(present, ("docker",)) == {"tests/test_c_container.py": ["docker"]}
    assert runtimes.missing_by_module(present, ()) == {}


# --- end to end: the real hooks in an inner pytest -----------------------------------------------


FAKE_CONTAINER = '''\
"""Shaped like tests/test_*_container.py; its own probe reads FAKE_MODULE_UP, nothing runs a runtime."""
import os

import pytest

RUNTIMES = [r for r in ("podman", "docker") if r in os.environ.get("FAKE_MODULE_UP", "").split(",")]
pytestmark = pytest.mark.skipif(not RUNTIMES, reason="no podman/docker runtime available")


@pytest.fixture(params=RUNTIMES)
def shell(request):
    if request.param in os.environ.get("FAKE_SKIP", "").split(","):
        pytest.skip(f"test image not built on {request.param}")
    return request.param


def test_shell(shell):
    assert shell in ("podman", "docker")


@pytest.mark.parametrize("case", ["a", "b"])
@pytest.mark.parametrize("runtime", RUNTIMES)
def test_cases(runtime, case):
    assert runtime in ("podman", "docker")


def test_sweep():
    assert RUNTIMES[0] in ("podman", "docker")
'''

FAKE_HOST = '''\
def test_host_one():
    assert True


def test_host_two():
    assert True
'''

INNER_CONFTEST = '''\
import os

import tests.conftest as real
import tests.runtimes as runtimes
from tests.conftest import *  # noqa: F401,F403  the real hooks under test

# Which checkout this is decides the host-session lock; pin it so the answer is the same whether the
# outer suite runs in a worktree or in the main checkout.
real._in_main_checkout = lambda: os.environ.get("FAKE_MAIN_CHECKOUT") == "1"


def _fake_runtime_up(exe):
    with open(os.environ["FAKE_PROBE_LOG"], "a", encoding="utf-8") as f:
        f.write(exe + "\\n")
    return exe in os.environ.get("FAKE_UP", "").split(",")


runtimes.runtime_up = _fake_runtime_up
'''


@pytest.fixture
def inner(tmp_path):
    """Run an inner pytest over the fake project; returns rc, output, node ids, counts, probes."""
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")  # rootdir here, not the repo
    (proj / "conftest.py").write_text(INNER_CONFTEST, encoding="utf-8")
    (proj / "test_fake_container.py").write_text(FAKE_CONTAINER, encoding="utf-8")
    (proj / "test_fake_host.py").write_text(FAKE_HOST, encoding="utf-8")
    log = tmp_path / "probes.log"
    lock = tmp_path / "state" / "run.lock"

    def run(*args, **env):
        base = {k: v for k, v in os.environ.items()
                if not k.startswith("PYTEST_") and k not in (runtimes.TEST_ENV, runtimes.REQUIRE_ENV, runlock.HELD_ENV)}
        # The inner session's run lock is this test's own file, never the machine's (a container
        # session takes the lock).
        base.update(PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1", FAKE_PROBE_LOG=str(log),
                    FAKE_UP="podman,docker", FAKE_MODULE_UP="podman,docker", **{runlock.PATH_ENV: str(lock)})
        base.update(env)
        log.unlink(missing_ok=True)
        xdist = [] if "-n" in args else ["-p", "no:xdist"]
        r = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *xdist, "-q", *args],
                           cwd=proj, env=base, capture_output=True, text=True, encoding="utf-8", timeout=120)
        out = r.stdout + r.stderr
        ids = [ln.strip() for ln in r.stdout.splitlines() if "::" in ln and not ln.startswith(("ERROR", "FAILED"))]
        last = next((ln for ln in reversed(r.stdout.splitlines()) if ln.strip()), "")
        counts = {k.rstrip("s") if k == "errors" else k: int(n)
                  for n, k in re.findall(r"(\d+) (passed|failed|skipped|deselected|errors?)", last)}
        probed = log.read_text(encoding="utf-8").split() if log.exists() else []
        return SimpleNamespace(rc=r.returncode, out=out, ids=ids, counts=counts, probes=probed, last=last)

    return run


def test_marker_splits_host_from_container(inner):
    every = inner("--collect-only")
    cont = inner("-m", "container", "--collect-only")
    host = inner("-m", "not container", "--collect-only")
    assert len(every.ids) == 9, every.out
    assert len(cont.ids) == 7 and all(i.startswith("test_fake_container.py::") for i in cont.ids), cont.out
    assert host.ids == ["test_fake_host.py::test_host_one", "test_fake_host.py::test_host_two"], host.out
    assert sorted(cont.ids + host.ids) == sorted(every.ids)


def test_runtime_env_picks_each_jobs_tests(inner):
    pod = inner("-m", "container", "--collect-only", LOCALBOT_TEST_RUNTIMES="podman")
    dock = inner("-m", "container", "--collect-only", LOCALBOT_TEST_RUNTIMES="docker")
    assert pod.ids == ["test_fake_container.py::test_shell[podman]",
                       "test_fake_container.py::test_cases[podman-a]", "test_fake_container.py::test_cases[podman-b]",
                       "test_fake_container.py::test_sweep"], pod.out  # RUNTIMES[0] is podman: P job
    assert dock.ids == ["test_fake_container.py::test_shell[docker]",
                        "test_fake_container.py::test_cases[docker-a]",
                        "test_fake_container.py::test_cases[docker-b]"], dock.out
    run = inner("-m", "container", LOCALBOT_TEST_RUNTIMES="docker")
    assert run.rc == 0 and run.counts == {"passed": 3, "deselected": 6}, run.out


@pytest.mark.parametrize("module_up", ["podman,docker", "docker", ""])  # "": an empty parameter set
def test_host_plus_podman_plus_docker_is_the_unfiltered_total(inner, module_up):
    total = len(inner("--collect-only", FAKE_MODULE_UP=module_up).ids)
    h = len(inner("-m", "not container", "--collect-only", FAKE_MODULE_UP=module_up).ids)
    p = len(inner("-m", "container", "--collect-only", FAKE_MODULE_UP=module_up, LOCALBOT_TEST_RUNTIMES="podman").ids)
    d = len(inner("-m", "container", "--collect-only", FAKE_MODULE_UP=module_up, LOCALBOT_TEST_RUNTIMES="docker").ids)
    assert h == 2 and total > h
    assert (h, p, d, h + p + d) == (h, p, d, total)


def test_require_stops_the_session_when_a_tested_runtime_is_down(inner):
    r = inner("-m", "container", FAKE_UP="docker", LOCALBOT_REQUIRE_RUNTIMES="podman,docker",
              LOCALBOT_TEST_RUNTIMES="podman")
    assert r.rc == 1, r.out
    assert "LOCALBOT_REQUIRE_RUNTIMES" in r.out and "podman is not up" in r.out, r.out
    assert "passed" not in r.last and r.probes == ["podman"]


def test_require_stops_the_session_when_a_files_own_probe_disagrees(inner):
    r = inner("-m", "container", FAKE_MODULE_UP="docker", LOCALBOT_REQUIRE_RUNTIMES="podman,docker",
              LOCALBOT_TEST_RUNTIMES="podman")
    assert r.rc == 1, r.out
    assert "test_fake_container.py has no podman tests" in r.out, r.out
    assert "passed" not in r.last


def test_require_ignores_a_runtime_this_job_does_not_test(inner):
    r = inner("-m", "container", FAKE_UP="docker", FAKE_MODULE_UP="docker",
              LOCALBOT_REQUIRE_RUNTIMES="podman,docker", LOCALBOT_TEST_RUNTIMES="docker")
    assert r.rc == 0 and r.counts == {"passed": 4, "deselected": 2}, r.out  # test_sweep is docker's here
    assert r.probes == ["docker"]


def test_require_turns_a_fixture_skip_into_a_failure(inner):
    r = inner("-m", "container", "-rA", FAKE_SKIP="podman", LOCALBOT_REQUIRE_RUNTIMES="podman,docker",
              LOCALBOT_TEST_RUNTIMES="podman")
    assert r.rc == 1, r.out
    assert r.counts == {"passed": 3, "deselected": 5, "error": 1}, r.out
    assert "skip turned into a failure" in r.out and "test image not built on podman" in r.out, r.out
    assert "skipped" not in r.last


def test_require_fails_a_skipped_item_whose_runtime_is_unknown(inner):
    # REQUIRE names docker only, the P job tests podman: no runtime check applies, and the file's
    # own probe found nothing, so its items (unknown runtime, skipped by its skipif) must fail.
    r = inner("-m", "container", FAKE_MODULE_UP="", LOCALBOT_REQUIRE_RUNTIMES="docker",
              LOCALBOT_TEST_RUNTIMES="podman")
    assert r.rc == 1, r.out
    assert r.counts.get("error") == 4 and "skipped" not in r.last, r.out
    assert "no podman/docker runtime available" in r.out


def test_without_require_a_skip_stays_a_skip(inner):
    r = inner("-m", "container", FAKE_SKIP="podman", LOCALBOT_TEST_RUNTIMES="podman")
    assert r.rc == 0 and r.counts == {"passed": 3, "skipped": 1, "deselected": 5}, r.out
    assert "skip turned into a failure" not in r.out
    assert r.probes == []  # no REQUIRE: nothing is probed by the hooks


def test_a_host_only_session_never_probes(inner):
    r = inner("-m", "not container", LOCALBOT_REQUIRE_RUNTIMES="podman,docker", LOCALBOT_TEST_RUNTIMES="podman")
    assert r.rc == 0 and r.counts == {"passed": 2, "deselected": 7}, r.out
    assert r.probes == []
    r = inner("test_fake_host.py", LOCALBOT_REQUIRE_RUNTIMES="podman,docker")
    assert r.rc == 0 and r.counts == {"passed": 2} and r.probes == [], r.out


def test_a_bad_runtime_name_is_a_usage_error(inner):
    r = inner("--collect-only", LOCALBOT_TEST_RUNTIMES="podmn")
    assert r.rc == 4, r.out
    assert "LOCALBOT_TEST_RUNTIMES" in r.out and "podmn" in r.out, r.out


# --- the run lock (audit E-3): a session that runs container tests holds it -----------------------

LOCK_PROBE = '''\
"""A container-file name with no runtime parameter: it runs in the podman job and checks the lock."""
import os

from scripts import runlock


def test_the_session_holds_the_run_lock():
    h = runlock.holder()
    assert runlock.is_held() and h and h["pid"] == os.getpid() and h["name"] == "pytest", h
'''


@pytest.fixture
def lock_probe(tmp_path):
    """Adds the lock-probe file to the inner project; returns the inner session's lock path."""
    (tmp_path / "proj" / "test_lock_container.py").write_text(LOCK_PROBE, encoding="utf-8")
    return tmp_path / "state" / "run.lock"


def test_a_container_session_takes_the_run_lock_and_lets_it_go(inner, lock_probe):
    r = inner("test_lock_container.py")
    assert r.rc == 0 and r.counts == {"passed": 1}, r.out
    assert not runlock.is_held(lock_probe) and runlock.holder(lock_probe) is None


def test_a_container_session_is_refused_while_another_run_holds_it(inner, lock_probe):
    with runlock.acquire("g2", path=lock_probe) as held:
        r = inner("-m", "container")
    assert r.rc == runlock.BUSY_EXIT, r.out
    assert "busy" in r.out and f'"pid": {held.info["pid"]}' in r.out and '"name": "g2"' in r.out, r.out
    assert not re.search(r"\d+ (passed|failed)", r.last), r.out  # nothing ran


def test_host_only_and_collect_only_sessions_leave_the_lock_alone(inner, lock_probe):
    with runlock.acquire("g2", path=lock_probe):
        host = inner("-m", "not container")
        collect = inner("-m", "container", "--collect-only")
    assert host.rc == 0 and host.counts == {"passed": 2, "deselected": 8}, host.out
    assert collect.rc == 0 and len(collect.ids) == 8, collect.out


def test_a_parent_that_holds_the_lock_covers_the_session(inner, lock_probe):
    # watch_run / the GPU batch hold the lock and set the flag; the session must not fight them.
    with runlock.acquire("suite_podman", path=lock_probe):
        r = inner("-m", "container", "-k", "not holds_the_run_lock", **{runlock.HELD_ENV: "1"})
    assert r.rc == 0 and r.counts == {"passed": 7, "deselected": 3}, r.out  # 2 host + the probe


def test_a_stray_held_flag_with_a_free_lock_still_takes_it(inner, lock_probe):
    r = inner("test_lock_container.py", **{runlock.HELD_ENV: "1"})
    assert r.rc == 0 and r.counts == {"passed": 1}, r.out


def test_container_tests_are_refused_under_xdist(inner, lock_probe):
    # Workers would each sweep the others' containers (audit S-2); container jobs run serially.
    r = inner("-m", "container", "-n", "2", "-rE")
    assert r.rc == 1 and "INTERNALERROR" not in r.out, r.out
    assert "container tests don't run under xdist" in r.out, r.out
    assert r.counts == {"error": 8}, r.out  # every container test; xdist's summary has no deselected count
    assert not runlock.is_held(lock_probe)
    host = inner("-m", "not container", "-n", "2")
    assert host.rc == 0 and "2 passed" in host.last, host.out


# --- the main checkout: every session holds the lock (audit S-3) ---------------------------------
# Exams run in the main checkout, and a host test there adds a workspaces/_trials folder that a
# running exam's HostWatch counts as a host change. A worktree has its own _trials, so its host
# sessions leave the lock alone (tested above).

HOST_LOCK_PROBE = '''\
"""A host file: under xdist the controller holds the lock, so the holder may be an ancestor."""
from scripts import runlock


def test_the_session_holds_the_run_lock():
    h = runlock.holder()
    assert runlock.is_held() and h and h["name"] == "pytest", h
'''


@pytest.fixture
def host_lock_probe(tmp_path):
    (tmp_path / "proj" / "test_lock_host.py").write_text(HOST_LOCK_PROBE, encoding="utf-8")
    return tmp_path / "state" / "run.lock"


@pytest.mark.parametrize("xdist", [[], ["-n", "2"]], ids=["serial", "xdist"])
def test_a_main_checkout_host_session_holds_the_lock(inner, host_lock_probe, xdist):
    r = inner("test_lock_host.py", *xdist, FAKE_MAIN_CHECKOUT="1")
    assert r.rc == 0 and "1 passed" in r.last, r.out
    assert not runlock.is_held(host_lock_probe)
    worktree = inner("test_lock_host.py", *xdist)  # the same file in a worktree: no lock held
    assert worktree.rc == 1 and "1 failed" in worktree.last, worktree.out


@pytest.mark.parametrize("xdist", [[], ["-n", "2"]], ids=["serial", "xdist"])
def test_a_main_checkout_host_session_is_refused_while_another_run_holds_it(inner, host_lock_probe, xdist):
    with runlock.acquire("gpu batch", path=host_lock_probe):
        r = inner("-m", "not container", *xdist, FAKE_MAIN_CHECKOUT="1")
        collect = inner("-m", "not container", "--collect-only", FAKE_MAIN_CHECKOUT="1")
        covered = inner("-m", "not container", *xdist, FAKE_MAIN_CHECKOUT="1", **{runlock.HELD_ENV: "1"})
    assert r.rc == runlock.BUSY_EXIT and '"name": "gpu batch"' in r.out, r.out
    assert not re.search(r"\d+ (passed|failed)", r.last), r.out  # nothing ran
    assert collect.rc == 0, collect.out  # a dry run is never refused
    assert covered.rc == 1 and "2 passed" in covered.last, covered.out  # only the probe fails: held by "gpu batch"


# --- the header says which runtimes this session tests (a leftover env var is never silent) -------


def test_the_report_header_names_the_runtime_selection(inner):
    r = inner("--collect-only", "-v", LOCALBOT_TEST_RUNTIMES="docker")  # -v undoes the fixture's -q
    assert r.rc == 0, r.out
    assert "podman container tests are deselected" in r.out, r.out
    plain = inner("--collect-only", "-v")
    assert "container runtimes tested" not in plain.out, plain.out


@pytest.mark.parametrize("env,want", [
    ({}, []),
    ({"LOCALBOT_TEST_RUNTIMES": "docker"},
     ["container runtimes tested: docker (LOCALBOT_TEST_RUNTIMES=docker; podman container tests are deselected)"]),
    ({"LOCALBOT_REQUIRE_RUNTIMES": "podman,docker"},
     ["container runtimes tested: podman, docker",
      "required (a down runtime or a skip fails): podman, docker (LOCALBOT_REQUIRE_RUNTIMES)"]),
])
def test_report_header_lines(monkeypatch, env, want):
    from tests import conftest

    for k in (runtimes.TEST_ENV, runtimes.REQUIRE_ENV):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    split = (runtimes.selected_runtimes(), runtimes.required_runtimes())
    config = SimpleNamespace(stash={conftest._SPLIT: split})
    assert conftest.pytest_report_header(config) == want


# --- H1-I item 1: xdist needs the same test ids in every worker ---------------------------------


def _host_ids() -> list[str]:
    r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-m", "not container", "-p", "no:cacheprovider",
                        "-p", "no:xdist", "--collect-only", "-q"],
                       cwd=REPO, capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    return [ln.strip() for ln in r.stdout.splitlines() if "::" in ln]


def test_host_test_ids_are_the_same_in_two_collections():
    """``pytest -n auto`` stops with "Different tests were collected between gw0 and gw1" when a
    parameter id depends on the clock (the no-owner-label sweep test once built ids from now())."""
    import time
    first = _host_ids()
    time.sleep(0.01)
    second = _host_ids()
    assert first, "nothing collected"
    assert sorted(set(first) ^ set(second)) == []


# --- H1-I item 8: the container files use tests.runtimes, and hold only runtime tests -----------

CONTAINER_FILES = sorted((REPO / "tests").glob("test_*_container.py"))


def _defs(path: Path):
    import ast
    return ast.parse(path.read_text(encoding="utf-8")).body


@pytest.mark.parametrize("path", CONTAINER_FILES, ids=[p.name for p in CONTAINER_FILES])
def test_a_container_file_takes_its_runtimes_from_tests_runtimes(path):
    """One probe and one selection for the whole suite (S8i A2): no local ``_runtime_up`` and no
    local ``RUNTIMES = [...]`` list."""
    import ast
    body = _defs(path)
    local = [n.name for n in body if isinstance(n, ast.FunctionDef) and n.name == "_runtime_up"]
    local += [t.id for n in body if isinstance(n, ast.Assign) for t in n.targets
              if isinstance(t, ast.Name) and t.id == "RUNTIMES"]
    imported = any(isinstance(n, ast.ImportFrom) and n.module == "tests.runtimes"
                   and any(a.name == "RUNTIMES" for a in n.names) for n in body)
    assert local == [] and imported, (local, imported)


@pytest.mark.parametrize("path", CONTAINER_FILES, ids=[p.name for p in CONTAINER_FILES])
def test_every_test_in_a_container_file_takes_a_runtime(path):
    """A test that takes only pytest's own fixtures can't depend on a runtime; in a container file
    it lands in the podman job only, so podman = docker + N (T's cross-track request 1)."""
    import ast
    builtin = {"monkeypatch", "tmp_path", "tmp_path_factory", "capsys", "capfd", "caplog", "recwarn"}
    bare = [n.name for n in _defs(path) if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
            and not ({a.arg for a in n.args.args + n.args.kwonlyargs} - builtin)]
    assert bare == []


# --- H1-I item 9: the suite never touches the user's real state folder --------------------------


def test_the_suite_state_dir_is_a_temp_folder():
    """L's request: a host run used to leave an empty ``%LOCALAPPDATA%\\localbot\\_exam``. The
    conftest points ``LOCALBOT_STATE_DIR`` at a temp folder before any profile is built."""
    import tempfile
    from localbot.config import UndoConfig, get_profile, user_state_dir
    sd = os.environ.get("LOCALBOT_STATE_DIR", "")
    assert sd and Path(sd).resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()), sd
    assert user_state_dir() == Path(sd) and UndoConfig().store_dir == Path(sd)
    assert get_profile("core4").undo.store_dir == Path(sd)
    assert P_AT_IMPORT == Path(sd)  # a profile built at a test module's import time too


P_AT_IMPORT = __import__("localbot.config", fromlist=["get_profile"]).get_profile("core4").undo.store_dir
