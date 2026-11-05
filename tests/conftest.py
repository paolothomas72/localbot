"""Shared fixtures, and the suite split (S8i A2). ``trial``-style fixtures for S2/S3 tests live in
their own files.

The suite split: every test in a ``tests/test_*_container.py`` file is marked ``container``, so G1
runs as three jobs, one after another: ``-m "not container"`` (host, may use ``-n auto``), then
``-m container`` with ``LOCALBOT_TEST_RUNTIMES=podman``, then with ``=docker``. A container test
belongs to the job of its runtime (tests/runtimes.py), so the three job counts add up to the
unfiltered total. With ``LOCALBOT_REQUIRE_RUNTIMES`` set (audit E-4), a tested runtime that is down
stops the session, and a container test that would skip fails instead.

The run lock (audit E-3): a session that will run container tests holds scripts/runlock.py's
machine-wide lock until it ends, and so does every session in the main checkout (exams run there,
and a host test adds a workspaces/_trials folder that a running exam counts as a host change, audit
S-3), unless the job that started it (watch_run, the GPU batch) holds it already; busy -> nothing
runs, exit 3. A host session in a worktree (its own _trials) and ``--collect-only`` leave it alone.
Container tests never run under xdist: each worker would sweep the others' containers (audit S-2).
"""
from __future__ import annotations

import atexit
import fnmatch
import os
import shutil
import tempfile
from pathlib import Path

# H1-I item 9 (L's request): the suite never touches the user's real state folder. Set before any
# localbot import, since profiles (UndoConfig.store_dir, PathsConfig.workspaces) read it when
# built, and test modules build some at import. One temp folder per process (xdist: per worker;
# an inner pytest makes its own); a second import of this file in the same process (pytest loads
# it as ``conftest``, a test imports ``tests.conftest``) keeps the first one.
_OWNER_ENV = "LOCALBOT_TEST_STATE_OWNER"
if os.environ.get(_OWNER_ENV) != str(os.getpid()):
    _state_dir = tempfile.mkdtemp(prefix="localbot-test-state-")
    os.environ["LOCALBOT_STATE_DIR"] = _state_dir
    os.environ[_OWNER_ENV] = str(os.getpid())
    atexit.register(shutil.rmtree, _state_dir, True)

import pytest  # noqa: E402

from scripts import runlock  # noqa: E402
from tests import runtimes  # noqa: E402
from tests.fakes import Trial  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONTAINER_FILES = "test_*_container.py"
# A container test's runtime (None: unknown). Only container tests carry it.
_RUNTIME = pytest.StashKey["str | None"]()
# (selected runtimes, required runtimes) for this session, read once from the env.
_SPLIT = pytest.StashKey["tuple[tuple[str, ...], tuple[str, ...]]"]()
# The run lock this session took (None: it took none).
_LOCK = pytest.StashKey["runlock.RunLock | None"]()


@pytest.fixture
def disk():
    """A fresh workspaces/_trials/s4_*/work folder, deleted after the test."""
    t = Trial()
    try:
        yield t
    finally:
        t.cleanup()
        assert not t.base.exists(), f"trial folder not deleted: {t.base}"


# --- the suite split (S8i A2) --------------------------------------------------------------------


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "container: runs against a real podman / docker runtime (a tests/test_*_container.py file)")
    try:
        config.stash[_SPLIT] = (runtimes.selected_runtimes(), runtimes.required_runtimes())
    except ValueError as e:
        raise pytest.UsageError(str(e)) from None
    # The xdist controller takes it here too, before any worker starts; workers never take it.
    if not hasattr(config, "workerinput") and not config.option.collectonly and _in_main_checkout():
        _take_lock(config)


def pytest_report_header(config) -> list[str]:
    """Say which runtimes this session tests when the env narrows or requires them, so a variable
    left set in the shell after a G1 job never deselects a runtime silently."""
    selected, required = config.stash[_SPLIT]
    if selected == runtimes.CANDIDATES and not required:
        return []
    lines = [f"container runtimes tested: {', '.join(selected)}"]
    if selected != runtimes.CANDIDATES:
        left_out = [r for r in runtimes.CANDIDATES if r not in selected]
        lines[0] += (f" ({runtimes.TEST_ENV}={','.join(selected)}; "
                     f"{' / '.join(left_out)} container tests are deselected)")
    if required:
        lines.append(f"required (a down runtime or a skip fails): {', '.join(required)} ({runtimes.REQUIRE_ENV})")
    return lines


@pytest.hookimpl(wrapper=True)
def pytest_collection_modifyitems(config, items):
    """Around every other plugin's hook: mark container tests before ``-m`` / ``-k`` deselect, then
    check ``LOCALBOT_REQUIRE_RUNTIMES`` and drop the other runtime's tests after them."""
    present = _mark_container_items(items)
    result = yield
    _split_by_runtime(config, items, present)
    return result


def _mark_container_items(items) -> dict[str, set[str]]:
    """Mark and note the runtime of every container test; returns the runtimes each container file
    has tests for, over everything collected (so a ``-k`` filter does not look like a gap)."""
    present: dict[str, set[str]] = {}
    for item in items:
        if not fnmatch.fnmatch(item.path.name, CONTAINER_FILES):
            continue
        item.add_marker(pytest.mark.container)
        params = getattr(getattr(item, "callspec", None), "params", {})
        rt = runtimes.item_runtime(params, getattr(getattr(item, "module", None), "RUNTIMES", None))
        item.stash[_RUNTIME] = rt
        present.setdefault(_file_of(item), set()).update([rt] if rt else [])
    return present


def _split_by_runtime(config, items, present) -> None:
    selected, required = config.stash[_SPLIT]
    container = [i for i in items if _RUNTIME in i.stash]
    if not container:
        return  # the host job: nothing to check, nothing probed
    need = [r for r in selected if r in required]
    if need:
        # Before the runtime filter below: the P job with podman down still has docker tests.
        req = f"{runtimes.REQUIRE_ENV}={','.join(required)}"
        for r in need:
            if not runtimes.is_up(r):
                pytest.exit(f"{req}: {r} is not up (`{r} info` failed or timed out). "
                            f"Start {r}, or unset {runtimes.REQUIRE_ENV}.", returncode=1)
        files = {_file_of(i) for i in container}
        gaps = runtimes.missing_by_module({f: present[f] for f in sorted(files)}, need)
        if gaps:
            say = "; ".join(f"{f} has no {' / '.join(m)} tests" for f, m in gaps.items())
            pytest.exit(f"{req}: {say} (did its own runtime probe find it down?). A partial run by node id "
                        f"can trip this too: unset {runtimes.REQUIRE_ENV} for that.", returncode=1)
    keep, drop = [], []
    for item in items:
        other = _RUNTIME in item.stash and runtimes.job_runtime(item.stash[_RUNTIME]) not in selected
        (drop if other else keep).append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    """With LOCALBOT_REQUIRE_RUNTIMES set, every skip of a container test on a required (or unknown)
    runtime becomes a failure: G1 allows 0 skipped, and a list of "allowed" skip reasons would let
    a new one slip through."""
    rep = yield
    required = item.config.stash[_SPLIT][1]
    if required and rep.skipped and not hasattr(rep, "wasxfail") and _RUNTIME in item.stash:
        rt = item.stash[_RUNTIME]
        if rt is None or rt in required:
            reason = rep.longrepr
            if isinstance(reason, tuple) and len(reason) == 3:
                reason = f"{reason[0]}:{reason[1]}: {reason[2]}"
            rep.outcome = "failed"
            rep.longrepr = (f"skip turned into a failure ({runtimes.REQUIRE_ENV}={','.join(required)} is set, "
                            f"runtime {rt or 'unknown'}); the skip said: {reason}")
    return rep


def _file_of(item) -> str:
    return item.nodeid.split("::", 1)[0]


# --- the run lock (audit E-3) --------------------------------------------------------------------


XDIST_REFUSED = ("container tests don't run under xdist (-n): every worker would sweep the others' "
                 "containers (audit S-2). Run the container job without -n.")


def _in_main_checkout() -> bool:
    """True when this suite runs in the main checkout (the one exams run in), not in a worktree."""
    return runlock.main_checkout(REPO).resolve() == REPO


def _take_lock(config) -> None:
    if config.stash.get(_LOCK, None) is not None or runlock.held_by_parent():
        return  # taken already, or watch_run / the GPU batch hold it for this whole job
    try:
        config.stash[_LOCK] = runlock.acquire("pytest")
    except runlock.LockBusy as e:
        pytest.exit(f"{e}. Another run holds the run lock; nothing ran. Wait for its verdict "
                    f"(`python scripts/runlock.py status`).", returncode=runlock.BUSY_EXIT)


def pytest_collection_finish(session):
    """After the split: if any container test will run, hold the run lock for the whole session."""
    config = session.config
    if config.option.collectonly or not any(_RUNTIME in i.stash for i in session.items):
        return
    if hasattr(config, "workerinput"):
        return  # an xdist worker: pytest_runtest_setup fails its container tests instead
    _take_lock(config)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Under xdist a container test fails before any of its fixtures (sweep, container) runs. A
    worker can't hold the lock for the session, and pytest.exit in a worker is an INTERNALERROR that
    hides the reason."""
    if _RUNTIME in item.stash and hasattr(item.config, "workerinput"):
        pytest.fail(XDIST_REFUSED, pytrace=False)


def pytest_unconfigure(config):
    lock = config.stash.get(_LOCK, None)
    if lock is not None:
        lock.release()
