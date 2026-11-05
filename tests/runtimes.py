"""The one place the suite decides which container runtimes to test (S8i A2, audit E-4).

- ``LOCALBOT_TEST_RUNTIMES``: the runtimes this session tests: ``podman``, ``docker``, or both
  (``podman,docker``, also ``podman|docker``). Unset or blank: both. An unknown name is an error.
- ``LOCALBOT_REQUIRE_RUNTIMES``: the runtimes that must be up, same spelling. Unset: none. With it
  set, tests/conftest.py stops the session when a tested runtime is down and turns the skip of a
  container test into a failure.

``RUNTIMES`` is the selected runtimes that are up, in ``CANDIDATES`` order. It is worked out on
first use only (``from tests.runtimes import RUNTIMES`` counts as a use), and once per process, so
a session that never asks for it runs no ``podman info`` / ``docker info`` because of this module.

The small helpers at the end are what the conftest hooks use; they are pure so they can be tested
without a pytest session.
"""
from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
from collections.abc import Iterable, Mapping

CANDIDATES = ("podman", "docker")
TEST_ENV = "LOCALBOT_TEST_RUNTIMES"
REQUIRE_ENV = "LOCALBOT_REQUIRE_RUNTIMES"
PROBE_TIMEOUT = 60  # seconds, as in the container test files


def runtime_up(exe: str) -> bool:
    """``<exe> info`` exits 0: the probe the container test files use."""
    if not shutil.which(exe):
        return False
    try:
        return subprocess.run([exe, "info"], capture_output=True, timeout=PROBE_TIMEOUT).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def parse_runtimes(raw: str | None, env: str) -> tuple[str, ...] | None:
    """The runtime names in an env value, in CANDIDATES order; None when unset or blank."""
    if raw is None or not raw.strip():
        return None
    names = [p.strip().lower() for p in re.split(r"[,|]", raw) if p.strip()]
    bad = [n for n in names if n not in CANDIDATES]
    if bad or not names:
        raise ValueError(f"{env}={raw!r}: unknown runtime {', '.join(map(repr, bad)) or '(none named)'}; "
                         f"use podman, docker, or podman,docker")
    return tuple(c for c in CANDIDATES if c in names)


def selected_runtimes() -> tuple[str, ...]:
    """The runtimes this session tests (``LOCALBOT_TEST_RUNTIMES``; unset: both)."""
    return parse_runtimes(os.environ.get(TEST_ENV), TEST_ENV) or CANDIDATES


def required_runtimes() -> tuple[str, ...]:
    """The runtimes that must be up (``LOCALBOT_REQUIRE_RUNTIMES``; unset: none)."""
    return parse_runtimes(os.environ.get(REQUIRE_ENV), REQUIRE_ENV) or ()


@functools.cache
def is_up(exe: str) -> bool:
    """``runtime_up``, asked once per runtime per process."""
    return runtime_up(exe)


@functools.cache
def _up_runtimes() -> tuple[str, ...]:
    return tuple(r for r in selected_runtimes() if is_up(r))


def __getattr__(name: str):
    if name == "RUNTIMES":
        return list(_up_runtimes())  # a copy: a caller's edit never changes the cached answer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def reset() -> None:
    """Forget the cached probe answers (for tests that fake the probe)."""
    is_up.cache_clear()
    _up_runtimes.cache_clear()


# --- helpers for the conftest hooks ------------------------------------------------------------


def item_runtime(params: Mapping[str, object], module_runtimes: Iterable[str] | None) -> str | None:
    """The runtime a container test runs on: a parameter whose value is a runtime name (from
    ``@pytest.fixture(params=RUNTIMES)`` or ``@pytest.mark.parametrize("runtime", RUNTIMES)``);
    else the first of its file's own ``RUNTIMES`` (a test that is not parametrized uses
    ``RUNTIMES[0]``); else None: unknown, e.g. the one item pytest makes for an empty parameter set."""
    for value in params.values():
        if isinstance(value, str) and value in CANDIDATES:
            return value
    first = next(iter(module_runtimes or ()), None)
    return first if first in CANDIDATES else None


def job_runtime(runtime: str | None) -> str:
    """The runtime job a container test belongs to. An unknown one belongs to the first candidate
    (podman), so it is counted in exactly one of the podman / docker jobs and the job counts still
    add up to the unfiltered total."""
    return runtime or CANDIDATES[0]


def missing_by_module(present: Mapping[str, set[str]], need: Iterable[str]) -> dict[str, list[str]]:
    """For each test file, the needed runtimes it has no test for (files missing none are left out)."""
    need = tuple(need)
    out: dict[str, list[str]] = {}
    for module, have in present.items():
        miss = [r for r in need if r not in have]
        if miss:
            out[module] = miss
    return out
