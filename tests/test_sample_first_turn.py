"""S8g 1a: scripts/sample_first_turn.py (the reviewer's first-turn sampler, moved from state/tools).

The script sends the ``parity_lab6`` first request of trust ``s_clean_list`` N times and counts the
model's first tool. It refuses to send unless the body it builds matches the S8f.2 capture (sha256
4447e52aa7d4e1e1, byte-identical on all three sides). Here it only runs with ``--dry-run``, or with
the task prompt changed so that it must refuse. Both run in a subprocess where every network path
raises, so an attempt to send fails the test instead of reaching Ollama. The prompt is changed in
memory only; nothing is written into lab_ref/.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sample_first_turn.py"

DRIVER = r"""
import runpy, socket, sys, urllib.request
script, repo, mode, *argv = sys.argv[1:]

def no_net(*_a, **_k):
    raise AssertionError("network call attempted")

urllib.request.urlopen = no_net
socket.create_connection = no_net
socket.socket.connect = lambda self, address: no_net()
if mode == "changed":
    sys.dont_write_bytecode = True  # the driver's own lab import must not write a cache either
    sys.path[:0] = [repo, repo + "/lab_ref/lib"]
    import safety_tasks
    for t in safety_tasks.TASKS:
        if t["id"] == "s_clean_list":
            t["prompt"] += " "  # in memory only: one byte more than the captured request
# The script runs with bytecode writing on, as a plain `python -I` run has it: keeping lab_ref/
# clean is the script's own job (E-8: the driver must not do it for the script).
sys.dont_write_bytecode = False
print("driver: bytecode writing on before the script:", not sys.dont_write_bytecode, file=sys.stderr)
sys.argv = [script, *argv]
runpy.run_path(script, run_name="__main__")
"""


def _run(mode: str, *argv: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)  # -I ignores PYTHON* env vars anyway; the DRIVER decides bytecode
    env.pop("PYTHONPATH", None)
    return subprocess.run([sys.executable, "-I", "-c", DRIVER, str(SCRIPT), str(REPO), mode, *argv],
                          capture_output=True, text=True, encoding="utf-8", env=env, cwd=REPO, timeout=120)


def test_dry_run_matches_the_captured_request_and_sends_nothing():
    r = _run("dry", "--dry-run")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "body 1923 B sha256 4447e52aa7d4e1e1" in r.stdout
    assert "-> MATCH" in r.stdout
    assert "network call attempted" not in r.stderr


def test_changed_prompt_refuses_to_send():
    r = _run("changed", "--n", "1")  # no --dry-run: only the sha check stands between it and Ollama
    assert r.returncode == 2, (r.returncode, r.stdout, r.stderr[-2000:])
    assert "MISMATCH" in r.stdout and "refusing to send: not the captured request" in r.stdout
    assert "network call attempted" not in r.stderr


def test_nothing_written_into_lab_ref():
    from tests._pycache_check import cache_snapshot, new_cache_entries

    before = cache_snapshot(REPO / "lab_ref")  # E-8: only a NEW cache fails, not one left in a checkout
    for mode, argv, rc in (("dry", ["--dry-run"], 0), ("changed", ["--n", "1"], 2)):
        r = _run(mode, *argv)  # "dry": the script's own `from safety_tasks import` loads the lab module
        assert r.returncode == rc, (mode, r.returncode, r.stderr[-2000:])
        assert "driver: bytecode writing on before the script: True" in r.stderr, r.stderr[-2000:]
    assert new_cache_entries(before, cache_snapshot(REPO / "lab_ref")) == []
