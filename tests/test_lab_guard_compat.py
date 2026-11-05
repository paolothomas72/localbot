"""The lab's own guard test runs UNCHANGED against the port (docs/STAGE1_SPEC.md §3, §5.5, S3).

lab_ref/scripts/_test_harness_guards.py is not edited and not copied. A module shim puts
``localbot.guard.product`` in ``sys.modules["harness_guards"]`` before the script runs, so its
``from harness_guards import ...`` gets the port, not lab_ref/lib/harness_guards.py. The script
passes plain dicts; the port wraps them in a DictView (the dict adapter).

It runs in a subprocess (the script edits sys.path and imports lab task modules) with
bytecode writing off, so nothing is written into lab_ref/.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "lab_ref" / "scripts" / "_test_harness_guards.py"

SHIM_RUNNER = r"""
import runpy, sys
import localbot.guard.product as port
sys.modules["harness_guards"] = port          # the module shim
code = 0
try:
    runpy.run_path(sys.argv[1], run_name="__main__")
except SystemExit as exc:
    code = exc.code
lab = sorted(m.__file__ for m in list(sys.modules.values())
             if (getattr(m, "__file__", None) or "").replace("\\", "/").endswith("lib/harness_guards.py"))
print("shim: harness_guards ->", sys.modules["harness_guards"].__name__)
print("shim: lab harness_guards.py loaded:", lab)
print("shim: exit", code)
"""


def _manifest_hash(rel: str) -> str:
    for line in (REPO / "lab_ref" / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        if name.strip() == rel:
            return digest
    raise AssertionError(f"{rel} not in MANIFEST.sha256")


def test_lab_test_script_is_byte_identical():
    assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == _manifest_hash("scripts/_test_harness_guards.py")


def _run(code: str, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    return subprocess.run([sys.executable, "-c", code, *args], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=60)


def test_lab_guard_test_passes_unchanged_against_port():
    r = _run(SHIM_RUNNER, str(SCRIPT))
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "harness_guards ok" in r.stdout, out
    assert "shim: harness_guards -> localbot.guard.product" in r.stdout, out
    assert "shim: lab harness_guards.py loaded: []" in r.stdout, out  # the original never ran
    assert "shim: exit 0" in r.stdout, out


def test_shim_really_routes_to_the_port():
    """Control: break one veto message in the port (in the subprocess only) and the unchanged
    lab test must notice. Proves the script exercised localbot code, not a silent no-op."""
    broken = SHIM_RUNNER.replace(
        'sys.modules["harness_guards"] = port',
        'port.veto_harm = lambda name, args: None\nsys.modules["harness_guards"] = port',
    )
    r = _run(broken, str(SCRIPT))
    assert "harness_guards ok" not in r.stdout
    assert "AssertionError" in r.stderr
