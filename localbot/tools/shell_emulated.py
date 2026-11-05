"""Emulated shell, parity profile only (port of lab_ref/lib/sandbox_bash.py, step S4).

Same regexes, same order, same texts; the dict became a ``DiskWorkspace``. ``ls`` lists every
file (sorted), ``cat <path>`` returns the raw body; anything else is "not handled" and the
caller falls back to the lab's "bash not found" stub.
"""
from __future__ import annotations

import re

from localbot.tools.fs import DiskWorkspace

_LS = re.compile(r"^ls(?:\s+(?:-l|-la|-1|\*))?$")
_CAT = re.compile(r"^cat\s+([A-Za-z0-9_./\\-]+)$")
_NET = re.compile(r"\b(curl|wget|invoke-webrequest|iwr)\b|https?://", re.I)


def sandbox_bash(cmd: str, ws: DiskWorkspace) -> tuple[bool, str]:
    """Return (handled, text). handled=False means unknown — caller may stub."""
    c = (cmd or "").strip()
    if not c:
        return True, "ERROR: empty command"
    if _NET.search(c):
        return True, "ERROR: harness: refused network. Workspace is local-only."
    if _LS.match(c):
        return True, "\n".join(sorted(ws.files())) or "(empty)"
    if ".." in c or c.startswith("/") or re.search(r"[A-Za-z]:[\\/]", c) or "~/" in c or re.search(r"(?:^|[\s])/", c.replace("\\", "/")):
        return True, "ERROR: harness: refused host path. Workspace only."
    m = _CAT.match(c)
    if m:
        # H1-J J3: lstrip("./") strips CHARACTERS (`.env` -> `env`). Kept on purpose: it is the
        # lab's line byte for byte, and parity must answer as the lab did. Parity profile only;
        # the guard checks the raw command first. Proven by test_sandbox_host.py::
        # test_emulated_bash_answers_exactly_as_the_lab (red with removeprefix).
        path = m.group(1).replace("\\", "/").lstrip("./")
        body = ws.get(path) or ws.get(m.group(1))  # lab: `or`, so an empty file falls through
        if body is None:
            return True, f"ERROR: not found: {path}"
        return True, body
    return False, ""
