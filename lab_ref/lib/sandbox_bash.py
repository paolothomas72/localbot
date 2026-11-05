"""Workspace-only bash. No host shell. Harm still vetoed in ProductGuard first."""

from __future__ import annotations

import re

_LS = re.compile(r"^ls(?:\s+(?:-l|-la|-1|\*))?$")
_CAT = re.compile(r"^cat\s+([A-Za-z0-9_./\\-]+)$")
_NET = re.compile(r"\b(curl|wget|invoke-webrequest|iwr)\b|https?://", re.I)


def sandbox_bash(cmd: str, files: dict[str, str]) -> tuple[bool, str]:
    """Return (handled, text). handled=False means unknown — caller may stub."""
    c = (cmd or "").strip()
    if not c:
        return True, "ERROR: empty command"
    if _NET.search(c):
        return True, "ERROR: harness: refused network. Workspace is local-only."
    if _LS.match(c):
        return True, "\n".join(sorted(files)) or "(empty)"
    if ".." in c or c.startswith("/") or re.search(r"[A-Za-z]:[\\/]", c) or "~/" in c or re.search(r"(?:^|[\s])/", c.replace("\\", "/")):
        return True, "ERROR: harness: refused host path. Workspace only."
    m = _CAT.match(c)
    if m:
        path = m.group(1).replace("\\", "/").lstrip("./")
        body = files.get(path) or files.get(m.group(1))
        if body is None:
            return True, f"ERROR: not found: {path}"
        return True, body
    return False, ""
