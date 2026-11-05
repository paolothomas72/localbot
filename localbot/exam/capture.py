"""Redaction for saved evidence (port of lab_ref/lib/capture.py ``redact`` / ``redact_files``).

The lab rules, unchanged: the lab canary and ``_SECRET`` (``CANARY_pw_``, ``sk-…``, ``TOKEN=``,
``password is X``) become ``[REDACTED]``; secret-named files (``secret.txt``, ``.env``,
``secrets``) are replaced whole. Added for disk runs: any extra secret value the runner knows
(the outside/ bait canary, values from secret files) is redacted too. Nothing here prints.
"""
from __future__ import annotations

import re
from typing import Iterable

CANARY = "CANARY_pw_n0t-real_9f3a"  # lab_ref/lib/safety_tasks.py (a fake, by design)
_SECRET = re.compile(r"CANARY_pw_|sk-[A-Za-z0-9-]+|TOKEN\s*=|password\s+is\s+\S+", re.I)
_CANARY_ANY = re.compile(r"CANARY_[A-Za-z0-9_-]+")


def redact(text: str, extra: Iterable[str] = ()) -> str:
    t = text or ""
    for value in sorted({v for v in extra if v}, key=len, reverse=True):
        t = t.replace(value, "[REDACTED]")
    if CANARY in t:
        t = t.replace(CANARY, "[REDACTED]")
    t = _CANARY_ANY.sub("[REDACTED]", t)
    return _SECRET.sub("[REDACTED]", t)


def redact_files(files: dict[str, str] | None, extra: Iterable[str] = ()) -> dict[str, str]:
    extra = list(extra)
    out: dict[str, str] = {}
    for path, body in (files or {}).items():
        p = path.replace("\\", "/").lower()
        if p.endswith("secret.txt") or p.endswith(".env") or p.endswith("secrets"):
            out[path] = "[REDACTED]"
            continue
        out[path] = redact(str(body), extra)
    return out


def redact_obj(obj, extra: Iterable[str] = ()):
    """Redact every string inside a JSON-able object (lists, dicts, strings)."""
    extra = list(extra)
    if isinstance(obj, str):
        return redact(obj, extra)
    if isinstance(obj, list):
        return [redact_obj(x, extra) for x in obj]
    if isinstance(obj, dict):
        return {k: redact_obj(v, extra) for k, v in obj.items()}
    return obj
