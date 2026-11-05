"""Fail snapshots for corrected gold. Never train the raw fail. Secrets stay out."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from paths import RAW
from safety_tasks import CANARY

_SECRET = re.compile(r"CANARY_pw_|sk-[A-Za-z0-9-]+|TOKEN\s*=|password\s+is\s+\S+", re.I)
CAPTURES = RAW / "captures"


def route_label(kind: str | None, prompt: str = "") -> str:
    k = (kind or "").lower()
    p = (prompt or "").lower()
    if k in {"search"} or "grep" in p or "find which file" in p or "contains" in p:
        return "grep-ish"
    if k in {"secret", "harm"} or "rm -rf" in p or ".env" in p:
        return "refuse"
    if "batch" in k or "three files" in p or "four files" in p:
        return "batch"
    if k in {"intent"} or "improve" in p or "make this" in p:
        return "one-file"
    return "one-file"


def redact(text: str) -> str:
    t = text or ""
    if CANARY in t:
        t = t.replace(CANARY, "[REDACTED]")
    return _SECRET.sub("[REDACTED]", t)


def redact_files(files: dict[str, str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for path, body in (files or {}).items():
        p = path.replace("\\", "/").lower()
        if p.endswith("secret.txt") or p.endswith(".env") or p.endswith("secrets"):
            out[path] = "[REDACTED]"
            continue
        out[path] = redact(str(body))
    return out


def write_fail_capture(
    *,
    model: str,
    task_id: str,
    prompt: str,
    kind: str | None,
    seed: dict[str, str],
    files: dict[str, str],
    tools: list[str],
    reason: str,
    text: str,
    vetoes: int,
) -> Path:
    CAPTURES.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = re.sub(r"[^\w.-]+", "_", model)[:48]
    dest = CAPTURES / f"{stamp}_{safe_model}_{task_id}.json"
    row = {
        "stamp": stamp,
        "model": model,
        "id": task_id,
        "kind": kind,
        "route": route_label(kind, prompt),
        "prompt": redact(prompt),
        "seed": redact_files(seed),
        "files_end": redact_files(files),
        "tools": tools,
        "reason": redact(reason),
        "text": redact(text)[:400],
        "vetoes": vetoes,
        "train": False,
        "note": "Raw fail. Write a corrected gold loop before any SFT.",
    }
    dest.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    return dest
