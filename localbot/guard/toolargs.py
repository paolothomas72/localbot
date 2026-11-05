"""S8f.1a — the argument gate (L1).

A native Ollama tool call carries whatever keys the model emitted; Ollama does not enforce the
tool schema, so a wrong-case or missing key used to reach ``lab_run_tool``, where
``args.get(key) or ""`` silently turned it into "" and the tool reported success with the disk
unchanged. This gate validates the call's arg KEYS against the profile's own advertised schema
(``profile.tools.schemas``) at the dispatch choke point, before the path check, the guards, and
the executor. It never auto-repairs. Extra keys are tolerated, recorded, and dropped so the tool
and the guards see only the schema keys. A tool name with no schema passes straight through
unchanged, so the loop's existing "unknown tool" handling still fires.

Same rules for every profile (owner decision, 2026-10-30, option A: fix parity_lab6 too). The name is used
exactly as it stands at the choke point: canonical when dialect aliases are on (core4), raw when
off (parity_lab6) — both are the profile's own advertised spelling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


def schema_map(schemas: Iterable[dict]) -> dict[str, tuple[list[str], list[str]]]:
    """``profile.tools.schemas`` -> {advertised name: (required keys, all property keys)}."""
    out: dict[str, tuple[list[str], list[str]]] = {}
    for t in schemas:
        fn = (t or {}).get("function", {}) or {}
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters", {}) or {}
        out[name] = (list(params.get("required", [])), list(params.get("properties", {}) or {}))
    return out


def _norm(k: str) -> str:
    return k.lower().replace("_", "")


@dataclass
class ArgCheck:
    error: str | None = None  # the exact "ERROR: …" text, or None to proceed
    missing: list[str] = field(default_factory=list)
    got: list[str] = field(default_factory=list)  # keys the model sent (for the try record)
    extra: list[str] = field(default_factory=list)  # tolerated extra keys (dropped before run)
    clean: dict | None = None  # args with extra keys dropped; set only when proceeding


def check_tool_args(name: str, args: dict, schemas) -> ArgCheck:
    """Validate ``args`` for tool ``name`` against ``schemas`` (``profile.tools.schemas`` or a
    pre-built :func:`schema_map`). Returns an :class:`ArgCheck`; ``error`` is set iff the call
    must be refused (tool not run)."""
    smap = schemas if isinstance(schemas, dict) else schema_map(schemas)
    got = list(args.keys())
    if name not in smap:  # unknown tool: leave it for the existing "unknown tool" paths
        return ArgCheck(got=got, clean=dict(args))
    required, props = smap[name]
    present = set(got)
    extra = [k for k in got if k not in props]
    norm_present = {_norm(k): k for k in present}

    missing = [k for k in required if k not in present]  # rules 1 & 4
    if missing:
        return ArgCheck(error=_missing_text(name, required, got, missing, norm_present),
                        missing=missing, got=got, extra=extra)
    bad_type = [k for k in required if not isinstance(args.get(k), str)]  # rule 2
    if bad_type:  # the keys are present; their VALUES are not strings
        return ArgCheck(error=f"ERROR: {name} values must be strings: {', '.join(bad_type)}",
                        got=got, extra=extra)
    if name == "edit" and args.get("old_string") == "":  # rule 3
        return ArgCheck(error="ERROR: edit old_string must not be empty.", got=got, extra=extra)
    clean = {k: v for k, v in args.items() if k in props} if extra else dict(args)  # rule 5
    return ArgCheck(got=got, extra=extra, clean=clean)


def _missing_text(name, required, got, missing, norm_present) -> str:
    got_str = ", ".join(got) if got else "no keys"
    text = f"ERROR: {name} needs keys {', '.join(required)} (got: {got_str})."
    near = [k for k in missing if _norm(k) in norm_present]  # rule 4: wrong-case / underscore
    if near:
        text += f" Keys are case-sensitive: use {', '.join(near)}."
    return text
