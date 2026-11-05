"""One-folder workspace on disk. ProductGuard still decides; this only maps names to files."""

from __future__ import annotations

import os
from pathlib import Path

_SKIP_DIR = {".git", "node_modules", "__pycache__", ".venv", ".localharness", ".localbot"}
_MAX_FILES = 120
_MAX_BYTES = 80_000


def workspace_root(explicit: str | None = None) -> Path:
    raw = explicit or os.environ.get("LOCAL_LAB_WORKSPACE") or os.getcwd()
    return Path(raw).expanduser().resolve()


def resolve(root: Path, rel: str) -> Path:
    """Relative name only. No host, no parent of root."""
    root = root.resolve()
    blob = (rel or "").replace("\\", "/").strip()
    if not blob or blob.startswith("/") or (len(blob) >= 2 and blob[1] == ":"):
        raise ValueError("host path")
    if blob.startswith("~") or ".." in blob.split("/"):
        raise ValueError("outside workspace")
    cand = (root / blob).resolve()
    try:
        cand.relative_to(root)
    except ValueError as exc:
        raise ValueError("outside workspace") from exc
    return cand


def load_workspace(root: Path, max_files: int = _MAX_FILES, max_bytes: int = _MAX_BYTES) -> dict[str, str]:
    root = root.resolve()
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*")):
        if len(out) >= max_files:
            break
        if not path.is_file():
            continue
        if any(part in _SKIP_DIR for part in path.parts):
            continue
        rel = path.relative_to(root).as_posix()
        try:
            if path.stat().st_size > max_bytes:
                continue
            out[rel] = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return out


def write_file(root: Path, rel: str, content: str) -> None:
    dest = resolve(root, rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8", newline="\n")
