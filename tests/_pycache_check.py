"""Before / after check that a step wrote no Python bytecode cache under a folder (audit E-8).

Used by the lab_ref/ "read only" tests. An old ``__pycache__`` left in a checkout by an earlier run
is not the step's doing, so only what is new or changed after the step counts: a new
``__pycache__`` folder, a new ``.pyc``, or an old ``.pyc`` rewritten (other size or mtime).
"""
from __future__ import annotations

from pathlib import Path

Snapshot = dict[str, "tuple[int, int] | None"]


def cache_snapshot(root: Path) -> Snapshot:
    """Every ``__pycache__`` folder (as ``rel/``) and ``.pyc`` file (size, mtime) under root."""
    out: Snapshot = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if p.name == "__pycache__" and p.is_dir():
            out[rel + "/"] = None
        elif p.suffix == ".pyc" and p.is_file():
            st = p.stat()
            out[rel] = (st.st_size, st.st_mtime_ns)
    return out


def new_cache_entries(before: Snapshot, after: Snapshot) -> list[str]:
    """What the step added or rewrote: in ``after`` and either not in ``before`` or changed."""
    return sorted(k for k, v in after.items() if k not in before or before[k] != v)
