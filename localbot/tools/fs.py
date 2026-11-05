"""read / write / edit (+ glob / grep for parity) on disk (docs/STAGE1_SPEC.md §4, step S4).

``DiskWorkspace`` is one trial's ``work/`` folder: all access goes through the hardened
``WorkspaceFS``, and it keeps the lab's file *order* — the lab's workspace was a dict, so
``grep`` hits came in seed order, then new files in the order they were written. That order
decides which hit is "first", and the caller guard keys on the first hit, so parity keeps it
(a sorted disk walk would put ``app.py`` before ``lib.py`` and change the task).

``lab_run_tool`` is lab_ref/scripts/run_all_fit.py ``run_tool`` on this workspace, output
for output: numbered ``read``, ``Wrote``/``Updated``, ``old_string not found``, ``glob`` that
lists everything, ``grep`` with the lab's path filter, ``unknown tool``.
"""
from __future__ import annotations

from localbot.guard.product import DiskView
from localbot.guard.workspace_fs import WorkspaceError, WorkspaceFS


class DiskWorkspace:
    def __init__(self, fs: WorkspaceFS, order: list[str] | None = None) -> None:
        self.fs = fs
        self.order: list[str] = list(order or [])
        self.view = DiskView(fs, listing=self.files)

    def rel(self, path: str) -> str | None:
        """The posix relative name a model path resolves to, or None if refused."""
        try:
            return self.fs.resolve(path).relative_to(self.fs.root).as_posix()
        except (WorkspaceError, ValueError):
            return None

    def files(self) -> list[str]:
        """Every file on disk: known order first (lab dict order), then the rest sorted. A name
        in the order matches its exact name on disk; only in a case-insensitive folder (the lab's
        NTFS, parity) may it match another spelling, which there is the same file. In a
        case-sensitive folder ``a.txt`` and ``A.txt`` are two files and both are listed (H1-K D1)."""
        left = dict.fromkeys(self.fs.walk())  # insertion-ordered set of exact names
        folded = {} if self.fs.case_sensitive else {p.casefold(): p for p in left}
        out = []
        for p in self.order:
            hit = p if p in left else folded.get(p.casefold())
            if hit is not None and hit in left:
                out.append(hit)
                del left[hit]
        return out + list(left)

    def get(self, path: str) -> str | None:
        """Like the lab's ``files.get(path)``: text, or None if missing or refused."""
        try:
            dest = self.fs.resolve(path)
            if not dest.is_file():
                return None
            return self.fs.read(path, cap_bytes=None)
        except (WorkspaceError, OSError):
            return None

    def put(self, path: str, content: str) -> str:
        """Write and keep the lab order (a new file goes last). Returns the relative name."""
        dest = self.fs.write(path, content)
        rel = dest.relative_to(self.fs.root).as_posix()
        known = set(self.order) if self.fs.case_sensitive else {p.casefold() for p in self.order}
        if (rel if self.fs.case_sensitive else rel.casefold()) not in known:
            self.order.append(rel)
        self.view.note_write(rel)
        return rel

    def snapshot(self) -> dict[str, str]:
        """{relpath: text} of the whole tree, in lab order (files_end for judges). A file the path
        layer refuses to read (a hard link, S7a.1) is left out, not shown as empty; the runner's
        negative check flags it and its secret scan reads it raw."""
        out = {}
        for p in self.files():
            body = self.get(p)
            if body is not None:
                out[p] = body
        return out


def numbered(body: str) -> str:
    return "\n".join(f"{i}\t{line}" for i, line in enumerate(body.splitlines() or [""], 1))


def honesty_violation(name: str, before, after, written) -> bool:
    """S8f.1a (L3): did a write/edit claim success without the disk changing as claimed?
    ``edit`` -> the bytes must differ; ``write`` -> the final bytes must equal the content
    written (the already-prepared content). A backstop for a false success from any cause."""
    if name == "edit":
        return before == after
    if name == "write":
        return after != written
    return False


def lab_run_tool(name: str, args: dict, ws: DiskWorkspace) -> str:
    """lab_ref/scripts/run_all_fit.py ``run_tool``, on disk, with S8f.1a (L2) executor hardening:
    the lab turned a missing/misnamed key into "" and reported success; here read/write/edit refuse
    an empty path, edit refuses an empty or unchanged old_string and a missing file, and write
    refuses a missing content key (an empty-string value stays legal for a new file). Defence in
    depth behind the L1 arg gate; these also protect a direct call."""
    path = args.get("path") or ""
    if name in {"read", "write", "edit"} and not path:
        return f"ERROR: {name} needs a 'path'"
    if name == "read":
        body = ws.get(path)
        if body is None:
            return f"ERROR: not found: {path}"
        return numbered(body)
    if name == "write":
        if "content" not in args:
            return "ERROR: write needs a 'content'"
        ws.put(path, args["content"])
        return f"Wrote {path}"
    if name == "edit":
        old, new = args.get("old_string") or "", args.get("new_string") or ""
        if old == "":
            return "ERROR: edit old_string must not be empty."
        cur = ws.get(path)
        if cur is None:
            return f"ERROR: not found: {path}"
        if old == new:
            return "ERROR: edit old_string and new_string are identical."
        if old not in cur:
            return "ERROR: old_string not found"
        ws.put(path, cur.replace(old, new, 1))
        return f"Updated {path}"
    if name == "glob":
        return "\n".join(sorted(ws.files())) or "(empty)"
    if name == "grep":
        pat = args.get("pattern") or ""
        if not pat:
            return "ERROR: grep needs a 'pattern'"
        hits = []
        for p in ws.files():
            if path and path not in (".", p) and not p.startswith(path.rstrip("/")):
                if p != path:
                    continue
            body = ws.get(p) or ""
            for i, line in enumerate(body.splitlines(), 1):
                if pat in line:
                    hits.append(f"{p}:{i}:{line}")
        return "\n".join(hits) or "(no matches)"
    if name == "bash_exec":
        return "ERROR: bash not found"
    return f"ERROR: unknown tool {name}"


def core_read(path: str, ws: DiskWorkspace) -> str:
    return lab_run_tool("read", {"path": path}, ws)


def core_write(path: str, content: str, ws: DiskWorkspace) -> str:
    return lab_run_tool("write", {"path": path, "content": content}, ws)


def core_edit(args: dict, ws: DiskWorkspace) -> str:
    return lab_run_tool("edit", args, ws)
