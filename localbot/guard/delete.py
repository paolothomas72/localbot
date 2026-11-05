"""Delete policy, core4 only (S8c.2). parity_lab6 never builds one, so the lab loop is unchanged.

Rule (owner decision, 2026-10-30): a file that existed at task start may not be deleted unless the task
has ``allow_delete: true`` (default false). Moves inside the workspace are allowed. Files the
model created during the task may be deleted.

Two layers, both driven by ``loop.exec_core4`` for ``bash``:

- **Pre-check** (``bash_delete_targets``): reads the command before it runs and names the
  protected files it would delete: ``rm`` / ``unlink`` / ``shred`` / ``git rm`` operands (globs,
  ``{a,b}`` and ``-r`` folders expanded against the protected list), ``find … -delete`` and
  ``find … -exec/-execdir rm``, ``git clean`` (any protected file may be untracked), and
  ``mv … /dev/null``. A hit refuses the command. It reads text only, and what it cannot read
  (``python -c os.remove``, ``$f``, ``xargs``, an unknown ``find`` test) it leaves to the
  post-step check instead of guessing.
- **Post-step check** (``DeletePolicy.check``): the safety net, for any route. After every
  bash step, a protected file that is gone and whose pre-step bytes are not at a new path in
  the workspace was deleted by the step: the loop restores the step from its snapshot and
  vetoes it. The same bytes at a new path is a move: allowed, and the new path is protected
  from then on (so ``mv a.txt sub/`` then ``rm sub/a.txt`` is still a delete).
"""
from __future__ import annotations

import fnmatch
import os
import re
import shlex
from dataclasses import dataclass, field

from localbot.guard.undo import Snapshot, UndoError, file_sha

DELETE_MSG = ("harness: Deleting existing files needs the user's OK. Don't delete. Tell the user "
              "which files you would remove and ask whether to delete or move them.")

_SEPARATORS = {";", "&&", "||", "|", "|&", "&", ";;", "(", ")", "!", "{", "}"}
_PREFIXES = {"then", "do", "else", "elif", "if", "while", "until", "time", "exec", "command",
             "builtin", "nohup", "sudo"}
_SHELLS = {"bash", "sh", "dash", "zsh"}
_DELETE_VERBS = {"rm", "unlink", "shred"}
_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_REDIRECT = re.compile(r"^[<>&]+$")
_GLOB = re.compile(r"[*?\[]")
_DELETE_WORD = re.compile(r"(?:^|[\s;&|(/])(?:rm|unlink|shred)(?:\s|$)")
_XARGS_ARG = {"-n", "-I", "-d", "-L", "-P", "-s", "-a", "-E"}
_SHRED_ARG = {"-n", "-s", "--iterations", "--size", "--random-source"}
_GIT_ARG = {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}
_FIND_TESTS = {"-name", "-iname", "-path", "-ipath", "-wholename", "-iwholename"}
_FIND_FLAGS = {"-print", "-print0", "-depth", "-xdev", "-mount", "-noleaf", "-ignore_readdir_race",
               "-daystart", "-true", "-ls"}
_FIND_EXEC = {"-exec", "-execdir", "-ok", "-okdir"}


# --- the pre-check --------------------------------------------------------------------------


def _segments(cmd: str) -> list[list[str]] | None:
    """The simple commands of ``cmd`` as word lists (operators, newlines and backticks split
    them). None when the text does not tokenize (bash would not run it either)."""
    lex = shlex.shlex(cmd.replace("\n", " ; ").replace("`", " ; "), posix=True, punctuation_chars=True)
    lex.whitespace_split = True
    try:
        tokens = list(lex)
    except ValueError:
        return None
    out: list[list[str]] = [[]]
    for tok in tokens:
        if tok in _SEPARATORS:
            out.append([])
        else:
            out[-1].append(tok)
    return [seg for seg in out if seg]


def _drop_redirects(words: list[str]) -> list[str]:
    """Words without redirections (``>``, ``2>&1``, ``< x``): the target is not an operand."""
    out: list[str] = []
    skip = False
    for i, w in enumerate(words):
        if skip:
            skip = False
            continue
        if _REDIRECT.match(w):
            skip = True
            continue
        if w.isdigit() and i + 1 < len(words) and _REDIRECT.match(words[i + 1]):
            continue  # the fd number of "2>"
        out.append(w)
    return out


def _verb(words: list[str]) -> tuple[str, list[str]]:
    """Strip assignments and transparent prefixes (``env``, ``nice``, ``timeout``, ``xargs``, …);
    return the command's base name and its arguments."""
    i = 0
    while i < len(words):
        w = words[i]
        base = os.path.basename(w)
        if _ASSIGN.match(w) or base in _PREFIXES:
            i += 1
        elif base == "env":
            i += 1
            while i < len(words) and (words[i].startswith("-") or _ASSIGN.match(words[i])):
                i += 2 if words[i] == "-u" else 1
        elif base == "nice":
            i += 1
            while i < len(words) and words[i].startswith("-"):
                i += 2 if words[i] == "-n" else 1
        elif base == "timeout":
            i += 1
            while i < len(words) and words[i].startswith("-"):
                i += 2 if words[i] in ("-k", "-s", "--signal", "--kill-after") else 1
            i += 1  # the duration
        elif base == "xargs":
            i += 1
            while i < len(words) and words[i].startswith("-"):
                i += 2 if words[i] in _XARGS_ARG else 1
        else:
            return base, words[i + 1:]
    return "", []


def _braces(word: str) -> list[str]:
    """Bash brace expansion for ``{a,b}`` (nested too); a word without one is itself."""
    depth, start = 0, -1
    for i, ch in enumerate(word):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                inner = word[start + 1:i]
                parts, d, cur = [], 0, ""
                for c in inner:
                    if c == "," and d == 0:
                        parts.append(cur)
                        cur = ""
                        continue
                    d += (c == "{") - (c == "}")
                    cur += c
                parts.append(cur)
                if len(parts) < 2:
                    return [word]
                head, tail = word[:start], word[i + 1:]
                return [x for p in parts for x in _braces(head + p + tail)]
    return [word]


def _operand_parts(op: str) -> list[str] | None:
    """A workspace path operand as components (``[]`` = the workspace root). ``/work/x`` is
    ``x`` (core4 allows the container's own /work). None for what we cannot place: other
    absolute paths and ``..`` (the guard refuses them), ``~``, variables, substitutions."""
    if not op or "$" in op or op.startswith("~"):
        return None
    if op == "/work" or op.startswith("/work/"):
        op = op[len("/work"):]
    elif op.startswith("/"):
        return None
    parts = [p for p in op.split("/") if p not in ("", ".")]
    if ".." in parts:
        return None
    return parts


def _comp_match(pat: str, name: str) -> bool:
    """One path component. A literal name is looked up on NTFS (case-insensitive); a glob is
    matched by bash itself (case-sensitive, ``*`` never matches a leading dot)."""
    if not _GLOB.search(pat):
        return pat.casefold() == name.casefold()
    if name.startswith(".") and not pat.startswith("."):
        return False
    return fnmatch.fnmatchcase(name, pat.replace("**", "*"))


def _parts_match(pat: list[str], path: list[str], recursive: bool, cd: bool) -> bool:
    """``pat`` names ``path``: the same file, or (``recursive``) a folder above it. After a
    ``cd`` the operand is relative to an unknown folder, so any tail of ``path`` may match."""
    starts = range(len(path)) if cd else range(1)
    for k in starts:
        tail = path[k:]
        if len(pat) > len(tail) or (not recursive and len(pat) != len(tail)):
            continue
        if all(_comp_match(p, n) for p, n in zip(pat, tail)):
            return True
    return False


def _match_operands(ops: list[str], protected: set[str], recursive: bool, cd: bool) -> set[str]:
    hits: set[str] = set()
    for op in ops:
        for word in _braces(op):
            parts = _operand_parts(word)
            if parts is None:
                continue
            if not parts and not recursive:
                continue  # `rm .` deletes nothing
            hits |= {p for p in protected if _parts_match(parts, p.split("/"), recursive, cd)}
    return hits


def _split_opts(args: list[str], takes_arg: set[str] = frozenset()) -> tuple[list[str], list[str]]:
    """(options, operands); ``--`` ends the options, an option in ``takes_arg`` eats the next word."""
    opts: list[str] = []
    ops: list[str] = []
    i, only_ops = 0, False
    while i < len(args):
        a = args[i]
        if only_ops or not a.startswith("-") or a == "-":
            ops.append(a)
        elif a == "--":
            only_ops = True
        else:
            opts.append(a)
            if a in takes_arg:
                i += 1
        i += 1
    return opts, ops


def _recursive(opts: list[str]) -> bool:
    return any(o == "--recursive" or (not o.startswith("--") and ("r" in o or "R" in o)) for o in opts)


def _find_targets(args: list[str], protected: set[str], cd: bool) -> set[str]:
    """Protected files a ``find`` with ``-delete`` or ``-exec rm`` would reach. Evaluated:
    starting points, -name/-iname/-path/-wholename, -type, -maxdepth/-mindepth. Any other
    test or operator (-o, !, -newer, …) cannot be read here: left to the post-step check."""
    i = 0
    while i < len(args) and args[i] in ("-H", "-L", "-P"):
        i += 1
    starts: list[str] = []
    while i < len(args) and not args[i].startswith("-"):
        starts.append(args[i])
        i += 1
    tests: list[tuple[str, str]] = []
    types: set[str] | None = None
    maxdepth, mindepth = None, 0
    deleting = exec_rm = unknown = False
    while i < len(args):
        a = args[i]
        if a == "-delete":
            deleting = True
        elif a in _FIND_EXEC:
            sub = args[i + 1:]
            exec_rm = exec_rm or bool(_DELETE_WORD.search(" " + " ".join(sub)))
            i = len(args)  # the segment ends at the exec's own terminator
            continue
        elif a in _FIND_TESTS and i + 1 < len(args):
            tests.append((a, args[i + 1]))
            i += 1
        elif a == "-type" and i + 1 < len(args):
            types = set(args[i + 1].split(","))
            i += 1
        elif a in ("-maxdepth", "-mindepth") and i + 1 < len(args) and args[i + 1].isdigit():
            if a == "-maxdepth":
                maxdepth = int(args[i + 1])
            else:
                mindepth = int(args[i + 1])
            i += 1
        elif a not in _FIND_FLAGS:
            unknown = True
        i += 1
    if not (deleting or exec_rm) or unknown:
        return set()
    if types is not None and "f" not in types:
        if not (exec_rm and "d" in types):
            return set()  # -delete on folders only removes empty ones; files are not reached
        tests, maxdepth = [], None  # `-exec rm -r` on a matched folder takes every file below it
    hits: set[str] = set()
    for start in starts or ["."]:
        root = _operand_parts(start)
        if root is None:
            continue
        shown = start.rstrip("/") or start
        for p in protected:
            path = p.split("/")
            for k in range(len(path)) if cd else range(1):  # after a cd: any folder may be the start
                head = path[k:k + len(root)]
                if len(head) < len(root) or not all(_comp_match(r, n) for r, n in zip(root, head)):
                    continue
                rel = path[k + len(root):]
                if len(rel) < mindepth or (maxdepth is not None and len(rel) > maxdepth):
                    continue
                printed = "/".join([shown, *rel]) if rel else shown
                if all(_find_test(t, v, path[-1], printed) for t, v in tests):
                    hits.add(p)
                    break
    return hits


def _find_test(test: str, pat: str, name: str, printed: str) -> bool:
    if test == "-name":
        return fnmatch.fnmatchcase(name, pat)
    if test == "-iname":
        return fnmatch.fnmatchcase(name.casefold(), pat.casefold())
    if test in ("-path", "-wholename"):
        return fnmatch.fnmatchcase(printed, pat)
    return fnmatch.fnmatchcase(printed.casefold(), pat.casefold())  # -ipath / -iwholename


def _git_targets(args: list[str], protected: set[str], cd: bool) -> set[str]:
    i = 0
    while i < len(args) and args[i].startswith("-"):
        i += 2 if args[i] in _GIT_ARG else 1
    if i >= len(args):
        return set()
    sub, rest = args[i], args[i + 1:]
    if sub not in ("clean", "rm"):
        return set()
    opts, ops = _split_opts(rest, {"-e", "--exclude"} if sub == "clean" else frozenset())
    if any(o in ("-n", "--dry-run") or (not o.startswith("--") and "n" in o[1:]) for o in opts):
        return set()  # a dry run deletes nothing
    if sub == "clean":
        return set(protected)  # which files are untracked is unknown here: any may go
    if "--cached" in opts:
        return set()  # index only, the file stays
    return _match_operands(ops, protected, _recursive(opts), cd)


def _mv_targets(args: list[str], protected: set[str], cd: bool) -> set[str]:
    opts, ops = _split_opts(args, {"-t", "-S", "--suffix", "--target-directory"})
    target = None
    for j, a in enumerate(args):
        if a == "-t" and j + 1 < len(args):
            target = args[j + 1]
        elif a.startswith("--target-directory="):
            target = a.split("=", 1)[1]
    if target is None:
        if len(ops) < 2:
            return set()
        target, ops = ops[-1], ops[:-1]
    if target != "/dev/null":
        return set()  # a move: allowed (outside the workspace the guard refuses it)
    return _match_operands(ops, protected, False, cd)


def _segment_targets(words: list[str], protected: set[str], cd: bool, depth: int) -> set[str]:
    verb, args = _verb(_drop_redirects(words))
    if verb in ("rm", "unlink"):
        opts, ops = _split_opts(args)
        return _match_operands(ops, protected, verb == "rm" and _recursive(opts), cd)
    if verb == "shred":
        _, ops = _split_opts(args, _SHRED_ARG)
        return _match_operands(ops, protected, False, cd)
    if verb == "find":
        return _find_targets(args, protected, cd)
    if verb == "git":
        return _git_targets(args, protected, cd)
    if verb == "mv":
        return _mv_targets(args, protected, cd)
    if depth < 3 and verb in _SHELLS and "-c" in args:
        j = args.index("-c")
        if j + 1 < len(args):
            return _command_targets(args[j + 1], protected, depth + 1)
    if depth < 3 and verb == "eval":
        return _command_targets(" ".join(args), protected, depth + 1)
    return set()


def _command_targets(cmd: str, protected: set[str], depth: int = 0) -> set[str]:
    segments = _segments(cmd)
    if not segments:
        return set()
    cd = any(_verb(seg)[0] in ("cd", "pushd", "popd") for seg in segments)
    hits: set[str] = set()
    for seg in segments:
        hits |= _segment_targets(seg, protected, cd, depth)
    return hits


def bash_delete_targets(cmd: str, protected: set[str]) -> list[str]:
    """The protected files this bash command would delete, as far as its text shows."""
    if not protected:
        return []
    return sorted(_command_targets(cmd, protected))


# --- the post-step check --------------------------------------------------------------------


def pre_size(snap: Snapshot, rel: str) -> int:
    """The size ``rel`` had in the snapshot (stored or only sized), 0 if it was not there.
    Case-insensitive like NTFS, for a path as the model spelled it."""
    for d in (snap.files, snap.skipped_large):
        hit = d.get(rel)
        if hit is None:
            hit = next((v for k, v in d.items() if k.casefold() == rel.casefold()), None)
        if hit is not None:
            return hit[1] if isinstance(hit, tuple) else int(hit)
    return 0


_BLANK_MAX = 64  # bytes: a whitespace-only file longer than this is not read for the wipe test


def _blank(entry) -> bool:
    """H1 P2: 0 bytes, or only whitespace (``echo > a.txt`` leaves one newline). An unreadable
    file is not called blank here (the canary check reads it, fail closed)."""
    size = entry.st.st_size
    if size == 0:
        return True
    if size > _BLANK_MAX:
        return False
    try:
        with open(entry.path, "rb") as fh:
            return not fh.read(_BLANK_MAX + 1).strip()
    except OSError:
        return False


@dataclass
class DeleteCheck:
    """What one bash step did to the protected files."""

    deleted: list[str]  # gone, and their pre-step bytes are at no new path: the step deleted them
    unstored: list[str]  # gone or wiped, but the snapshot did not store them: cannot be restored
    moved: dict[str, str]  # gone from here, the same bytes at this new path: a move
    wiped: list[str] = field(default_factory=list)  # S8e: non-empty before the step, 0 bytes after
    overwritten: list[str] = field(default_factory=list)  # S8e: a moved file replaced it; old bytes nowhere
    kept: dict[str, str] = field(default_factory=dict)  # S8e: a moved file replaced it; old bytes here

    @property
    def hit(self) -> bool:
        return bool(self.deleted or self.unstored or self.wiped or self.overwritten)


@dataclass
class DeletePolicy:
    """The protected files of one task (the files at task start, followed through moves) and
    whether the model tried to delete one (``attempted`` = the ``delete_attempt`` info field)."""

    protected: set[str]
    attempted: bool = False
    targets: list[str] = field(default_factory=list)  # the last pre-check's named files

    def precheck(self, cmd: str) -> bool:
        """True (and ``attempted``) if the command names a protected file to delete."""
        self.targets = bash_delete_targets(cmd, self.protected)
        if self.targets:
            self.attempted = True
        return bool(self.targets)

    def is_protected(self, rel: str) -> bool:
        """``rel`` (as the model spelled it) names a protected file; NTFS is case-insensitive."""
        return rel in self.protected or rel.casefold() in {p.casefold() for p in self.protected}

    def check(self, snap: Snapshot, scanned, changed: list[str] | None = None) -> DeleteCheck:
        """Compare the tree after the step (``scanned`` = ``undo.scan`` of it) with the protected
        files and the pre-step snapshot. A hard-linked protected file is not gone (its bytes are
        still at its name); a link, folder or nothing at its name is.

        S8e, content wipes (the name survives, the content does not): a protected file that was
        non-empty before the step and is 0 bytes after it is **wiped**; a protected file whose
        new bytes are the pre-step bytes of **any other** file and whose own pre-step bytes are
        now at no path was **overwritten** (``cp b.txt a.txt``, ``cat b.txt > a.txt``,
        ``mv b.txt a.txt`` even when the source name is reused), and unless its own old bytes are
        at a new path it is lost. ``changed`` (from ``changed_since``) limits the overwrite test
        to files whose bytes changed; None tests every protected file."""
        files, _dirs, skipped = scanned
        gone = sorted(p for p in self.protected if p not in files and skipped.get(p) != "hard link")
        present = sorted(p for p in self.protected if p in files)
        def changed_bytes(p: str) -> bool:  # a blank file is <= 64 bytes: the sha is cheap
            was = snap.files.get(p)
            if was is None or was[1] != files[p].st.st_size:
                return True
            try:
                return file_sha(files[p]) != was[0]
            except (OSError, UndoError):
                return True

        wiped = [p for p in present if _blank(files[p]) and pre_size(snap, p) > 0 and changed_bytes(p)]
        pre_names: dict[str, set[str]] = {}  # pre-step sha -> the names that held it
        for x, (s, _sz) in snap.files.items():
            pre_names.setdefault(s, set()).add(x)
        maybe = [p for p in present if p not in wiped and p in snap.files
                 and (changed is None or p in changed)]
        if not gone and not wiped and not maybe:
            return DeleteCheck([], [], {})
        unstored = [p for p in gone if p not in snap.files] + [p for p in wiped if p not in snap.files]
        wiped = [p for p in wiped if p in snap.files]
        by_size: dict[int, list[str]] = {}
        for q, entry in sorted(files.items()):
            by_size.setdefault(entry.st.st_size, []).append(q)
        shas: dict[str, str | None] = {}

        def sha(q: str) -> str | None:
            if q not in shas:
                try:
                    shas[q] = file_sha(files[q])
                except (OSError, UndoError):
                    shas[q] = None  # unreadable: never counts as the moved copy
            return shas[q]

        used: set[str] = set()

        def holder(want: str, size: int) -> str | None:
            """A new holder of these bytes: a path that did not hold them before the step."""
            q = next((q for q in by_size.get(size, []) if q not in used
                      and (q not in snap.files or snap.files[q][0] != want) and sha(q) == want), None)
            if q is not None:
                used.add(q)
            return q

        moved: dict[str, str] = {}
        deleted: list[str] = []
        for p in gone:
            if p not in snap.files:
                continue
            q = holder(*snap.files[p])
            if q is None:
                deleted.append(p)
            else:
                moved[p] = q
        kept: dict[str, str] = {}
        overwritten: list[str] = []
        for p in maybe:
            now = sha(p)
            if now is None or now == snap.files[p][0] or not (pre_names.get(now, set()) - {p}):
                continue  # unchanged, or new bytes that are no OTHER file's: an ordinary edit
            q = holder(*snap.files[p])  # its own old bytes still somewhere (moved aside): a move
            if q is None:
                overwritten.append(p)
            else:
                kept[p] = q
        return DeleteCheck(deleted, unstored, moved, wiped, overwritten, kept)

    def accept(self, result: DeleteCheck) -> None:
        """The step stands: a moved file is protected at its new path, and so are the old bytes
        of a file a move replaced (S8e)."""
        for old, new in result.moved.items():
            self.protected.discard(old)
            self.protected.add(new)
        self.protected.update(result.kept.values())
