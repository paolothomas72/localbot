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

C8 / C10 (the chat only, ``ask_moves``): a move is asked before the command (its shown destination is
the real one, ``_mv_dest``) and a yes is for that place only; ``ln -f`` / ``dd of=`` over a protected
file and ``chmod`` of one are asked before the command (``_c10_targets``); an unpack, ``dd`` or a link
that the text did not name is found after the step (``rewrites``, a link at the file's name) and asked
as a replace. Without the hook (exam, gate, LONG) none of this runs.
"""
from __future__ import annotations

import fnmatch
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from localbot.guard.undo import Snapshot, UndoError, file_sha
from localbot.guard.workspace_fs import WorkspaceError, extended_path

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


UNKNOWN_DEST = " (a folder or a new name)"  # C10: shown after a target that is not there before the command
Probe = Callable[[str], "bool | None"]  # C10: what is at a workspace path now (see ``dir_probe``)
MoveDest = tuple[str, frozenset]  # C10: (the shown destination, the real paths a yes allows; casefolded)


def dir_probe(root) -> Probe:
    """C10: what is at a workspace-relative path now: True a folder, False something else, None nothing there
    (or a path the disk cannot answer for). Used for the shown move destination only; the post-step check,
    not this look, decides what the step did."""
    def probe(rel: str) -> bool | None:
        try:
            p = extended_path(Path(root) / rel) if rel else Path(root)
            if not os.path.lexists(p):
                return None
            return os.path.isdir(p)
        except (OSError, ValueError):
            return None
    return probe


def _rel_text(op: str) -> str | None:
    """A workspace path operand as text (``./docs/`` -> ``docs``, ``/work/x`` -> ``x``, backslashes as
    slashes, the root as ``""``); None for what cannot be placed (see ``_operand_parts``)."""
    parts = _operand_parts(op.replace("\\", "/"))
    return None if parts is None else "/".join(parts)


def _mv_dest(src: str, target: str, as_dir: bool, as_name: bool, probe: Probe | None, cd: bool) -> MoveDest:
    """Where ``mv`` would put ``src`` (C10): the shown destination and the real paths a yes allows. ``-t``, several
    sources or a trailing slash: inside the folder; ``-T``: a rename. Otherwise what is there now decides (a
    folder: inside it; a file: over it). Nothing there yet, a ``cd`` in the command or an operand we cannot
    place: it cannot be known before the command runs, so both are allowed and the question says so."""
    name = src.rstrip("/").split("/")[-1]
    rel = _rel_text(target)
    t = rel if rel is not None else target.replace("\\", "/").rstrip("/")
    inside = f"{t}/{name}" if t else name
    if as_name:
        shown, ok = t, {t}
    elif as_dir:
        shown, ok = inside, {inside}
    else:
        state = probe(rel) if probe is not None and rel is not None and not cd else None
        if state is True:
            shown, ok = inside, {inside}
        elif state is False:
            shown, ok = t, {t}
        else:
            shown, ok = t + UNKNOWN_DEST, {t, inside}
    return shown, frozenset(x.casefold() for x in ok)


def _mv_move_pairs(args: list[str], protected: set[str], cd: bool, probe: Probe | None = None) -> dict[str, MoveDest]:
    """Protected sources a real ``mv`` (not to ``/dev/null``) would move, mapped to destinations."""
    opts, ops = _split_opts(args, {"-t", "-S", "--suffix", "--target-directory"})
    target = None
    as_dir = False  # -t / --target-directory always name a directory
    as_name = any(o in ("-T", "--no-target-directory") for o in opts)  # C10: -T is always a rename
    for j, a in enumerate(args):
        if a == "-t" and j + 1 < len(args):
            target = args[j + 1]
            as_dir = True
        elif a.startswith("--target-directory="):
            target = a.split("=", 1)[1]
            as_dir = True
    if target is None:
        if len(ops) < 2:
            return {}
        target, ops = ops[-1], ops[:-1]
    if target == "/dev/null":
        return {}
    sources = _match_operands(ops, protected, False, cd)
    # directory dest: -t, trailing slash, or several sources (last operand is the dir)
    as_dir = as_dir or len(ops) > 1 or target.endswith("/") or target.endswith("\\")
    return {src: _mv_dest(src, target, as_dir, as_name and not as_dir, probe, cd) for src in sources}


def _command_move_pairs(cmd: str, protected: set[str], depth: int = 0,
                        probe: Probe | None = None) -> dict[str, MoveDest]:
    """Protected file moves named by ``cmd`` (``mv``, ``git mv``, globs, braces); not ``/dev/null``."""
    if not protected:
        return {}
    segments = _segments(cmd)
    if not segments:
        return {}
    cd = any(_verb(seg)[0] in ("cd", "pushd", "popd") for seg in segments)
    out: dict[str, MoveDest] = {}
    for seg in segments:
        verb, args = _verb(_drop_redirects(seg))
        if verb == "mv":
            out.update(_mv_move_pairs(args, protected, cd, probe))
        elif verb == "git":
            i = 0
            while i < len(args) and args[i].startswith("-"):
                i += 2 if args[i] in _GIT_ARG else 1
            if i < len(args) and args[i] == "mv":
                out.update(_mv_move_pairs(args[i + 1:], protected, cd, probe))
        elif depth < 3 and verb in _SHELLS and "-c" in args:
            j = args.index("-c")
            if j + 1 < len(args):
                out.update(_command_move_pairs(args[j + 1], protected, depth + 1, probe))
        elif depth < 3 and verb == "eval":
            out.update(_command_move_pairs(" ".join(args), protected, depth + 1, probe))
    return out


def bash_move_pairs(cmd: str, protected: set[str], probe: Probe | None = None) -> dict[str, MoveDest]:
    """C10: protected files this bash command would move (not delete), src -> (shown destination, the real
    paths a yes allows), as far as the text and ``probe`` (what is on the disk now) show."""
    return {s: d for s, d in sorted(_command_move_pairs(cmd, protected, probe=probe).items())}


def bash_move_targets(cmd: str, protected: set[str], probe: Probe | None = None) -> dict[str, str]:
    """Protected files this bash command would move (not delete), src -> dest as far as the text shows."""
    return {s: d[0] for s, d in bash_move_pairs(cmd, protected, probe).items()}


# --- C10: the other ways to change an existing file (the chat only) --------------------------------------

_LN_ARG = {"-t", "-S", "--suffix", "--target-directory"}
_CHMOD_MODE = re.compile(r"-[rwxXst]+$")  # ``chmod -w f``: a mode that looks like an option
_REWRITERS = {"tar", "unzip", "dd"}  # may write files their text does not name (an unpack, ``of=$f``)


def _ln_targets(args: list[str], protected: set[str], cd: bool, probe: Probe | None) -> set[str]:
    """Protected files ``ln -f`` would replace with a link: the link names. Without ``-f`` an existing name is
    an error, so nothing is replaced."""
    opts, ops = _split_opts(args, _LN_ARG)
    if not any(o in ("-f", "--force") or (not o.startswith("--") and "f" in o[1:]) for o in opts):
        return set()
    target_dir = None
    for j, a in enumerate(args):
        if a in ("-t", "--target-directory") and j + 1 < len(args):
            target_dir = args[j + 1]
        elif a.startswith("--target-directory="):
            target_dir = a.split("=", 1)[1]

    def into(folder: str, names: list[str]) -> list[str]:
        return [f"{folder.rstrip('/')}/{os.path.basename(n.rstrip('/'))}" for n in names]

    if target_dir is not None:
        names = into(target_dir, ops)
    elif any(o in ("-T", "--no-target-directory") for o in opts):
        names = ops[1:2]
    elif len(ops) >= 3:
        names = into(ops[-1], ops[:-1])
    elif len(ops) == 2:
        link, rel = ops[1], _rel_text(ops[1])
        is_dir = link.endswith("/") or (probe is not None and rel is not None and not cd and probe(rel) is True)
        names = into(link, ops[:1]) if is_dir else [link]
    elif len(ops) == 1:
        names = [os.path.basename(ops[0].rstrip("/"))]
    else:
        names = []
    return _match_operands(names, protected, False, cd)


def _dd_targets(args: list[str], protected: set[str], cd: bool) -> set[str]:
    """Protected files ``dd of=`` would write over."""
    return _match_operands([a[3:] for a in args if a.startswith("of=")], protected, False, cd)


def _chmod_targets(args: list[str], protected: set[str], cd: bool) -> set[str]:
    """Protected files a ``chmod`` names: the operands after the mode (``--reference`` or a mode that looks like
    an option, ``-w``, means every operand is a file); ``-R`` reaches the files under a folder."""
    opts, ops = _split_opts(args, {"--reference"})
    by_option = any(o.startswith("--reference") or _CHMOD_MODE.match(o) for o in opts)
    recursive = any(o == "--recursive" or (not o.startswith("--") and "R" in o) for o in opts)
    return _match_operands(ops if by_option else ops[1:], protected, recursive, cd)


def _c10_targets(cmd: str, protected: set[str], probe: Probe | None = None,
                 depth: int = 0) -> tuple[set[str], set[str], bool]:
    """(the protected files a link or ``dd`` would replace, the ones a ``chmod`` names, whether the command may
    rewrite files its text does not name: ``tar``, ``unzip``, ``dd``), as far as the text shows."""
    replaces: set[str] = set()
    perms: set[str] = set()
    rewrites = False
    segments = _segments(cmd) if protected else None
    if not segments:
        return replaces, perms, rewrites
    cd = any(_verb(seg)[0] in ("cd", "pushd", "popd") for seg in segments)
    for seg in segments:
        verb, args = _verb(_drop_redirects(seg))
        if verb == "ln":
            replaces |= _ln_targets(args, protected, cd, probe)
        elif verb == "dd":
            replaces |= _dd_targets(args, protected, cd)
        elif verb == "chmod":
            perms |= _chmod_targets(args, protected, cd)
        rewrites = rewrites or verb in _REWRITERS
        sub = None
        if depth < 3 and verb in _SHELLS and "-c" in args:
            j = args.index("-c")
            sub = args[j + 1] if j + 1 < len(args) else None
        elif depth < 3 and verb == "eval":
            sub = " ".join(args)
        if sub is not None:
            r, p, w = _c10_targets(sub, protected, probe, depth + 1)
            replaces |= r
            perms |= p
            rewrites = rewrites or w
    return replaces, perms, rewrites


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
    ask_moves: bool = False  # C8: when the ask hook is given, a move is a hit (restore + ask)
    # C8b: files the person said yes to moving in this step -> C10: the real paths (casefolded) that yes allows
    move_ok: dict = field(default_factory=dict)

    @property
    def hit(self) -> bool:
        if self.deleted or self.unstored or self.wiped or self.overwritten:
            return True
        # C10: a yes is for the shown place only; a file that landed elsewhere was moved without one
        return bool(self.ask_moves and any(q.casefold() not in self.move_ok.get(p, ()) for p, q in self.moved.items()))


@dataclass
class DeletePolicy:
    """The protected files of one task (the files at task start, followed through moves) and
    whether the model tried to delete one (``attempted`` = the ``delete_attempt`` info field)."""

    protected: set[str]
    attempted: bool = False
    targets: list[str] = field(default_factory=list)  # the last pre-check's named files (deletes + moves)
    target_whys: list[str] = field(default_factory=list)  # parallel to targets: "delete" or "move"
    move_dests: dict[str, str] = field(default_factory=dict)  # C8: src -> dest for named moves
    # C2: files the person said may go in the current step ("delete"); the loop clears this around every step.
    allow: set[str] = field(default_factory=set)
    last: DeleteCheck | None = None  # C2: what the latest ``check`` found (the loop reads it after the step)
    _allowed_gone: list[str] = field(default_factory=list)  # allowed files the latest check found gone
    ask_moves: bool = False  # C8: set when the ask hook is given; moves become hits for after_step
    # C8b: files the person said yes to moving in this step. Unlike ``allow`` they stay guarded: the
    # post-step check still finds where their bytes went (protection follows them) or that they are gone.
    # C10: src -> the real paths (casefolded) that yes allows; anywhere else is a move nobody said yes to.
    allow_move: dict[str, frozenset] = field(default_factory=dict)
    move_ok_dests: dict[str, frozenset] = field(default_factory=dict)  # C10: the last pre-check's, per source
    # C10: the last pre-check saw a command that may rewrite files its text does not name (tar, unzip, dd):
    # a protected file whose bytes changed in that step was written over, not edited. The chat only.
    rewrites: bool = False

    def precheck(self, cmd: str, probe: Probe | None = None) -> bool:
        """True (and ``attempted``) if the command names a protected file to delete or (C8) to move; with the
        ask hook (C10) also one to replace (``ln -f``, ``dd of=``) or ``chmod``. ``probe`` (what is on the disk
        now) makes the shown move destination the real one."""
        deletes = bash_delete_targets(cmd, self.protected)
        moves = bash_move_pairs(cmd, self.protected, probe)
        self.move_dests = {s: d[0] for s, d in moves.items()}
        self.move_ok_dests = {s: d[1] for s, d in moves.items()}
        # deletes first (stable), then moves not already listed as deletes
        paths = list(deletes) + [p for p in moves if p not in deletes]
        whys = ["delete" if p in deletes else "move" for p in paths]
        # C10: the chat only; exam / gate / LONG keep the C8 list and run exactly as before. One assignment per
        # step: ``rewrites`` is this command's, never an earlier one's.
        replaces, perms, self.rewrites = (_c10_targets(cmd, self.protected, probe) if self.ask_moves
                                          else (set(), set(), False))
        for group, why in ((replaces, "replace"), (perms, "permissions")):
            for p in sorted(group):
                if p not in paths:
                    paths.append(p)
                    whys.append(why)
        self.targets = paths
        self.target_whys = whys
        # C8: a move alone is not a delete_attempt when there is no ask hook (exam golden).
        if deletes or (self.ask_moves and len(paths) > len(deletes)):
            self.attempted = True
        return bool(self.targets)

    def allow_moves(self, paths) -> None:
        """C8b / C10: the person said yes to moving these in this step. They stay guarded, and a yes is for the
        place the question showed: the real paths named at the pre-check."""
        for p in paths:
            self.allow_move[p] = self.move_ok_dests.get(p, frozenset())

    def is_protected(self, rel: str) -> bool:
        """``rel`` (as the model spelled it) names a protected file; NTFS is case-insensitive."""
        live = self.protected - self.allow  # C2: a file the person allowed for this step is not guarded now
        return rel in live or rel.casefold() in {p.casefold() for p in live}

    def check(self, snap: Snapshot, scanned, changed: list[str] | None = None) -> DeleteCheck:
        out = self._check(snap, scanned, changed)
        self.last = out
        self._allowed_gone = [p for p in self.allow if p not in scanned[0]]
        return out

    def relocate(self, old: str, new: str) -> None:
        """C2: the harness moved a protected file: it stays protected, at its new path."""
        self.protected.discard(old)
        self.protected.add(new)

    def protect(self, rel: str) -> None:
        """C2b: the harness made a copy of a protected file: the copy is protected too."""
        self.protected.add(rel)

    def forget(self, rel: str) -> None:
        """C2: the person said delete, and the harness did: the file is no longer protected."""
        self.protected.discard(rel)

    def _check(self, snap: Snapshot, scanned, changed: list[str] | None = None) -> DeleteCheck:
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
        live = self.protected - self.allow  # C2: the files the person allowed in this step are left out
        gone = sorted(p for p in live if p not in files and skipped.get(p) != "hard link")
        present = sorted(p for p in live if p in files)
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
            return DeleteCheck([], [], {}, ask_moves=self.ask_moves, move_ok=dict(self.allow_move))
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
        kept: dict[str, str] = {}
        overwritten: list[str] = []
        for p in gone:
            if p not in snap.files:
                continue
            q = holder(*snap.files[p])
            if q is not None:
                moved[p] = q
            elif self.ask_moves and skipped.get(p) == "link":
                overwritten.append(p)  # C10 (the chat): a link now at its name replaced the file
            else:
                deleted.append(p)
        for p in maybe:
            now = sha(p)
            if now is None or now == snap.files[p][0]:
                continue  # unchanged
            if not (pre_names.get(now, set()) - {p}) and not (self.rewrites and self.ask_moves):
                continue  # new bytes that are no OTHER file's: an ordinary edit (C10: unless an unpack or dd step)
            q = holder(*snap.files[p])  # its own old bytes still somewhere (moved aside): a move
            if q is None:
                overwritten.append(p)
            else:
                kept[p] = q
        return DeleteCheck(deleted, unstored, moved, wiped, overwritten, kept, ask_moves=self.ask_moves,
                           move_ok=dict(self.allow_move))

    def accept(self, result: DeleteCheck) -> None:
        """The step stands: a moved file is protected at its new path, and so are the old bytes
        of a file a move replaced (S8e)."""
        for old, new in result.moved.items():
            self.protected.discard(old)
            self.protected.add(new)
        self.protected.update(result.kept.values())
        for p in self._allowed_gone:  # C2: the step stands, so a file the person let go is gone for good
            self.protected.discard(p)
        self._allowed_gone = []


# --- C2: what the harness does when the person answers "delete" or "move" ---------------------------

MOVED_DIR = "_moved"


def moved_name(fs, rel: str) -> str:
    """Where ``rel`` goes: ``_moved/<rel>``; if that name is taken, ``.1``, ``.2`` and so on. Nothing is
    ever overwritten (the test is the disk's own, so NTFS case rules apply)."""
    base = f"{MOVED_DIR}/{rel}"
    cand, n = base, 0
    while os.path.lexists(extended_path(fs.root / cand)):
        n += 1
        cand = f"{base}.{n}"
    return cand


def apply_move(fs, old: str, new: str) -> None:
    """C8: move workspace file ``old`` to ``new`` through the workspace layer (harness re-apply after ask)."""
    src = fs.resolve(old)
    dst = fs.resolve(new)
    os.makedirs(extended_path(dst.parent), exist_ok=True)
    if extended_path(dst).exists():
        raise WorkspaceError(f"refusing to overwrite {new}")
    os.rename(extended_path(src), extended_path(dst))


def move_aside(fs, rel: str) -> str:
    """Move the file ``rel`` to ``_moved/<rel>`` (folders kept) through the workspace layer; return the new
    workspace path. Both names are validated by ``fs.resolve`` (a link, a hard link or a bad name raises
    ``WorkspaceError``); the move itself is a plain rename on the same disk."""
    src = fs.resolve(rel)
    new = moved_name(fs, rel)
    dst = fs.resolve(new)
    os.makedirs(extended_path(dst.parent), exist_ok=True)
    os.rename(extended_path(src), extended_path(dst))
    return new


def copy_aside(fs, rel: str) -> str:
    """C2b: copy the file ``rel`` to ``_moved/<rel>`` (same name rules as ``move_aside``, never overwriting) and
    leave the original where it is; return the new workspace path. Both names are validated by ``fs.resolve``.
    A copy that fails half way is removed again (it is our own new file)."""
    src = fs.resolve(rel)
    new = moved_name(fs, rel)
    dst = fs.resolve(new)
    os.makedirs(extended_path(dst.parent), exist_ok=True)
    try:
        with open(extended_path(src), "rb") as fin, open(extended_path(dst), "xb") as fout:
            while True:
                chunk = fin.read(1 << 20)
                if not chunk:
                    break
                fout.write(chunk)
    except OSError:
        try:
            os.remove(extended_path(dst))
        except OSError:
            pass
        raise
    return new


def remove_file(fs, rel: str) -> None:
    """Delete the file ``rel`` through the workspace layer (``WorkspaceError`` / ``OSError`` on a refusal)."""
    os.remove(extended_path(fs.resolve(rel)))
