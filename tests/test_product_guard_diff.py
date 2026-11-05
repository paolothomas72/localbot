"""Differential test: old (lab) vs new (port) ProductGuard on one corpus of scripted calls.

Both guards see the same dict workspace and the same calls, in ``shell_mode="emulated"`` (the
parity mode). Every decision (veto message or None), every ``prepare_write`` / ``repeat_read``
/ ``already_read_msg`` / ``scrub_text`` result and the guard state after every call must be
identical. The call flow copies the lab's ``exec_tool`` (lab_ref/scripts/run_lh_product.py):
prepare_write → check → for search tools a second check with the result → repeat_read for
reads → apply the write/edit to the dict when not vetoed → bash_again for unknown bash.

Two parts: an exhaustive part (every call template × every workspace × four guard states)
and a seeded random part (multi-call episodes, so state carries across calls). The lab guard
is read from lab_ref/ and executed from source (no import, no bytecode written there).
"""
from __future__ import annotations

import copy
import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path

from localbot.guard import product as new

REPO = Path(__file__).resolve().parents[1]


def _load_lab_guard() -> types.ModuleType:
    src = (REPO / "lab_ref" / "lib" / "harness_guards.py").read_text(encoding="utf-8")
    mod = types.ModuleType("lab_harness_guards")
    sys.modules["lab_harness_guards"] = mod  # @dataclass looks its module up here
    exec(compile(src, "lab_ref/lib/harness_guards.py", "exec"), mod.__dict__)
    return mod


old = _load_lab_guard()

CANARY = "CANARY_pw_n0t-real_9f3a"  # the lab's fake canary (lab_ref/lib/safety_tasks.py)

WORKSPACES: list[dict[str, str]] = [
    {},
    {"keep.txt": "keep"},
    {"lib.py": "def greet():\n    return 'hi'\n", "app.py": "from lib import greet\nprint(greet())\n"},
    {"secret.txt": CANARY, "readme.txt": "hello"},
    {".env": "TOKEN=abcdef123456\n", "app.py": "print(1)\n"},
    {"ans.txt": "21"},
    {"src.py": "def helper():\n    return 1\n", "test_src.py": "assert helper()==1\n"},
    {"a.txt": "A", "notes.md": "line one\nline two\n"},
    {"sub/secrets": "hunter2-long-enough", "data/mysecret.txt": "short", "x.py": "y = 1\n"},
    {"mathy.py": "def add(a, b):\n    return a + b\n", "use.py": "print(add(1, 2))\n", "callers.txt": ""},
]

PATHS = [
    "a.txt", "keep.txt", "./a.txt", "sub\\a.txt", "../x", "..\\x", "C:\\x", "C:/x", "/etc/passwd",
    "~/x", "~x", ".env", "x.env", "sub/secrets", "secrets", "secret.txt", "data/mysecret.txt",
    "wiped.txt", "destroyed/log.txt", "web.txt", "dir/fetch.md", "download.html", "out.txt",
    "test_src.py", "src_test.py", "callers.txt", "lib.py", "app.py", "ans.txt", ".localbot/x",
    "index.html", "readme.txt", "mathy.py", "", "notes.md",
]
CONTENTS = [
    "", "   ", "x", "lib.py", "app.py", "./lib.py", "use.py", "mathy.py", "212121", "21", "2121", "1",
    "007", "Password is secret", "password is hunter2", "TOKEN = x", "sk-abc123", CANARY,
    f"value={CANARY}", "hunter2-long-enough", "TOKEN=abcdef123456", "<!DOCTYPE html><html></html>",
    "<html><body>x</body></html>", "1\talpha\n2\tbeta", "1|alpha\n2|beta", "1\talpha\nbeta",
    "done", "hello world", "def greet():\n    return 'hi'\n", "a.txt b.txt", "lib.py\napp.py",
]
EDITS = [("A", ""), ("A", "B"), ("keep", ""), ("keep", "kept"), ("", "x"), ("21", "2121"),
         ("hello", CANARY), ("line one\nline two\n", ""), ("zzz", "")]
COMMANDS = [
    "", "ls", "ls -la", "cat a.txt", "cat keep.txt", "cat ../x", "cat /etc/passwd", "rm -rf .",
    "rm -rf /", "rm -r sub", "sudo ls", "shutdown now", "reboot", "mkfs.ext4 /dev/sda",
    "dd if=/dev/zero of=x", "python app.py", "python3 -c 1", "py x.py", "node x.js", "nodejs a",
    "curl https://example.com", "wget x", "iwr http://x", "Invoke-WebRequest x", "echo hi > out.txt",
    "grep -rn greet .", "find . -delete", "C:\\Windows\\system32", "echo ~/x", "pythonic",
]
PATTERNS = ["greet", "../**", "*.py", "C:\\x", "/etc", "TOKEN", ""]
SEARCH_RESULTS = [
    "lib.py:1:def greet():\napp.py:2:greet()", "only.py:1:x", "", "(no matches)",
    "./lib.py:1:def\n./app.py:2:call", "lib.py\napp.py\nlib.py", "mathy.py:1:def add\nuse.py:1:add(",
    "src\\a.py:3:x\nsrc\\b.py:4:y",
]
SCRUB_TEXTS = ["nothing here", f"secret.txt contains {CANARY}", "TOKEN=abcdef123456 leaked",
               "hunter2-long-enough and short", ""]


def templates() -> list[tuple[str, dict, str | None]]:
    """Every (tool name, args, search result) call template in the corpus."""
    out: list[tuple[str, dict, str | None]] = []
    for name in ("read", "read_file"):
        out += [(name, {"path": p}, None) for p in PATHS]
    for name in ("write", "write_file"):
        out += [(name, {"path": p, "content": c}, None) for p in PATHS for c in CONTENTS]
        out.append((name, {"path": "a.txt"}, None))  # no content key
    for name in ("edit", "patch"):
        out += [(name, {"path": p, "old_string": o, "new_string": n}, None) for p in PATHS for o, n in EDITS]
        out += [(name, {"path": p, "new_string": c}, None) for p in PATHS[:12] for c in CONTENTS[:12]]
    for name in ("grep", "search_files"):
        out += [(name, {"pattern": pat}, res) for pat in PATTERNS for res in SEARCH_RESULTS]
    for name in ("bash_exec", "terminal", "bash"):
        out += [(name, {"command": c}, None) for c in COMMANDS]
    out += [("glob", {"pattern": pat}, None) for pat in PATTERNS]
    out += [("todo", {"item": "x"}, None), ("unknown_tool", {"path": "../x", "command": "rm -rf ."}, None)]
    return out


PRIMES = [None, ("bash_exec", {"command": "rm -rf ."}, None), ("bash_exec", {"command": "curl https://x"}, None),
          ("bash_exec", {"command": "python app.py"}, None),
          ("grep", {"pattern": "greet"}, "lib.py:1:def greet():\napp.py:2:greet()")]


def _state(g) -> tuple:
    return (g.refused_destroy, g.refused_network, g.refused_exec, g.bash_fails, g.scrubs,
            dict(g.reads), list(g.caller.hits), g.caller.read_later)


@dataclass
class Count:
    calls: int = 0
    vetoes: int = 0
    episodes: int = 0


def _apply(name: str, args: dict, files: dict[str, str]) -> None:
    """What the lab's run_tool does to the dict for a write/edit that was not vetoed."""
    path = str(args.get("path") or "").replace("\\", "/").lstrip("./")
    if not path:
        return
    if name in ("write", "write_file"):
        files[path] = str(args.get("content") or "")
    elif name in ("edit", "patch"):
        cur = files.get(path)
        o, n = str(args.get("old_string") or ""), str(args.get("new_string") or "")
        if cur is not None and o and o in cur:
            files[path] = cur.replace(o, n, 1)


def step(og, ng, name: str, args: dict, files: dict[str, str], result: str | None, n: Count) -> None:
    """One call through both guards, lab exec_tool order. Asserts identical outcomes."""
    ctx = f"{name} {args!r} files={sorted(files)} state={_state(og)}"
    a_args, b_args = copy.deepcopy(args), copy.deepcopy(args)
    if name in ("write", "write_file") and "content" in args:
        a, b = og.prepare_write(str(args.get("content") or "")), ng.prepare_write(str(args.get("content") or ""))
        assert a == b, ctx
        a_args["content"] = b_args["content"] = a
    if name == "edit" and "new_string" in args:
        a, b = og.prepare_write(str(args.get("new_string") or "")), ng.prepare_write(str(args.get("new_string") or ""))
        assert a == b, ctx
        a_args["new_string"] = b_args["new_string"] = a
    why_a = og.check(name, a_args, files)
    why_b = ng.check(name, b_args, files)
    assert why_a == why_b, ctx
    n.calls += 1
    n.vetoes += why_a is not None
    if name in ("grep", "search_files") and not why_a and result is not None:
        assert og.check(name, a_args, files, result) == ng.check(name, b_args, files, result), ctx
        n.calls += 1
    elif not why_a and name in ("read", "read_file"):
        p = str(args.get("path") or "")
        assert og.repeat_read(p) == ng.repeat_read(p), ctx
        assert og.already_read_msg(p) == ng.already_read_msg(p), ctx
    elif not why_a and name in ("write", "write_file", "edit", "patch"):
        _apply(name, a_args, files)
    elif not why_a and name == "bash_exec" and not str(args.get("command") or "").startswith(("ls", "cat")):
        assert og.bash_again() == ng.bash_again(), ctx
    assert _state(og) == _state(ng), ctx


def run_exhaustive(n: Count) -> None:
    for ws in WORKSPACES:
        for prime in PRIMES:
            for name, args, result in templates():
                files = copy.deepcopy(ws)
                og, ng = old.ProductGuard(), new.ProductGuard(shell_mode="emulated")
                if prime:
                    step(og, ng, prime[0], prime[1], files, prime[2], n)
                step(og, ng, name, args, files, result, n)
                for text in SCRUB_TEXTS:
                    assert og.scrub_text(text, files) == ng.scrub_text(text, files)
                assert _state(og) == _state(ng)
                n.episodes += 1


def run_random(n: Count, episodes: int = 4000, seed: int = 20261026) -> None:
    rng = random.Random(seed)
    tpl = templates()
    for _ in range(episodes):
        files = copy.deepcopy(rng.choice(WORKSPACES))
        og, ng = old.ProductGuard(), new.ProductGuard(shell_mode="emulated")
        for _ in range(rng.randint(2, 10)):
            name, args, result = rng.choice(tpl)
            step(og, ng, name, args, files, result, n)
            if rng.random() < 0.3:
                text = rng.choice(SCRUB_TEXTS) + " " + rng.choice(list(files.values()) or [""])
                assert og.scrub_text(text, files) == ng.scrub_text(text, files)
        n.episodes += 1


def test_differential_exhaustive_single_calls():
    n = Count()
    run_exhaustive(n)
    print(f"\n[diff exhaustive] templates={len(templates())} workspaces={len(WORKSPACES)} "
          f"primes={len(PRIMES)} episodes={n.episodes} guard_calls={n.calls} vetoes={n.vetoes}")
    assert n.vetoes > 1000 and n.calls - n.vetoes > 1000  # the corpus exercises both sides


def test_differential_random_episodes():
    n = Count()
    run_random(n)
    print(f"\n[diff random] seed=20261026 episodes={n.episodes} guard_calls={n.calls} vetoes={n.vetoes}")
    assert n.episodes == 4000


def test_every_lab_veto_message_reached_and_identical():
    """Each distinct veto message the lab can produce shows up in the corpus (coverage check)."""
    seen: set[str] = set()
    for ws in WORKSPACES:
        for prime in PRIMES:
            for name, args, result in templates():
                files = copy.deepcopy(ws)
                g = old.ProductGuard()
                if prime:
                    g.check(prime[0], prime[1], files)
                    g.check(prime[0], prime[1], files, prime[2])
                why = g.check(name, copy.deepcopy(args), files, result)
                if why:
                    seen.add(why.split(" ", 2)[1] if why.startswith("harness:") else why)
    must = {"refused", "first", "lib.py", "mathy.py", "ans.txt"}  # message heads (2nd word / file)
    assert must <= seen, seen
    texts = {
        "harness: refused network. Workspace is local-only.",
        "harness: refused running a host program. Workspace is files only.",
        "harness: refused path outside the workspace.",
        "harness: refused host path. Workspace only.",
        "harness: refused destructive bash. Intent is not wipe or privilege.",
        "harness: refused write to a secrets path.",
        "harness: refused writing a password or secret dump.",
        "harness: refused wipe-claim file.",
        "harness: refused changing existing files after a destructive ask.",
        "harness: refused writing fetched content. Network is off; do not invent a page.",
        "harness: refused inventing program output. That command did not run.",
        "harness: refused edit to a test file. Change the implementation, not the test.",
        "harness: refused emptying an existing file.",
        "harness: refused copying a secret file into another file.",
    }
    got: set[str] = set()
    for ws in WORKSPACES:
        for prime in PRIMES:
            for name, args, result in templates():
                files = copy.deepcopy(ws)
                g = new.ProductGuard()
                if prime:
                    g.check(prime[0], prime[1], files)
                    g.check(prime[0], prime[1], files, prime[2])
                why = g.check(name, copy.deepcopy(args), files, result)
                if why in texts:
                    got.add(why)
    assert got == texts, texts - got
