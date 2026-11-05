"""S8f.1a — flag suspicious tries in saved exam results (read-only, run after every batch).

Lists, from results/stage1 result JSONs:
  (a) failing tries where the model still ended "done" (it thought it was finished),
  (b) false successes — a write/edit whose reply was "Wrote"/"Updated" on a call the arg gate
      would reject (missing / wrong-case / non-string key): the lab bug where a misnamed key
      became "" and the tool reported success with the disk unchanged,
  (c) arg_errors / arg_extra per model (present only in S8f.1a-and-later runs).

It opens result files for reading only; it prints tool names, arg KEY names and counts, never
an argument VALUE (rows are already redacted; this adds no new exposure). On the committed files
it must surface the 4 nemotron h_intent_tests tries (2 files x 2 calls).

Arguments (H1, E-1): each one is a file path or a glob, so a quoted glob and the file list that
the shell makes of an unquoted one scan the same files. Matches are de-duplicated (one file reached
through a path and a glob counts once). The header names every argument with how many files it
matched; an argument that matched no file gets its own WARNING line. Only result-shaped files (a
"tasks" dict) are scanned; other JSON in the folder (g2_*, sample_first_turn_*, probe_*) is skipped.

Usage:  python -I scripts/suspicious_tries.py [PATH_OR_GLOB ...]
        default (no arguments): results/stage1/*.json (relative to the current directory)
        quote globs anyway: with a shell's nullglob on, an unmatched unquoted glob vanishes, and with
        no argument left the default runs instead (the header then says "no arguments, default")

Exit codes:
  0  at least one try checked; the VERDICT line says "FALSE SUCCESS FOUND" or
     "no false success" (a found false success does not change the exit code; an argument
     that matched nothing next to ones that did is a WARNING line only)
  2  VERDICT: NOTHING SCANNED — no try was checked: no result-shaped file among the matches
     (empty glob, wrong directory, a typo, or only other-shaped JSON), or result files whose
     "tasks" is {} / whose trials lists are all empty; never read this as "no false success"
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for -I

from localbot.config import get_profile  # noqa: E402
from localbot.guard.toolargs import check_tool_args  # noqa: E402

_SCHEMA_CACHE: dict[str, object] = {}


def _schemas(profile_name: str):
    if profile_name not in _SCHEMA_CACHE:
        try:
            _SCHEMA_CACHE[profile_name] = get_profile(profile_name).tools.schemas
        except KeyError:
            _SCHEMA_CACHE[profile_name] = ()
    return _SCHEMA_CACHE[profile_name]


def scan(paths: list[str]) -> dict:
    done_fail: list[str] = []
    false_success: list[str] = []
    arg_err_by_model: dict[str, int] = {}
    arg_extra_by_model: dict[str, int] = {}
    files = tries = 0
    for p in sorted(paths):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(d, dict) or not isinstance(d.get("tasks"), dict):
            continue
        files += 1
        model, profile = d.get("model", "?"), d.get("profile", "?")
        schemas = _schemas(profile)
        fname = p.replace("\\", "/").rsplit("/", 1)[-1]
        for tid, t in d["tasks"].items():
            for r in t.get("trials", []):
                tries += 1
                k = r.get("k")
                if not r.get("pass") and r.get("end_reason") == "done":
                    done_fail.append(f"{fname} :: {tid} t{k} (model {model})")
                for c in r.get("calls", []):
                    name, args = c.get("name"), c.get("args") or {}
                    reply = str(c.get("result") or "")
                    if not isinstance(args, dict):
                        continue
                    if (reply.startswith("Wrote") or reply.startswith("Updated")) and \
                            check_tool_args(name, args, schemas).error is not None:
                        false_success.append(
                            f"{fname} :: {tid} t{k} (model {model}) {name} reply={reply[:20]!r} "
                            f"got keys={sorted(args)}")
                arg_err_by_model[model] = arg_err_by_model.get(model, 0) + len(r.get("arg_errors", []))
                arg_extra_by_model[model] = arg_extra_by_model.get(model, 0) + len(r.get("arg_extra", []))
    return dict(files=files, tries=tries, done_fail=done_fail, false_success=false_success,
                arg_err=arg_err_by_model, arg_extra=arg_extra_by_model)


DEFAULT_GLOB = "results/stage1/*.json"


def expand(args: list[str]) -> tuple[list[str], list[tuple[str, int]]]:
    """Each argument is a path or a glob -> (unique files, [(argument, files it matched)]).

    A plain path that exists counts as itself even when its name holds glob characters ("[1].json").
    Directories are not files: an argument that names only a folder matches 0."""
    seen: dict[str, str] = {}
    per_arg: list[tuple[str, int]] = []
    for a in args:
        hits = [h for h in glob.glob(a) if os.path.isfile(h)]
        if not hits and os.path.isfile(a):
            hits = [a]
        per_arg.append((a, len(hits)))
        for h in hits:
            seen.setdefault(os.path.normcase(os.path.realpath(h)), h)
    return list(seen.values()), per_arg


def main(argv: list[str]) -> int:
    args = list(argv) or [DEFAULT_GLOB]
    paths, per_arg = expand(args)
    given = f"{len(args)} argument{'s' if len(args) != 1 else ''}" if argv else "no arguments, default"
    print(f"suspicious_tries: {given}")
    for a, n in per_arg:
        print(f"    {n:>4} file{'s' if n != 1 else ' '}  {a!r}")
    for a, n in per_arg:
        if n == 0:
            print(f"WARNING: {a!r} matched no file")
    s = scan(paths)
    print(f"suspicious_tries: scanned {s['files']} files, {s['tries']} tries "
          f"({len(paths)} unique files matched, {len(paths) - s['files']} skipped: "
          f"not result-shaped or unreadable)\n")
    if s["files"] == 0 or s["tries"] == 0:
        if s["files"] == 0:
            print("no result-shaped file among the matches: nothing was checked (this is not a pass)")
        else:
            print(f"{s['files']} result files found, but they held no tries: nothing was checked "
                  f"(this is not a pass)")
        print("\nVERDICT: NOTHING SCANNED")
        return 2
    print(f"(b) FALSE SUCCESS (Wrote/Updated on a call the arg gate rejects): {len(s['false_success'])}")
    for line in s["false_success"]:
        print(f"    {line}")
    print(f"\n(a) failed try but model ended 'done': {len(s['done_fail'])}")
    for line in s["done_fail"][:50]:
        print(f"    {line}")
    if len(s["done_fail"]) > 50:
        print(f"    ... (+{len(s['done_fail']) - 50} more)")
    print("\n(c) arg_errors / arg_extra per model (S8f.1a+ runs only):")
    for m in sorted(set(s["arg_err"]) | set(s["arg_extra"])):
        print(f"    {m}: arg_errors={s['arg_err'].get(m, 0)} arg_extra={s['arg_extra'].get(m, 0)}")
    print(f"\nVERDICT: {'FALSE SUCCESS FOUND' if s['false_success'] else 'no false success'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
