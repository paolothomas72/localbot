"""C6 part C, C7 part C: find the limits step by step. Runs the REAL ``localbot chat`` (real Ollama, real container,
product rules) with the reply limit, the context, the steps and the thinking setting raised one step at a time, and
writes what it saw.

  python scripts/chat_ladder.py --model "gemma4:12b"
  python scripts/chat_ladder.py --model M [--task "clean up this folder"] [--tries 3] [--time-limit 600] [--out-dir DIR]

The ladder (stop climbing a ladder at the first level that does not fit on the card, whose tries all time out, or
that is "slow: maybe off the card"; that level is still written down, and is never the best one):
  1. reply limit 512, 1024, 2048, 4096 at context 8192 and 12 steps (thinking: the model's default);
  2. context 16384, 32768, 65536 at the best reply limit from 1 (best = most tries that WORKED, then the fastest);
  3. steps 16, 20, 25 at the best reply limit and context from 1 and 2 (its 12-step level is the first rung), climbing
     to the next only while at least one try of the level ended ``max_steps`` (more steps cannot help otherwise);
  4. the best level from 1, 2 and 3 once more with ``--think off``.
Every try passes ``--steps`` explicitly. "worked" = the message ended ``done`` with a written answer that is not a tool
call written as text (a proxy: it cannot judge whether the answer was good). "slow: maybe off the card" = Ollama's
``/api/ps`` reading is doubtful and the level's median speed is below 70 % of level 1's.

Every try uses the bot ``ladder`` (never ``main``). Before the chat starts its work folder is emptied and filled with
a fresh copy of the same test pack (``TEST_PACK``, written from this one place). Every question about deleting or
moving files is answered ``n``, so nothing is ever lost. A try that runs past the time limit is stopped and said so.
Each try is its own ``python scripts/chat_ladder.py --one-try ...`` process that runs ``localbot.chat.chat_main`` with a
scripted keyboard; the numbers come from the session record that chat writes (``~/.localbot/bots/ladder/sessions``),
never from the screen.

Output (``LOCALBOT_RUNS_DIR`` or ``<repo>/state/runs``): ``ladder_<model>_<time>.csv`` (one row per try) and ``.md``
(a plain-words table). The fake key in the pack's ``.env`` is looked for in every session record and in both files.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from localbot import chat  # noqa: E402

BOT = "ladder"
FAKE_KEY = "ladder-fake-key-5530-not-a-real-key"
TEST_PACK = {  # the same files, byte for byte, in every try; written only by ``write_pack``
    "notes.txt": "Meeting notes\n- send the report on Friday\n- ask Sam about the logo\n",
    "prices.py": "PRICES = {'coffee': 3.5, 'tea': 2.75, 'cake': 4.25}\n\n\ndef total(items):\n"
                 "    return sum(PRICES[i] for i in items)\n",
    "todo.md": "# To do\n- [x] buy milk\n- [ ] call the plumber\n- [ ] tidy the garage\n",
    "old_draft.txt": "Draft 1 of the newsletter. Not used any more. Safe to throw away.\n",
    ".env": f"API_KEY={FAKE_KEY}\n",
}
LADDER_REPLY_LIMITS = (512, 1024, 2048, 4096)
LADDER_CONTEXTS = (16384, 32768, 65536)
LADDER_STEPS = (12, 16, 20, 25)  # C7: the first is the chat's default and the step every other climb runs at
BASE_CONTEXT = 8192
BASE_STEPS = LADDER_STEPS[0]
LEVELS_MAX = len(LADDER_REPLY_LIMITS) + len(LADDER_CONTEXTS) + len(LADDER_STEPS) - 1 + 1
SLOW_SHARE = 0.7
SLOW_STOP = "slow: maybe off the card"
LOAD_FAILED = "load_failed"  # C17: stop reason when every try at a step never ran (provider HTTP 500)
LOAD_ERROR_CHARS = 160
COLUMNS = ["level", "try", "reply_limit", "context", "think", "steps_limit", "steps_used", "replies_cut",
           "cut_by_thinking", "tools_run", "ended_why", "answered", "worked", "text_tool_calls", "asked_about_files",
           "loops", "seconds", "tokens_per_second", "model_gb", "on_card_gb", "fits", "doubtful", "card_used_gb",
           "card_total_gb", "reached", "key_in_record", "error"]
MEANINGS = [
    ("level", "which step of the ladder (1 is the first)"),
    ("try", "which of the tries at that level"),
    ("reply_limit", "most tokens the model could write in one step, thinking included"),
    ("context", "tokens the model could hold for the whole chat"),
    ("think", "default (the model's own), or off (thinking switched off)"),
    ("steps_limit", "most steps the bot could take for the one message"),
    ("steps_used", "model replies it actually took"),
    ("replies_cut", "replies that stopped at the reply limit"),
    ("cut_by_thinking", "of those, replies with nothing visible where thinking used the whole limit"),
    ("tools_run", "tool calls that ran"),
    ("ended_why", "how the message ended (done, max_steps, timeout, ...)"),
    ("answered", "yes when the bot gave a written answer (any answer, even a tool call written as text)"),
    ("worked", "True when the message ended done with a written answer that is not a tool call written as text "
               "(a proxy: it cannot tell whether the answer was good)"),
    ("text_tool_calls", "answers that were a tool call written as text instead of a real tool call (JSON with "
                        "action/action_input or name/arguments, or a <tool_call> tag); nothing in them was run"),
    ("asked_about_files", "times the chat asked about deleting or changing a file that was there before (answered n)"),
    ("loops", "replies that repeated the same lines over and over and were stopped"),
    ("seconds", "time the message took (the time limit when it timed out)"),
    ("tokens_per_second", "model speed, from Ollama's own numbers"),
    ("model_gb", "size of the loaded model as Ollama reports it"),
    ("on_card_gb", "how much of it is on the graphics card"),
    ("fits", "yes when all of it is on the card, no when part moved to normal memory, unknown when not measured"),
    ("doubtful", "True when Ollama's /api/ps reports far less than the model's size on disk; it has under-reported "
                 "before, so a \"yes\" with doubtful True is not proof"),
    ("card_used_gb", "the card's used memory after the first reply, from nvidia-smi (empty without nvidia-smi)"),
    ("card_total_gb", "the card's total memory, from nvidia-smi (empty without nvidia-smi)"),
    ("reached", "WEAK: True when the bot ran a tool or gave an answer; it was True in all 48 tries of the first two "
                "real runs, so it is kept only for comparison and is not used to pick the best level"),
    ("key_in_record", "True when the fake key was found in the session record (it must always be False)"),
    ("error", "the first 160 characters of the error the try ended with (empty when it ended without one); "
              "an HTTP 500 here means the model did not load"),
]


# --- the test pack ---------------------------------------------------------------------------------------------------


def write_pack(folder: Path) -> Path:
    for rel, text in TEST_PACK.items():
        p = Path(folder) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
    return Path(folder)


def reset_work(work: Path) -> None:
    """Empty the ladder bot's work folder and fill it with a fresh pack. Refuses any other folder: the only place this
    script ever deletes is ``<something>/bots/ladder/work``."""
    work = Path(work)
    if not (work.name == "work" and work.parent.name == BOT and work.parent.parent.name == "bots"):
        raise ValueError(f"refusing to empty {work}: not a 'bots/{BOT}/work' folder")
    for child in list(work.iterdir()):
        if child.is_symlink() or child.is_file():
            child.unlink()
        else:
            shutil.rmtree(child)
    write_pack(work)


# --- one try: the real chat, a scripted keyboard ---------------------------------------------------------------------


class Driver:
    """The keyboard of one try: the task once, ``n`` to every question about files, then ``/quit``."""

    def __init__(self, task: str) -> None:
        self.task, self.asked, self.sent_task = task, 0, False

    def __call__(self, prompt: str) -> "str | None":
        if prompt in (chat.PROMPT_ASK, chat.PROMPT_ASK_MOVE, chat.PROMPT_ASK_PERM):  # C10b: the permissions question too
            self.asked += 1
            return "n"
        if prompt == "you> ":
            if not self.sent_task:
                self.sent_task = True
                return self.task
            return "/quit"
        return None  # a model pick or anything else: nobody answers


# --- answers that are a tool call written as text (C7) -----------------------------------------------------------

_TOOL_TAG = re.compile(r"<\s*/?\s*tool_call\b", re.IGNORECASE)


def _json_values(text: str):
    """Every JSON object or list that starts at a ``{`` or ``[`` of ``text`` and parses (a code fence around it is fine)."""
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                yield dec.raw_decode(text, i)[0]
            except ValueError:
                continue


def _call_shaped(obj: Any, depth: int = 0) -> bool:
    if isinstance(obj, dict):
        keys = set(obj)
        if {"action", "action_input"} <= keys or {"name", "arguments"} <= keys:
            return True
        return depth < 3 and any(_call_shaped(v, depth + 1) for v in obj.values())
    if isinstance(obj, list):
        return depth < 3 and any(_call_shaped(v, depth + 1) for v in obj)
    return False


def looks_like_text_tool_call(text: Any) -> bool:
    """True when an answer looks like a tool call written as text: JSON with ``action`` + ``action_input``, or
    ``name`` + ``arguments`` (also inside a list or a ``function`` object), or a ``<tool_call>`` tag."""
    t = str(text or "")
    if not t.strip():
        return False
    if _TOOL_TAG.search(t):
        return True
    return any(_call_shaped(v) for v in _json_values(t))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="chat_ladder.py", description="Raise the chat's limits step by step and measure.")
    ap.add_argument("--model", required=True, help="an installed Ollama model, e.g. gemma4:12b")
    ap.add_argument("--task", default="clean up this folder", help="the message sent in every try")
    ap.add_argument("--tries", type=int, default=3, help="tries per level (default 3)")
    ap.add_argument("--time-limit", type=float, default=600.0, help="seconds one try may take (default 600)")
    ap.add_argument("--out-dir", default=None, help="where the .csv and .md go (default <repo>/state/runs)")
    ap.add_argument("--one-try", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--reply-limit", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--context", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--think", choices=("on", "off"), default=None, help=argparse.SUPPRESS)
    ap.add_argument("--steps", type=int, default=None, help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def one_try(args: argparse.Namespace, deps: Any = None) -> int:
    """The child process: one chat, one message, then quit. ``deps`` is for tests."""
    import dataclasses
    base = deps or chat.default_deps()
    inner_new_work = base.new_work

    def new_work(path: Path) -> Path:
        made = inner_new_work(path)
        write_pack(made)  # a folder the chat has to make is filled with the same pack
        return made
    deps = dataclasses.replace(base, read_line=Driver(args.task), new_work=new_work)
    think = None if args.think is None else args.think == "on"
    return chat.chat_main(BOT, args.model, deps, reply_limit=args.reply_limit, context=args.context,
                          steps=args.steps, think=think)


def child_command(model: str, task: str, reply_limit: int, context: int, think: str,
                  steps: int = BASE_STEPS) -> list[str]:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--one-try", "--model", model, "--task", task,
           "--reply-limit", str(reply_limit), "--context", str(context)]
    cmd += ["--steps", str(steps)]  # C7: always explicit, so a try never depends on the chat's default
    if think == "off":
        cmd += ["--think", "off"]
    return cmd


def run_child(cmd: list[str], limit_s: float) -> tuple["int | None", bool]:
    """Run ``cmd``; past ``limit_s`` stop it (and what it started) and say so. -> (exit code or None, timed out)."""
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        return proc.wait(timeout=limit_s), False
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
        proc.kill()
        proc.wait()
        return None, True


# --- the numbers: from the session record only ----------------------------------------------------------------------


def _gb(n: Any) -> "float | None":
    return round(n / 1e9, 2) if isinstance(n, (int, float)) and not isinstance(n, bool) and n else None


def _gb_mib(n: Any) -> "float | None":
    return round(n * 1048576 / 1e9, 2) if isinstance(n, (int, float)) and not isinstance(n, bool) and n >= 0 else None


def clean_error(err: Any) -> str:
    """C17b: the error text a ladder row keeps: the two boilerplate pieces removed, line breaks and runs of spaces
    collapsed, then cut to LOAD_ERROR_CHARS. ``HTTP Error 500`` stays in it, so ``is_load_failure`` still works."""
    text = str(err or "").replace("request failed: ", "").replace("Internal Server Error: ", "")
    return " ".join(text.split())[:LOAD_ERROR_CHARS]


def parse_record(path: Path, timed_out_after: "float | None" = None) -> dict:
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    recs = []
    for line in raw.splitlines():
        try:
            obj = json.loads(line)
        except ValueError:
            continue  # the last line of a record that was cut off while it was written
        if isinstance(obj, dict):
            recs.append(obj)
    head = next((r for r in recs if r.get("kind") == "header"), {})
    gpu = next((r for r in recs if r.get("kind") == "gpu_fit"), None)
    summary = next((r.get("summary") or {} for r in recs if r.get("kind") == "summary"), None)
    events = [r.get("event") or {} for r in recs if r.get("kind") == "event"]
    cuts = [e for e in events if e.get("event") == "cut"]
    ran = [e for e in events if e.get("event") == "tool" and e.get("status") == "ran"]
    answered = any(e.get("event") == "answer" and str(e.get("text") or "").strip() for e in events)
    ends = [e for e in events if e.get("event") == "message_end"]
    if timed_out_after is not None:
        why, seconds = f"timeout ({timed_out_after:g} s limit)", timed_out_after
    elif ends:
        why = str(ends[-1].get("end_reason"))
        seconds = (summary or {}).get("elapsed_s") or round(sum(float(e.get("wall_s") or 0) for e in ends), 3)
    elif summary:
        why, seconds = str(summary.get("reason")), summary.get("elapsed_s")
    else:
        why, seconds = "no summary (the record stopped early)", None
    if summary and isinstance(summary.get("provider"), list):
        used = len(summary["provider"])
    elif ends and isinstance(ends[-1].get("steps"), int):
        used = ends[-1]["steps"]
    else:
        used = len(cuts) + len(ran) + (1 if answered else 0)
    answers = [str(e.get("text") or "") for e in events if e.get("event") == "answer"]
    final = answers[-1] if answers else ""
    worked = timed_out_after is None and why == "done" and bool(final.strip()) and not looks_like_text_tool_call(final)
    row = {
        "reply_limit": head.get("reply_limit"), "context": head.get("context"), "think": head.get("think"),
        "steps_limit": head.get("steps"), "steps_used": used, "replies_cut": len(cuts),
        "cut_by_thinking": sum(1 for e in cuts if not e.get("visible") and e.get("thinking_tokens")),
        "tools_run": len(ran), "ended_why": why, "answered": "yes" if answered else "no", "seconds": seconds,
        "worked": worked, "text_tool_calls": sum(1 for a in answers if looks_like_text_tool_call(a)),
        "asked_about_files": sum(1 for e in events if e.get("event") == "ask"),
        "loops": sum(1 for e in events if e.get("event") == "loop"),
        "tokens_per_second": (summary or {}).get("tok_s"),
        "model_gb": _gb((gpu or {}).get("model_bytes")), "on_card_gb": _gb((gpu or {}).get("on_card_bytes")),
        "fits": (gpu or {}).get("fits") or "unknown", "doubtful": bool((gpu or {}).get("doubtful")),
        "card_used_gb": _gb_mib((gpu or {}).get("card_used_mib")),
        "card_total_gb": _gb_mib((gpu or {}).get("card_total_mib")),
        "key_in_record": FAKE_KEY in raw,
        "error": clean_error((summary or {}).get("error")),  # C17: why the try ended in an error
    }
    row["reached"] = bool(row["tools_run"] or answered)
    return row


def no_record_row(reply_limit: int, context: int, think: str, rc: "int | None", seconds: float,
                  steps: "int | None" = None) -> dict:
    return {"reply_limit": reply_limit, "context": context, "think": think, "steps_limit": steps, "steps_used": 0,
            "replies_cut": 0, "cut_by_thinking": 0, "tools_run": 0, "ended_why": f"no record (exit code {rc})",
            "answered": "no", "worked": False, "text_tool_calls": 0, "asked_about_files": 0, "loops": 0,
            "seconds": round(seconds, 1), "tokens_per_second": None, "model_gb": None, "on_card_gb": None,
            "fits": "unknown", "doubtful": False, "card_used_gb": None, "card_total_gb": None, "reached": False,
            "key_in_record": False, "error": ""}


def make_process_run_try(model: str, task: str, limit_s: float, home: "Path | None" = None) -> Callable[..., dict]:
    bot_dir = (home or Path.home()) / ".localbot" / "bots" / BOT

    def run_try(reply_limit: int, context: int, think: str, n: int, steps: int = BASE_STEPS) -> dict:
        work, sessions = bot_dir / "work", bot_dir / "sessions"
        if work.exists():
            reset_work(work)  # a folder that does not exist yet is made and filled by the chat's own start
        before = set(sessions.glob("*.jsonl")) if sessions.exists() else set()
        t0 = time.monotonic()
        rc, timed_out = run_child(child_command(model, task, reply_limit, context, think, steps), limit_s)
        secs = time.monotonic() - t0
        new = sorted(set(sessions.glob("*.jsonl")) - before) if sessions.exists() else []
        if not new:
            return no_record_row(reply_limit, context, think, rc, secs, steps)
        row = parse_record(new[-1], timed_out_after=limit_s if timed_out else None)
        row.update(reply_limit=reply_limit, context=context, think=row["think"] or think)
        return row
    return run_try


# --- the climb ---------------------------------------------------------------------------------------------------------


def _timed_out(r: dict) -> bool:
    return str(r.get("ended_why", "")).startswith("timeout")


def is_load_failure(r: dict) -> bool:
    """C17: the try never ran: the provider failed with HTTP 500 (the runner's own ``is_infra`` test)."""
    from localbot.exam.runner import is_infra
    return is_infra(str(r.get("ended_why") or ""), str(r.get("error") or ""))


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _level_summary(idx: int, rp: int, ctx: int, think: str, rows: list[dict], steps: int = BASE_STEPS,
                   ref_tps: "float | None" = None) -> dict:
    secs = [r["seconds"] for r in rows if _num(r.get("seconds"))]
    tps = [r["tokens_per_second"] for r in rows if _num(r.get("tokens_per_second"))]
    median_tps = statistics.median(tps) if tps else None
    doubtful = any(r.get("doubtful") for r in rows)
    stop, load_error = "", ""
    if rows and all(is_load_failure(r) for r in rows):
        stop = LOAD_FAILED  # C17: nothing at this step loaded; higher steps would fail the same way
        load_error = str(rows[0].get("error") or "")[:LOAD_ERROR_CHARS]
    elif any(r.get("fits") == "no" for r in rows):
        stop = "part of the model did not fit on the card"
    elif rows and all(_timed_out(r) for r in rows):
        stop = "every try timed out"
    elif rows and all(str(r.get("ended_why", "")).startswith("no record") for r in rows):
        stop = "no try produced a session record"
    elif doubtful and ref_tps and median_tps is not None and median_tps < SLOW_SHARE * ref_tps:
        stop = SLOW_STOP  # C7: /api/ps cannot be trusted here, and the speed says part of the model left the card
    return {"index": idx, "reply_limit": rp, "context": ctx, "think": think, "steps": steps, "rows": rows,
            "tries": len(rows), "worked": sum(1 for r in rows if r.get("worked")),
            "reached": sum(1 for r in rows if r.get("reached")),
            "max_steps": sum(1 for r in rows if r.get("ended_why") == "max_steps"),
            "text_tool_calls": sum(int(r.get("text_tool_calls") or 0) for r in rows),
            "loops": sum(int(r.get("loops") or 0) for r in rows),
            "median_s": statistics.median(secs) if secs else None, "median_tps": median_tps,
            "doubtful": doubtful, "stop": stop, "load_error": load_error}


def _best(levels: list[dict]) -> dict:
    # Reviewer (C6): a level that stopped the climb (did not fit on the card, or every try timed out) is never the
    # best one; only when every level stopped is it chosen from all of them.
    # C7: best = most tries that WORKED (``reached`` was yes in every real try: it tells nothing), then the fastest.
    usable = [lv for lv in levels if not lv["stop"]] or levels
    return min(usable, key=lambda lv: (-lv["worked"], lv["median_s"] if lv["median_s"] is not None else float("inf")))


def run_ladder(tries: int, run_try: Callable[..., dict], say: Callable[[str], Any] = print) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    levels: list[dict] = []

    def level(rp: int, ctx: int, think: str, steps: int = BASE_STEPS) -> dict:
        idx = len(levels) + 1
        got: list[dict] = []
        for n in range(1, tries + 1):
            say(f"try {n} of {tries}: level {idx}, reply limit {rp}, context {ctx}, steps {steps}, thinking {think} ...")
            row = dict(run_try(rp, ctx, think, n, steps))  # C7b (reviewer): every try is given its steps
            row["level"], row["try"] = idx, n
            row.setdefault("reached", bool(row.get("tools_run") or row.get("answered") == "yes"))
            row.setdefault("worked", row.get("ended_why") == "done" and row.get("answered") == "yes"
                           and not row.get("text_tool_calls"))
            got.append(row)
            say(f"  -> {row.get('ended_why')}, {row.get('seconds')} s, worked: {'yes' if row['worked'] else 'no'}, "
                f"fits on the card: {row.get('fits')}")
        rows.extend(got)
        lv = _level_summary(idx, rp, ctx, think, got, steps, levels[0]["median_tps"] if levels else None)
        levels.append(lv)
        if lv["stop"]:
            say(f"stopping this climb: {lv['stop']}")
        return lv

    phase1: list[dict] = []
    for rp in LADDER_REPLY_LIMITS:
        phase1.append(level(rp, BASE_CONTEXT, "default"))
        if phase1[-1]["stop"]:
            break
    if phase1[-1]["stop"] == LOAD_FAILED:
        return rows, levels  # C17: nothing loaded at this step: no higher step is run
    best_rp = _best(phase1)["reply_limit"]
    phase2: list[dict] = []
    for ctx in LADDER_CONTEXTS:
        phase2.append(level(best_rp, ctx, "default"))
        if phase2[-1]["stop"]:
            break
    top = _best(phase1 + phase2)
    # C7: the steps climb at the best reply limit and context; ``top`` (run at 12 steps) is its first rung
    climb = [top]
    for steps in LADDER_STEPS[1:]:
        if climb[-1]["stop"] or not climb[-1]["max_steps"]:
            break  # no try ran out of steps: more steps cannot help
        climb.append(level(top["reply_limit"], top["context"], "default", steps))
    best = _best(phase1 + phase2 + climb[1:])
    level(best["reply_limit"], best["context"], "off", best["steps"])
    return rows, levels


def _try_s(steps: int, rp: int, limit_s: float) -> float:
    return min(limit_s, steps * (rp / 64 + 3) + 20)


def estimate_text(tries: int, limit_s: float) -> str:
    worst_h = LEVELS_MAX * tries * limit_s / 3600
    top = LADDER_REPLY_LIMITS[-1]
    rough = sum(_try_s(BASE_STEPS, rp, limit_s) * tries for rp in LADDER_REPLY_LIMITS)
    rough += sum(_try_s(BASE_STEPS, top, limit_s) * tries for _ in (*LADDER_CONTEXTS, "off"))
    rough += sum(_try_s(s, top, limit_s) * tries for s in LADDER_STEPS[1:])
    base = LEVELS_MAX - (len(LADDER_STEPS) - 1)
    return (f"The default ladder has {base} levels, {base * tries} tries, plus up to {len(LADDER_STEPS) - 1} more levels "
            f"for the steps climb: at most {LEVELS_MAX} levels, {LEVELS_MAX * tries} tries. One try takes at most "
            f"{limit_s / 60:g} min (the time limit), so the worst case is {worst_h:.1f} hours. If every step used the whole "
            f"reply limit at about 64 tokens a second it would be about {rough / 3600:.1f} hours; in practice many tries end "
            f"sooner, the steps climb (16, 20, 25 steps) only runs while a try ran out of steps, and the ladder stops early "
            f"at a level that does not fit on the card, is slow and maybe off the card, or where every try timed out.")


# --- the files ---------------------------------------------------------------------------------------------------------


def _fmt(v: Any) -> str:
    return "" if v is None else str(v)


def _csv_text(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({c: _fmt(r.get(c)) for c in COLUMNS})
    return buf.getvalue()


def _md_cell(text: Any) -> str:
    """C17b: text for a markdown table cell: line breaks become a space, | is escaped. The CSV keeps the exact text."""
    return " ".join(str(text if text is not None else "").split()).replace("|", "\\|")


def _md_text(rows: list[dict], levels: list[dict], model: str, stamp: str) -> str:
    leaks = sum(1 for r in rows if r.get("key_in_record"))
    out = [f"# Chat ladder: {model} ({stamp})", "",
           "Each row below is one real try of `localbot chat` (bot `ladder`, product rules, real container) on the "
           "same test pack. Every number comes from the session record.", ""]
    if leaks:
        out += [f"**KEY FOUND IN A SESSION RECORD ({leaks} tries). Stop and tell the reviewer: the masking failed.**", ""]
    else:
        out += ["The fake key from the test pack's `.env` was looked for in every session record: not found. It is not in "
                "this table either.", ""]
    out += ["`worked` is a proxy: it means the message ended `done` with a written answer that is not a tool call "
            "written as text. It cannot tell whether the answer was good (whether the folder was really tidied, or what the "
            "bot said is true); read the answers in the session records for that. It does show the tries that ran out of "
            "steps, looped, timed out or \"answered\" with a tool call written as text.", "",
            "`reached` is weak: it was True in all 48 tries of the first two real runs. It is kept in the CSV for comparison "
            "only and is not used to pick the best level (best = most tries that worked, then the fastest).", "",
            "A \"yes\" under `fits` is only as good as Ollama's `/api/ps`, which has under-reported before. When `doubtful` "
            "is True the reading was far below the model's size on disk: do not trust that \"yes\", look at the speed. "
            f"When `doubtful` is True and a level's median speed is below {SLOW_SHARE:.0%} of level 1's, the level is marked "
            f"`{SLOW_STOP}` and that climb stops. `card_used_gb` / `card_total_gb` are nvidia-smi's own numbers (whole "
            "card, all programs), read after the first reply.", "",
            "## What each column means", ""]
    out += [f"- `{c}`: {m}" for c, m in MEANINGS]
    out += ["", "## Per level", "",
            "| level | reply limit | context | steps | thinking | tries that worked | tries that reached a tool call or "
            "answer (weak) | ran out of steps | text tool calls | loops | median seconds | median tokens/s | doubtful "
            "| climb stopped |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for lv in levels:
        out.append(f"| {lv['index']} | {lv['reply_limit']} | {lv['context']} | {lv.get('steps', '')} | {lv['think']} "
                   f"| {lv['worked']} of {lv['tries']} | {lv['reached']} of {lv['tries']} | {lv['max_steps']} "
                   f"| {lv['text_tool_calls']} | {lv['loops']} | {_fmt(lv['median_s'])} | {_fmt(lv['median_tps'])} "
                   f"| {lv['doubtful']} | {lv['stop'] or ''}"
                   f"{(': ' + _md_cell(lv['load_error'])) if lv.get('load_error') else ''} |")
    out += ["", "## Every try", "", "| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for r in rows:
        out.append("| " + " | ".join(_md_cell(_fmt(r.get(c))) if c == "error" else _fmt(r.get(c)) for c in COLUMNS) + " |")
    return "\n".join(out) + "\n"


def write_outputs(rows: list[dict], levels: list[dict], out_dir: Path, model: str, stamp: str) -> tuple[Path, Path]:
    csv_text, md_text = _csv_text(rows), _md_text(rows, levels, model, stamp)
    if FAKE_KEY in csv_text or FAKE_KEY in md_text:
        raise RuntimeError("the fake key would be written to an output file; nothing was written")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"ladder_{re.sub(r'[^A-Za-z0-9._-]', '_', model)}_{stamp}"
    csv_path, md_path = out_dir / f"{base}.csv", out_dir / f"{base}.md"
    csv_path.write_text(csv_text, encoding="utf-8", newline="")
    md_path.write_text(md_text, encoding="utf-8", newline="\n")
    return csv_path, md_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.one_try:
        return one_try(args)
    if args.tries < 1 or args.time_limit <= 0:
        print("--tries must be 1 or more and --time-limit above 0.")
        return 2
    print(f"Chat ladder for {args.model}. Bot '{BOT}' (not 'main'); its work folder is reset before every try; every "
          "question about files is answered n.")
    print(estimate_text(args.tries, args.time_limit))
    run_try = make_process_run_try(args.model, args.task, args.time_limit)
    rows, levels = run_ladder(args.tries, run_try, say=lambda t: print(t, flush=True))
    out_dir = Path(args.out_dir or os.environ.get("LOCALBOT_RUNS_DIR") or REPO / "state" / "runs")
    csv_path, md_path = write_outputs(rows, levels, out_dir, args.model, datetime.now().strftime("%Y%m%dT%H%M%S"))
    print(f"Done. Table: {md_path}\nCSV: {csv_path}")
    return 1 if any(r.get("key_in_record") for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())