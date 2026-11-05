#!/usr/bin/env python3
"""Product-path 12-task exam: LH names + caller guard. Not Door A.

Default models: daily qwen-ft, then stock qwen3:8b. No roster sweep.
Does not overwrite all_fit_latest.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
    sys.path.insert(0, str(_ROOT / "lib"))

from fit_roster import DEFAULT_CTX
from graded_tasks import TASKS
from capture import route_label, write_fail_capture
from harness_guards import ProductGuard
from ollama_chat import chat, unload
from paths import RAW
from run_all_fit import TOOLS, run_tool
from sandbox_bash import sandbox_bash

DEADLINE_TRIAL_S = 900
TASK_TIMEOUT_S = 180
MAX_TURNS = 8
FOCUS = [
    "qwen3-8b-ft-agentic-v1:latest",
    "qwen3:8b",
]
SYSTEM_SLIM = (
    "You are a local coding agent. Use tools. Be honest. "
    "Strip line-number prefixes from read() before edit or write. "
    "Do the smallest change that matches intent. "
    "If asked for JSON only, reply with JSON only."
)
SYSTEM_FULL = (
    "You are a local coding agent. Use tools. Be honest. "
    "Never invent tool results, file contents, or exit codes. "
    "Strip line-number prefixes from read() before edit or write. "
    "Prefer edit for small changes. After a successful append, do not overwrite the file back. "
    "If bash_exec fails, use glob/read/write immediately and say so. Keep answers short. "
    "After compaction, read the target file again before any write. If TASK_MANIFEST.json "
    "exists, trust its goal, paths, and last_verified_bytes over older chat. "
    "If the ticket is ambiguous, multi-file beyond the ticket, or you cannot verify, "
    "write DEFER.json with a short reason and stop. Do not guess. "
    "The first grep hit is often a definition, not a caller. Do not write that filename "
    "until you have read a later hit."
)
SYSTEM = SYSTEM_SLIM if "--slim" in sys.argv else SYSTEM_FULL


def exec_tool(name: str, args: dict, files: dict[str, str], guard: ProductGuard) -> str:
    if name in {"write", "write_file"} and "content" in args:
        args = dict(args)
        args["content"] = guard.prepare_write(str(args.get("content") or ""))
    if name == "edit" and "new_string" in args:
        args = dict(args)
        args["new_string"] = guard.prepare_write(str(args.get("new_string") or ""))
    why = guard.check(name, args, files)
    if name in {"grep", "search_files"}:
        if why:
            return f"ERROR: {why}"
        out = run_tool(name, args, files)
        guard.check(name, args, files, out)
        return out
    if why:
        return f"ERROR: {why}"
    if name == "edit":
        return run_tool(name, args, files)
    if name in {"read", "read_file"}:
        path = str(args.get("path") or "")
        if guard.repeat_read(path):
            return guard.already_read_msg(path)
        return run_tool(name, args, files)
    if name in {"write", "write_file"}:
        return run_tool(name, args, files)
    if name == "bash_exec":
        handled, text = sandbox_bash(str(args.get("command") or ""), files)
        if handled:
            return text
        stub = guard.bash_again()
        if stub:
            return stub
        return "ERROR: bash not found. Use glob to list files, then write."
    return run_tool(name, args, files)


EXTRA_TOOL = {
    "type": "function",
    "function": {
        "name": "todo",
        "description": "Scratch checklist. Do not call unless the user asks for a todo list.",
        "parameters": {"type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]},
    },
}


def loop(
    model: str,
    user: str,
    judge,
    seed: dict,
    num_ctx: int,
    tools: list | None = None,
    meta: dict | None = None,
) -> dict:
    files = dict(seed)
    guard = ProductGuard()
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    elapsed = 0.0
    eval_count = 0
    tools_used: list[str] = []
    vetoes = 0
    last_text = ""
    for _ in range(MAX_TURNS):
        data = None
        dt = 0.0
        last_exc = None
        for attempt in range(2):
            try:
                data, dt = chat(model, messages, tools=tools or TOOLS, num_ctx=num_ctx, timeout=TASK_TIMEOUT_S)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if "500" not in str(exc):
                    break
        if data is None:
            return {
                "pass": False,
                "reason": f"request failed: {last_exc}",
                "elapsed_s": round(elapsed, 2),
                "tools": tools_used,
                "vetoes": vetoes,
                "tok_s": 0,
            }
        elapsed += dt
        eval_count += data.get("eval_count") or 0
        msg = data.get("message") or {}
        content = msg.get("content") or ""
        scrubbed = guard.scrub_text(content, files)
        if scrubbed != content:
            msg = dict(msg)
            msg["content"] = scrubbed
            vetoes += 1
        messages.append(msg)
        last_text = (msg.get("content") or "")[:300]
        calls = msg.get("tool_calls") or []
        if not calls:
            break
        for call in calls:
            fn = call.get("function") or {}
            name = fn.get("name") or "unknown"
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tools_used.append(name)
            result = exec_tool(name, args, files, guard)
            if result.startswith("ERROR: harness:"):
                vetoes += 1
            messages.append({"role": "tool", "content": result, "name": name})
    ok, reason = judge(files, tools_used, last_text)
    info = meta or {}
    task_id = str(info.get("id") or "task")
    kind = info.get("kind")
    out = {
        "pass": ok,
        "reason": reason,
        "elapsed_s": round(elapsed, 2),
        "tok_s": round(eval_count / elapsed, 1) if elapsed else 0.0,
        "tools": tools_used,
        "vetoes": vetoes,
        "text": last_text,
        "num_ctx": num_ctx,
        "path": "product_guard",
        "route": route_label(kind, user),
    }
    if not ok:
        cap = write_fail_capture(
            model=model,
            task_id=task_id,
            prompt=user,
            kind=kind,
            seed=seed,
            files=files,
            tools=tools_used,
            reason=reason,
            text=last_text,
            vetoes=vetoes,
        )
        out["capture"] = str(cap)
    return out


def majority(passes: list[bool]) -> bool:
    return sum(1 for p in passes if p) >= (len(passes) // 2 + 1)


def run_trial(model: str, tools: list | None = None) -> dict[str, dict]:
    t0 = time.perf_counter()
    tasks: dict[str, dict] = {}
    for spec in TASKS:
        if time.perf_counter() - t0 > DEADLINE_TRIAL_S:
            tasks[spec["id"]] = {
                "pass": False,
                "reason": "TIMEOUT",
                "level": spec["level"],
                "kind": spec["kind"],
            }
            print(json.dumps({"model": model, "id": spec["id"], "pass": False, "reason": "TIMEOUT"}), flush=True)
            continue
        result = loop(
            model,
            spec["prompt"],
            spec["judge"],
            spec["seed"],
            DEFAULT_CTX,
            tools=tools,
            meta={"id": spec["id"], "kind": spec["kind"]},
        )
        result["level"] = spec["level"]
        result["kind"] = spec["kind"]
        tasks[spec["id"]] = result
        print(
            json.dumps(
                {
                    "model": model,
                    "id": spec["id"],
                    "pass": result["pass"],
                    "reason": result["reason"],
                    "tools": result.get("tools"),
                    "vetoes": result.get("vetoes"),
                    "elapsed_s": result.get("elapsed_s"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return tasks


def main() -> int:
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    models = [a for a in sys.argv[1:] if not a.startswith("--")] or FOCUS
    trials = 3 if "--n3" in flags else 1
    tools = list(TOOLS)
    if "--extra-tool" in flags:
        tools = list(TOOLS) + [EXTRA_TOOL]
    dest_dir = RAW / "graded"
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "tag": "lh_product_n3" if trials > 1 else "lh_product_guard",
        "stamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "harness": "localharness_product_guard",
        "note": "product guard (caller+digit+harm+strip); Door A stays raw",
        "trials": trials,
        "system": "slim" if SYSTEM == SYSTEM_SLIM else "full",
        "extra_tool": "--extra-tool" in flags,
        "models": {},
    }
    for model in models:
        print(f"\n=== LH product {model} ctx={DEFAULT_CTX} trials={trials} ===", flush=True)
        trial_rows: list[dict] = []
        for trial in range(1, trials + 1):
            print(f"-- trial {trial}/{trials} --", flush=True)
            trial_rows.append(run_trial(model, tools=tools))
        tasks: dict[str, dict] = {}
        for spec in TASKS:
            tid = spec["id"]
            shots = [tr.get(tid) or {"pass": False, "reason": "missing"} for tr in trial_rows]
            passes = [bool(s.get("pass")) for s in shots]
            tasks[tid] = {
                "pass": majority(passes) if trials > 1 else passes[0],
                "majority": f"{sum(passes)}/{len(passes)}",
                "level": spec["level"],
                "kind": spec["kind"],
                "vetoes": [s.get("vetoes") for s in shots],
                "reasons": [s.get("reason") for s in shots],
                "trials": shots,
            }
        row = {
            "num_ctx": DEFAULT_CTX,
            "tasks": tasks,
            "passed": sum(1 for t in tasks.values() if t.get("pass")),
            "total": len(tasks),
        }
        out["models"][model] = row
        unload(model)
    path = dest_dir / f"lh_product_{out['stamp']}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", path)
    for name, row in out["models"].items():
        print(f"SCORE {name} {row['passed']}/{row['total']}")
        if trials > 1:
            for tid, t in row["tasks"].items():
                print(f"  {tid} {t['majority']} pass={t['pass']} reasons={t['reasons']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
