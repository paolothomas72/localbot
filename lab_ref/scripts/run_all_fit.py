#!/usr/bin/env python3
"""Graded battery on every 12GB-fit model, at that model's runnable context.

  python scripts/run_all_fit.py --tag fit50
  python scripts/run_all_fit.py --only granite4:tiny-h
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
    sys.path.insert(0, str(_ROOT / "lib"))

from fit_roster import DEFAULT_CTX, FIT_MODELS
from graded_tasks import TASKS
from ollama_chat import chat, unload
from paths import RAW

DEADLINE_TRIAL_S = 900  # 15 min per full 12-task trial; then TIMEOUT remaining tasks
TASK_TIMEOUT_S = 180
N3_QWEN = [
    "qwen3-8b-ft-agentic-v1:latest",
    "qwen3:8b",
    "qwen3-8b-gold-v2:latest",
]
SYSTEM = (
    "You are a local coding agent. Use tools. Be honest. "
    "Strip line-number prefixes from read() before edit or write. "
    "Do the smallest change that matches intent. "
    "If asked for JSON only, reply with JSON only."
)
TOOLS = [
    {"type": "function", "function": {"name": "read", "description": "Read a file. Numbered lines (N\\tcontent).", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write", "description": "Create or overwrite a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit", "description": "Replace old_string with new_string.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}, "required": ["path", "old_string", "new_string"]}}},
    {"type": "function", "function": {"name": "glob", "description": "List files.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search file contents.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "bash_exec", "description": "Run bash. Often fails here.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]


def load_ctx_map() -> dict[str, int]:
    p = RAW / "context" / "runnable_ctx.json"
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for name, row in (data.get("models") or {}).items():
        out[name] = int(row.get("runnable_ctx") or DEFAULT_CTX)
    return out


def run_tool(name: str, args: dict, files: dict[str, str]) -> str:
    path = args.get("path") or ""
    if name == "read":
        body = files.get(path)
        if body is None:
            return f"ERROR: not found: {path}"
        return "\n".join(f"{i}\t{line}" for i, line in enumerate(body.splitlines() or [""], 1))
    if name == "write":
        files[path] = args.get("content") or ""
        return f"Wrote {path}"
    if name == "edit":
        cur = files.get(path, "")
        old, new = args.get("old_string") or "", args.get("new_string") or ""
        if old not in cur:
            return "ERROR: old_string not found"
        files[path] = cur.replace(old, new, 1)
        return f"Updated {path}"
    if name == "glob":
        return "\n".join(sorted(files)) or "(empty)"
    if name == "grep":
        pat = args.get("pattern") or ""
        hits = []
        for p, body in files.items():
            if path and path not in (".", p) and not p.startswith(path.rstrip("/")):
                if p != path:
                    continue
            for i, line in enumerate(body.splitlines(), 1):
                if pat in line:
                    hits.append(f"{p}:{i}:{line}")
        return "\n".join(hits) or "(no matches)"
    if name == "bash_exec":
        return "ERROR: bash not found"
    return f"ERROR: unknown tool {name}"


def loop(model: str, user: str, judge, seed: dict, num_ctx: int) -> dict:
    files = dict(seed)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    elapsed = 0.0
    eval_count = 0
    tools_used: list[str] = []
    last_text = ""
    for _ in range(6):
        try:
            data, dt = chat(model, messages, tools=TOOLS, num_ctx=num_ctx, timeout=TASK_TIMEOUT_S)
        except Exception as exc:
            return {
                "pass": False,
                "reason": f"request failed: {exc}",
                "elapsed_s": round(elapsed, 2),
                "tools": tools_used,
                "tok_s": 0,
            }
        elapsed += dt
        eval_count += data.get("eval_count") or 0
        msg = data.get("message") or {}
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
            messages.append({"role": "tool", "content": run_tool(name, args, files), "name": name})
    ok, reason = judge(files, tools_used, last_text)
    return {
        "pass": ok,
        "reason": reason,
        "elapsed_s": round(elapsed, 2),
        "tok_s": round(eval_count / elapsed, 1) if elapsed else 0.0,
        "tools": tools_used,
        "text": last_text,
        "num_ctx": num_ctx,
    }


def save(out: dict, dest_dir: Path, tag: str, *, touch_latest: bool) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    tagged = dest_dir / f"all_fit_{tag}_{out['stamp']}.json"
    text = json.dumps(out, indent=2)
    tagged.write_text(text, encoding="utf-8")
    if touch_latest:
        (dest_dir / "all_fit_latest.json").write_text(text, encoding="utf-8")
    return tagged


def majority(passes: list[bool]) -> bool:
    return sum(1 for p in passes if p) >= (len(passes) // 2 + 1)


def run_one_trial(model: str, num_ctx: int) -> dict[str, dict]:
    tasks: dict[str, dict] = {}
    t0 = time.perf_counter()
    for spec in TASKS:
        if time.perf_counter() - t0 > DEADLINE_TRIAL_S:
            tasks[spec["id"]] = {
                "pass": False,
                "reason": "TIMEOUT",
                "level": spec["level"],
                "kind": spec["kind"],
                "num_ctx": num_ctx,
            }
            print(json.dumps({"model": model, "id": spec["id"], "pass": False, "reason": "TIMEOUT"}), flush=True)
            continue
        result = loop(model, spec["prompt"], spec["judge"], spec["seed"], num_ctx)
        result["level"] = spec["level"]
        result["kind"] = spec["kind"]
        tasks[spec["id"]] = result
        print(
            json.dumps(
                {
                    "model": model,
                    "id": spec["id"],
                    "level": spec["level"],
                    "pass": result["pass"],
                    "elapsed_s": result["elapsed_s"],
                    "tok_s": result["tok_s"],
                    "reason": result["reason"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fit50")
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--n3-qwen", action="store_true", help="qwen-ft + base + gold-v2, 3 trials, do not overwrite all_fit_latest")
    args = ap.parse_args()
    if args.n3_qwen:
        args.trials = 3
        args.only = list(N3_QWEN)
        args.tag = args.tag if args.tag != "fit50" else "n3"
    ctx_map = load_ctx_map()
    models = args.only or FIT_MODELS
    dest_dir = RAW / "graded"
    touch_latest = args.trials == 1 and not args.n3_qwen
    out = {
        "tag": args.tag,
        "stamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "harness": "localharness_names",
        "trials": args.trials,
        "ctx_source": "runnable_ctx.json" if ctx_map else "default_8192",
        "models": {},
    }
    for model in models:
        num_ctx = ctx_map.get(model) or ctx_map.get(model.split(":")[0]) or ctx_map.get(model.replace(":latest", "")) or DEFAULT_CTX
        if "gold-v2" in model and num_ctx == DEFAULT_CTX:
            num_ctx = 32768
        print(f"\n=== LH {model} ctx={num_ctx} trials={args.trials} ===", flush=True)
        trial_rows: list[dict] = []
        for trial in range(1, args.trials + 1):
            print(f"-- trial {trial}/{args.trials} --", flush=True)
            trial_rows.append(run_one_trial(model, num_ctx))
        tasks: dict[str, dict] = {}
        for spec in TASKS:
            tid = spec["id"]
            shots = [tr.get(tid) or {"pass": False, "reason": "missing"} for tr in trial_rows]
            passes = [bool(s.get("pass")) for s in shots]
            tasks[tid] = {
                "pass": majority(passes) if args.trials > 1 else passes[0],
                "majority": f"{sum(passes)}/{len(passes)}",
                "level": spec["level"],
                "kind": spec["kind"],
                "num_ctx": num_ctx,
                "trials": shots,
            }
        row = {
            "num_ctx": num_ctx,
            "tasks": tasks,
            "passed": sum(1 for t in tasks.values() if t.get("pass")),
            "total": len(tasks),
        }
        out["models"][model] = row
        unload(model)
        save(out, dest_dir, args.tag, touch_latest=touch_latest)
    path = save(out, dest_dir, args.tag, touch_latest=touch_latest)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
