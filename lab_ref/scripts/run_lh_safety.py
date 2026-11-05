#!/usr/bin/env python3
"""Product-path TRUST exam (10 tickets). Not Door A. Not a second core 12."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if "--slim" not in sys.argv:
    sys.argv.append("--slim")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
    sys.path.insert(0, str(_ROOT / "lib"))
    sys.path.insert(0, str(_ROOT / "scripts"))

from fit_roster import DEFAULT_CTX
from ollama_chat import unload
from paths import RAW
from run_lh_product import loop, majority
from safety_tasks import TASKS

FOCUS = ["qwen3-8b-ft-agentic-v1:latest"]


def run_trial(model: str) -> dict[str, dict]:
    tasks: dict[str, dict] = {}
    for spec in TASKS:
        result = loop(model, spec["prompt"], spec["judge"], spec["seed"], DEFAULT_CTX)
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
    dest_dir = RAW / "graded"
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "tag": "lh_safety_n3" if trials > 1 else "lh_safety",
        "stamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "harness": "localharness_product_guard",
        "note": "TRUST exam; Door A stays raw",
        "trials": trials,
        "models": {},
    }
    for model in models:
        print(f"\n=== LH safety {model} trials={trials} ===", flush=True)
        trial_rows: list[dict] = []
        for trial in range(1, trials + 1):
            print(f"-- trial {trial}/{trials} --", flush=True)
            trial_rows.append(run_trial(model))
        tasks: dict[str, dict] = {}
        for spec in TASKS:
            tid = spec["id"]
            shots = [tr.get(tid) or {"pass": False, "reason": "missing"} for tr in trial_rows]
            passes = [bool(s.get("pass")) for s in shots]
            tasks[tid] = {
                "pass": majority(passes) if trials > 1 else passes[0],
                "majority": f"{sum(passes)}/{len(passes)}",
                "kind": spec["kind"],
                "reasons": [s.get("reason") for s in shots],
                "vetoes": [s.get("vetoes") for s in shots],
                "trials": shots,
            }
        row = {
            "tasks": tasks,
            "passed": sum(1 for t in tasks.values() if t.get("pass")),
            "total": len(tasks),
        }
        out["models"][model] = row
        unload(model)
    path = dest_dir / f"lh_safety_{out['stamp']}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", path)
    for name, row in out["models"].items():
        print(f"TRUST {name} {row['passed']}/{row['total']}")
        if trials > 1:
            for tid, t in row["tasks"].items():
                print(f"  {tid} {t['majority']} pass={t['pass']} reasons={t['reasons']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
