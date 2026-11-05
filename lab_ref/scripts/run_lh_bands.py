#!/usr/bin/env python3
"""Four-hands or deep-trust band. Does not replace core 12 or safety 10.

  python scripts/run_lh_bands.py --hands --n3 qwen3-8b-ft-agentic-v1:latest
  python scripts/run_lh_bands.py --trust2 --n3 qwen3-8b-ft-agentic-v1:latest
  python scripts/run_lh_bands.py --session --n3 qwen3-8b-ft-agentic-v1:latest
  python scripts/run_lh_bands.py --privacy --n3 qwen3-8b-ft-agentic-v1:latest
  python scripts/run_lh_bands.py --gap --n3 qwen3-8b-ft-agentic-v1:latest
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if "--slim" not in sys.argv:
    sys.argv.append("--slim")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
    sys.path.insert(0, str(_ROOT / "lib"))
    sys.path.insert(0, str(_ROOT / "scripts"))

from fit_roster import DEFAULT_CTX
from four_hands_tasks import TASKS as HANDS
from ollama_chat import unload
from paths import RAW
from run_lh_product import loop, majority
from session_pack import TASKS as SESSION
from sandbox_privacy_tasks import TASKS as PRIVACY
from gap_pack import TASKS as GAP
from trust_deep_tasks import TASKS as TRUST2

FOCUS = ["qwen3-8b-ft-agentic-v1:latest"]


def run_trial(model: str, tasks: list) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for spec in tasks:
        result = loop(
            model,
            spec["prompt"],
            spec["judge"],
            spec["seed"],
            DEFAULT_CTX,
            meta={"id": spec["id"], "kind": spec["kind"]},
        )
        result["level"] = spec["level"]
        result["kind"] = spec["kind"]
        out[spec["id"]] = result
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
    return out


def main() -> int:
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    models = [a for a in sys.argv[1:] if not a.startswith("--")] or FOCUS
    trials = 3 if "--n3" in flags else 1
    if "--trust2" in flags:
        tasks, tag, word = TRUST2, "trust2", "TRUST2"
    elif "--session" in flags:
        tasks, tag, word = SESSION, "session", "SESSION"
    elif "--privacy" in flags:
        tasks, tag, word = PRIVACY, "privacy", "PRIVACY"
    elif "--gap" in flags:
        tasks, tag, word = GAP, "gap", "GAP"
    else:
        tasks, tag, word = HANDS, "hands", "HANDS"
    dest_dir = RAW / "graded"
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "tag": f"lh_{tag}_n3" if trials > 1 else f"lh_{tag}",
        "stamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "harness": "localharness_product_guard",
        "trials": trials,
        "models": {},
    }
    for model in models:
        print(f"\n=== {word} {model} trials={trials} ===", flush=True)
        rows = []
        for trial in range(1, trials + 1):
            print(f"-- trial {trial}/{trials} --", flush=True)
            rows.append(run_trial(model, tasks))
        scored = {}
        for spec in tasks:
            tid = spec["id"]
            shots = [tr.get(tid) or {"pass": False, "reason": "missing"} for tr in rows]
            passes = [bool(s.get("pass")) for s in shots]
            scored[tid] = {
                "pass": majority(passes) if trials > 1 else passes[0],
                "majority": f"{sum(passes)}/{len(passes)}",
                "kind": spec["kind"],
                "reasons": [s.get("reason") for s in shots],
                "trials": shots,
            }
        row = {"tasks": scored, "passed": sum(1 for t in scored.values() if t.get("pass")), "total": len(scored)}
        out["models"][model] = row
        unload(model)
    path = dest_dir / f"lh_{tag}_{out['stamp']}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", path)
    for name, row in out["models"].items():
        print(f"{word} {name} {row['passed']}/{row['total']}")
        if trials > 1:
            for tid, t in row["tasks"].items():
                print(f"  {tid} {t['majority']} pass={t['pass']} reasons={t['reasons']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
