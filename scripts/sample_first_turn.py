"""Reviewer test (not part of the repo; state/ is gitignored): send the exact parity_lab6
first /api/chat request for one task N times and count the model's first tool choice.

No tool is executed, no workspace is touched. The body is rebuilt with Local Bot's own
lab_chat_body and must match the sha256 S8f.2 captured (byte-identical to the lab's), or the
script refuses to send.

  python -I state\\tools\\sample_first_turn.py [--n 30] [--dry-run]
"""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True  # never write __pycache__ (keeps lab_ref clean)
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "lab_ref" / "lib")]

from localbot.config import SYSTEM_SLIM, parity_lab6  # noqa: E402
from localbot.provider.ollama import lab_chat_body, urllib_transport  # noqa: E402
from safety_tasks import TASKS  # noqa: E402  (lab, read-only)

MODEL = "hf.co/deepreinforce-ai/Ornith-1.0-9B-GGUF:latest"
TASK = "s_clean_list"
S8F2_SHA16 = "4447e52aa7d4e1e1"  # S8f.2 capture, all three sides


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    prof = parity_lab6()
    spec = next(t for t in TASKS if t["id"] == TASK)
    messages = [{"role": "system", "content": SYSTEM_SLIM}, {"role": "user", "content": spec["prompt"]}]
    body = lab_chat_body(prof.provider, MODEL, messages, list(prof.tools.schemas))
    raw = json.dumps(body).encode()
    sha = hashlib.sha256(raw).hexdigest()[:16]
    print(f"body {len(raw)} B sha256 {sha} (S8f.2: 1923 B {S8F2_SHA16}) -> {'MATCH' if sha == S8F2_SHA16 else 'MISMATCH'}")
    if sha != S8F2_SHA16:
        print("refusing to send: not the captured request")
        return 2
    if a.dry_run:
        return 0

    base = prof.provider.base_url
    get = lambda p: urllib_transport(f"{base}{p}", None, 8)
    ver = get("/api/version").get("version")
    digest = next((m.get("digest", "")[:12] for m in get("/api/tags").get("models", [])
                   if (m.get("name") or m.get("model")) == MODEL), None)
    print(f"ollama {ver}  model digest {digest}  n={a.n}")

    rows, tally = [], Counter()
    for i in range(1, a.n + 1):
        t0 = time.perf_counter()
        data = urllib_transport(f"{base}/api/chat", body, 180)
        msg = data.get("message") or {}
        calls = msg.get("tool_calls") or []
        if calls:
            fn = calls[0].get("function") or {}
            first, args = fn.get("name"), fn.get("arguments")
        else:
            first, args = "text_only", None
        tally[first] += 1
        rows.append({"i": i, "first_tool": first, "args": args, "n_calls": len(calls),
                     "prompt_eval_count": data.get("prompt_eval_count"),
                     "wall_s": round(time.perf_counter() - t0, 2)})
        print(f"{i:>3} {first:<10} {json.dumps(args)[:70]}  pe={data.get('prompt_eval_count')}")

    urllib_transport(f"{base}/api/generate", {"model": MODEL, "keep_alive": 0, "prompt": ""}, 30)
    print("TALLY", dict(tally), f"glob share {tally['glob']}/{a.n}")
    out = REPO / "state" / "runs" / f"sample_first_turn_{TASK}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"task": TASK, "model": MODEL, "digest": digest, "ollama": ver,
                               "body_sha16": sha, "n": a.n, "tally": dict(tally), "rows": rows}, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
