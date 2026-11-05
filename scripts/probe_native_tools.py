"""One native /api/chat tool-call probe per model: do calls come back as tool_calls or as text?

    .venv\\Scripts\\python.exe scripts\\probe_native_tools.py MODEL [MODEL ...]

Per model: check nothing is loaded, send one request (temperature 0, one tool), record the
raw reply, check /api/ps for GPU fit, unload, check the unload. One model at a time.
Writes results/stage1/probe_native_tools_<stamp>.json. Localhost Ollama only.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from localbot.config import SYSTEM_SLIM, TOOLS_CORE4, get_profile  # noqa: E402
from localbot.parse.fn_call import FnCallConverter, has_tool_call_attempt  # noqa: E402

PROVIDER = get_profile("core4").provider
TOOL = [TOOLS_CORE4[0]]  # read(path): one simple tool
USER = "Read the file notes.txt and tell me what it says."


def _post(path: str, body: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        PROVIDER.base_url + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _ps() -> list[dict]:
    with urllib.request.urlopen(PROVIDER.base_url + "/api/ps", timeout=10) as resp:
        return json.loads(resp.read().decode()).get("models") or []


def probe(model: str) -> dict:
    before = [m.get("name") for m in _ps()]
    body = {
        "model": model,
        "stream": False,
        "keep_alive": "2m",
        "options": {"temperature": 0, "num_ctx": PROVIDER.num_ctx, "num_predict": PROVIDER.num_predict},
        "messages": [{"role": "system", "content": SYSTEM_SLIM}, {"role": "user", "content": USER}],
        "tools": TOOL,
    }
    t0 = time.perf_counter()
    data = _post("/api/chat", body)
    wall = time.perf_counter() - t0
    row = next((m for m in _ps() if m.get("name") == model), None)
    _post("/api/generate", {"model": model, "keep_alive": 0, "prompt": ""}, timeout=60)
    time.sleep(2)
    after = [m.get("name") for m in _ps()]

    msg = data.get("message") or {}
    native = msg.get("tool_calls") or []
    content = msg.get("content") or ""
    text_calls = FnCallConverter().extract_tool_calls(content) if content else []
    return {
        "model": model,
        "loaded_before": before,
        "request": {k: body[k] for k in ("options", "messages", "tools")},
        "wall_s": round(wall, 2),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "eval_count": data.get("eval_count"),
        "done_reason": data.get("done_reason"),
        "gpu": {"size": row.get("size"), "size_vram": row.get("size_vram")} if row else None,
        "message": msg,
        "call_source": "native" if native else ("text" if text_calls else "none"),
        "native_tool_calls": native,
        "text_parsed_calls": [{"name": c.name, "arguments": c.arguments} for c in text_calls],
        "text_has_tool_call_attempt": has_tool_call_attempt(content),
        "loaded_after_unload": after,
    }


def main(models: list[str]) -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows = []
    for model in models:
        print(f"=== {model}", flush=True)
        r = probe(model)
        print(json.dumps({k: r[k] for k in ("call_source", "native_tool_calls", "text_parsed_calls",
                                             "text_has_tool_call_attempt", "wall_s", "gpu",
                                             "loaded_before", "loaded_after_unload")}, ensure_ascii=False))
        print("content:", json.dumps(r["message"].get("content") or "", ensure_ascii=False)[:600], flush=True)
        rows.append(r)
    out = ROOT / "results" / "stage1" / f"probe_native_tools_{stamp}.json"
    out.write_text(json.dumps({"stamp": stamp, "user": USER, "tool": TOOL, "results": rows},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
