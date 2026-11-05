"""Shared Ollama chat + GPU-fit helpers."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

CHAT = "http://localhost:11434/api/chat"
GENERATE = "http://localhost:11434/api/generate"
PS = "http://localhost:11434/api/ps"


def ollama_json(url: str, body: dict, timeout: int = 180) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def chat(
    model: str,
    messages: list,
    *,
    tools: list | None,
    num_ctx: int,
    timeout: int = 180,
    temperature: float = 0.2,
    num_predict: int = 512,
    keep_alive: str = "2m",
) -> tuple[dict, float]:
    body: dict = {
        "model": model,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "num_ctx": num_ctx,
            "temperature": temperature,
            "num_predict": num_predict,
        },
        "messages": messages,
    }
    if tools:
        body["tools"] = tools
    t0 = time.perf_counter()
    data = ollama_json(CHAT, body, timeout=timeout)
    return data, time.perf_counter() - t0


def unload(model: str) -> None:
    try:
        ollama_json(GENERATE, {"model": model, "keep_alive": 0, "prompt": ""}, timeout=30)
    except Exception:
        pass


def loaded() -> list[dict]:
    try:
        with urllib.request.urlopen(PS, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        return data.get("models") or []
    except Exception:
        return []


def gpu_row(model: str) -> dict | None:
    short = model.split(":")[0]
    for m in loaded():
        name = m.get("name") or m.get("model") or ""
        if name == model or name.startswith(short):
            return m
    return None


def is_100pct_gpu(model: str) -> tuple[bool, str]:
    row = gpu_row(model)
    if not row:
        return False, "not loaded"
    size = int(row.get("size") or 0)
    vram = int(row.get("size_vram") or 0)
    gb = vram / (1024**3) if vram else 0.0
    if size and vram and vram < size:
        spill = (size - vram) / (1024**3)
        return False, f"SPILL {spill:.2f}GB off VRAM ({gb:.2f}GB on GPU)"
    return True, f"100% GPU ({gb:.2f}GB)"
