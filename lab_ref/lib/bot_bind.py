"""Local Bot bind: one bot = one local model. No GPU. Files only.

Store lives under the project folder at `.localbot/` (ProductGuard + workspace
load skip this tree so the model cannot rewrite its own brain).
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

SCHEMA = "localbot.bot/v1"
_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
_RUNTIMES = {"ollama", "llamacpp"}


class BotBindError(ValueError):
    pass


def store_dir(workspace: Path | None = None) -> Path:
    raw = workspace or os.environ.get("LOCAL_LAB_WORKSPACE") or os.getcwd()
    return Path(raw).expanduser().resolve() / ".localbot"


def bots_dir(workspace: Path | None = None) -> Path:
    return store_dir(workspace) / "bots"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def validate(bot: dict[str, Any]) -> None:
    if bot.get("schema") != SCHEMA:
        raise BotBindError(f"schema must be {SCHEMA}")
    bid = str(bot.get("id") or "")
    if not _ID.match(bid):
        raise BotBindError("id must be lowercase letters, digits, _ or -")
    if not str(bot.get("name") or "").strip():
        raise BotBindError("name required")
    rt = str(bot.get("runtime") or "")
    if rt not in _RUNTIMES:
        raise BotBindError("runtime must be ollama or llamacpp")
    model = str(bot.get("model") or "").strip()
    if not model:
        raise BotBindError("model required — a bot is not a bot until a local tag is bound")
    if "," in model or " " in model.strip():
        raise BotBindError("one bot = one model tag (no comma list)")
    extra = bot.get("models")
    if extra:
        raise BotBindError("do not set models[]; one bot = one model. Switch = new session.")
    ep = str(bot.get("endpoint") or "").strip()
    if not ep.startswith("http://") and not ep.startswith("https://"):
        raise BotBindError("endpoint must be local HTTP (Ollama or llama.cpp)")
    if "openai.com" in ep or "anthropic.com" in ep or "openrouter" in ep:
        raise BotBindError("Local Bot runtime is local-only. No cloud provider URL.")


def new_bot(
    bot_id: str,
    *,
    name: str | None = None,
    model: str,
    runtime: str = "ollama",
    endpoint: str | None = None,
    instructions: str = "",
) -> dict[str, Any]:
    if runtime == "llamacpp":
        endpoint = endpoint or "http://127.0.0.1:8090"
    else:
        endpoint = endpoint or "http://127.0.0.1:11434"
    bot = {
        "schema": SCHEMA,
        "id": bot_id.strip().lower(),
        "name": (name or bot_id).strip(),
        "instructions": instructions.strip(),
        "runtime": runtime,
        "model": model.strip(),
        "endpoint": endpoint,
        "saves": [],
        "created_at": _now(),
        "updated_at": _now(),
    }
    validate(bot)
    return bot


def save(bot: dict[str, Any], workspace: Path | None = None) -> Path:
    validate(bot)
    bot["updated_at"] = _now()
    d = bots_dir(workspace)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{bot['id']}.json"
    path.write_text(json.dumps(bot, indent=2) + "\n", encoding="utf-8")
    return path


def load(bot_id: str, workspace: Path | None = None) -> dict[str, Any]:
    path = bots_dir(workspace) / f"{bot_id}.json"
    if not path.is_file():
        raise BotBindError(f"no bot {bot_id}")
    bot = json.loads(path.read_text(encoding="utf-8"))
    validate(bot)
    return bot


def list_bots(workspace: Path | None = None) -> list[dict[str, Any]]:
    d = bots_dir(workspace)
    if not d.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(d.glob("*.json")):
        bot = json.loads(path.read_text(encoding="utf-8"))
        validate(bot)
        out.append(bot)
    return out


def bind_model(bot_id: str, model: str, workspace: Path | None = None) -> dict[str, Any]:
    bot = load(bot_id, workspace)
    new = model.strip()
    if bot["model"] != new:
        raise BotBindError(
            f"bot {bot_id} is bound to {bot['model']}. "
            "Switching models is a new session (copy id + compact), not a silent rebind."
        )
    return bot


def rebind_new_session(
    bot_id: str,
    model: str,
    workspace: Path | None = None,
    new_id: str | None = None,
) -> dict[str, Any]:
    """Explicit handoff: new bot id, empty saves, same instructions."""
    old = load(bot_id, workspace)
    nid = (new_id or f"{bot_id}-s2").strip().lower()
    if nid == old["id"]:
        raise BotBindError("new session needs a new id")
    bot = new_bot(
        nid,
        name=old["name"],
        model=model,
        runtime=str(old["runtime"]),
        endpoint=str(old["endpoint"]),
        instructions=str(old.get("instructions") or ""),
    )
    bot["handoff_from"] = old["id"]
    return bot


def active_path(workspace: Path | None = None) -> Path:
    return store_dir(workspace) / "active.json"


def activate(bot_id: str, workspace: Path | None = None) -> dict[str, Any]:
    bot = load(bot_id, workspace)
    ap = active_path(workspace)
    if ap.is_file():
        cur = json.loads(ap.read_text(encoding="utf-8"))
        other = str(cur.get("bot_id") or "")
        if other and other != bot_id:
            raise BotBindError(
                f"concurrent=1: {other} is active. Deactivate it before {bot_id}."
            )
    payload = {"bot_id": bot_id, "model": bot["model"], "since": _now()}
    ap.parent.mkdir(parents=True, exist_ok=True)
    ap.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def deactivate(workspace: Path | None = None) -> None:
    ap = active_path(workspace)
    if ap.is_file():
        ap.unlink()
