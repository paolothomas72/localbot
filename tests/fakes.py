"""Fakes for S4 tests: a scripted / reactive model behind the REAL OllamaProvider (only the
HTTP transport is fake), a scripted shell, a slow provider, and a disk trial folder.
No live model calls, no GPU."""
from __future__ import annotations

import copy
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Callable

from localbot.config import ProviderConfig
from localbot.guard.workspace_fs import WorkspaceFS
from localbot.provider.ollama import ChatResult, OllamaProvider
from localbot.tools.fs import DiskWorkspace

REPO = Path(__file__).resolve().parents[1]
TRIALS = REPO / "workspaces" / "_trials"


def call(name: str, **args) -> dict:
    return {"function": {"name": name, "arguments": args}}


def reply(content: str = "", calls: list[dict] | None = None, thinking: str | None = None,
          done_reason: str = "stop", pe: int = 120, ev: int = 30) -> dict:
    """An Ollama /api/chat response (shape of the S1.4 recorded replies)."""
    msg: dict = {"role": "assistant", "content": content}
    if thinking is not None:
        msg["thinking"] = thinking
    if calls:
        msg["tool_calls"] = calls
    return {"model": "fake:1", "message": msg, "done": True, "done_reason": done_reason,
            "prompt_eval_count": pe, "eval_count": ev}


class ScriptedTransport:
    """Returns the scripted replies in order (deep copies). Records every request body."""

    def __init__(self, replies: list[dict | BaseException]) -> None:
        self.replies = list(replies)
        self.bodies: list[dict] = []

    def __call__(self, url: str, body: dict | None, timeout: float) -> dict:
        self.bodies.append(json.loads(json.dumps(body)))
        if not self.replies:
            raise AssertionError("fake model: script ran out")
        r = self.replies.pop(0)
        if isinstance(r, BaseException):
            raise r
        return copy.deepcopy(r)


class ReactiveTransport:
    """A fake model that decides from the conversation so far: fn(messages) -> reply dict."""

    def __init__(self, fn: Callable[[list[dict]], dict]) -> None:
        self.fn = fn
        self.bodies: list[dict] = []

    def __call__(self, url: str, body: dict | None, timeout: float) -> dict:
        self.bodies.append(json.loads(json.dumps(body)))
        return copy.deepcopy(self.fn(body["messages"]))


class RoutedTransport:
    """/api/chat -> a chat transport (scripted or reactive); /api/ps, /api/tags, /api/generate
    -> fixed recorded-shape answers. Records unloads."""

    def __init__(self, chat, ps_models: list[dict] | None = None, tags: list[dict] | None = None) -> None:
        self.chat = chat
        self.ps_models = ps_models or []
        self.tags = tags or []
        self.unloads: list[str] = []

    def __call__(self, url: str, body: dict | None, timeout: float) -> dict:
        if url.endswith("/api/chat"):
            return self.chat(url, body, timeout)
        if url.endswith("/api/ps"):
            return {"models": copy.deepcopy(self.ps_models)}
        if url.endswith("/api/tags"):
            return {"models": copy.deepcopy(self.tags)}
        if url.endswith("/api/generate"):
            self.unloads.append(body["model"])
            return {}
        raise AssertionError(f"unexpected url {url}")


def provider(transport, cfg: ProviderConfig | None = None) -> OllamaProvider:
    return OllamaProvider(cfg or ProviderConfig(), transport=transport)


class SlowProvider:
    """Never answers in time: sleeps ``sleep_s`` per call (the watchdog must not wait)."""

    def __init__(self, sleep_s: float) -> None:
        self.sleep_s = sleep_s
        self.calls = 0

    def chat(self, model, messages, tools, timeout, first_turn=False) -> ChatResult:
        self.calls += 1
        time.sleep(self.sleep_s)
        return ChatResult(data={}, message={"role": "assistant", "content": "late"}, wall_s=self.sleep_s)


class FakeShell:
    """Scripted bash: command -> output (or a callable for side effects)."""

    def __init__(self, script: dict[str, str | Callable[[], str]]) -> None:
        self.script = script
        self.ran: list[str] = []

    def run(self, command: str, timeout_s: float) -> str:
        self.ran.append(command)
        out = self.script.get(command, f"bash: {command.split()[0] if command else ''}: scripted miss")
        return out() if callable(out) else out


class Trial:
    def __init__(self) -> None:
        self.base = TRIALS / f"s4_{uuid.uuid4().hex[:12]}"
        self.root = self.base / "work"
        self.root.mkdir(parents=True)

    def ws(self, seed: dict[str, str]) -> DiskWorkspace:
        for rel, text in seed.items():
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8", newline="\n")
        return DiskWorkspace(WorkspaceFS(self.root), order=list(seed))

    def cleanup(self) -> None:
        shutil.rmtree(self.base, ignore_errors=True)
