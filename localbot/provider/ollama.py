"""Ollama native /api/chat provider (docs/STAGE1_SPEC.md §5.2, step S4).

The request is the lab's exactly (lab_ref/lib/ollama_chat.py ``chat``): native ``/api/chat``,
``stream: false``, ``keep_alive``, ``options = {num_ctx, temperature, num_predict}``,
``messages``, and ``tools`` only when there are tools. tests/test_provider.py builds the lab's
body and this one from the same input and requires them to be equal.

Added per call (recorded, never sent): ``prompt_eval_count``, ``eval_count``, wall time,
attempts / ``infra_retry`` (one retry when the error mentions HTTP 500, as the lab did), a
*possible* silent-truncation flag, and thinking size against ``num_predict``.

GPU fit: ``gpu_fit`` combines ``/api/ps`` (size vs size_vram, the lab's check) with an
``nvidia-smi`` VRAM-delta cross-check and the on-disk size from ``/api/tags``, because
``/api/ps`` alone was seen to under-report (gemma4:12b: 1.14 GB vs 8.0 GB on disk, S1.4).

Talks to the configured Ollama URL only (localhost, enforced in config). It never sends
anything to llama-server on :8090 and never stops it.
"""
from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from localbot.config import ProviderConfig
from localbot.telemetry import VramReading

# transport(url, body or None for GET, timeout_s) -> decoded JSON. Tests pass a fake.
Transport = Callable[[str, dict | None, float], dict]

CHARS_PER_TOKEN = 4.0  # rough estimate for flags only; never used for a decision that counts


class ProviderError(RuntimeError):
    """The provider call failed (after the allowed retry). Ends the task: ``provider_error``."""


class ProviderTimeout(ProviderError):
    """The call ran past its time budget. Ends the task: ``task_timeout`` / ``trial_deadline``."""


def urllib_transport(url: str, body: dict | None, timeout: float) -> dict:
    if body is None:
        req = urllib.request.Request(url, method="GET")
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (socket.timeout, TimeoutError) as exc:
        raise ProviderTimeout(f"timed out after {timeout:.0f}s") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (socket.timeout, TimeoutError)):
            raise ProviderTimeout(f"timed out after {timeout:.0f}s") from exc
        raise


def _is_connection_error(exc: BaseException) -> bool:
    """Refused, reset or dropped before an HTTP status came back (not a timeout, not an HTTP error)."""
    if isinstance(exc, urllib.error.HTTPError):
        return False
    if isinstance(exc, ConnectionError):  # refused, reset, aborted, http.client.RemoteDisconnected
        return True
    return (isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, OSError)
            and not isinstance(exc.reason, (socket.timeout, TimeoutError)))


def _error_text(exc: BaseException) -> str:
    """``str(exc)`` plus, for an HTTP error, the body Ollama puts the cause in (``{"error": …}``),
    which ``str`` drops. ``str(exc)`` stays first: runner.is_infra matches "HTTP Error 500"."""
    text = str(exc)
    if isinstance(exc, urllib.error.HTTPError):
        try:
            raw = exc.read() or b""
        except Exception:
            raw = b""
        body = raw.decode("utf-8", "replace").strip()
        if body:
            text += f": {body[:500]}"
    return text


def _ns_to_s(ns: Any) -> float | None:
    """Ollama durations are integer nanoseconds."""
    return ns / 1e9 if isinstance(ns, (int, float)) and not isinstance(ns, bool) and ns > 0 else None


def estimate_tokens(obj: Any) -> int:
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return int(len(text) / CHARS_PER_TOKEN)


@dataclass
class ChatResult:
    data: dict  # the raw response, untouched
    message: dict  # data["message"], exactly as returned (thinking and tool_calls included)
    wall_s: float
    attempts: int = 1
    infra_retry: bool = False
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    done_reason: str | None = None
    est_prompt_tokens: int = 0
    truncation_suspect: bool = False
    truncation_why: str = ""
    thinking_chars: int = 0
    thinking_tokens_est: int = 0
    hit_num_predict: bool = False
    eval_s: float | None = None  # H1 P2: Ollama's eval_duration (generation only), seconds
    load_s: float | None = None  # load_duration: model load, which wall_s includes

    @property
    def tok_s(self) -> float | None:
        """Generation speed: eval_count / eval_duration. None when Ollama gave no duration
        (eval_count / wall_s would count the model load: 5.2 vs ~95 tok/s in real data)."""
        if not self.eval_count or not self.eval_s:
            return None
        return round(self.eval_count / self.eval_s, 1)

    def stats(self) -> dict:
        """What the loop records per call (no message bodies)."""
        return {
            "wall_s": round(self.wall_s, 3),
            "attempts": self.attempts,
            "infra_retry": self.infra_retry,
            "prompt_eval_count": self.prompt_eval_count,
            "eval_count": self.eval_count,
            "done_reason": self.done_reason,
            "est_prompt_tokens": self.est_prompt_tokens,
            "truncation_suspect": self.truncation_suspect,
            "truncation_why": self.truncation_why,
            "thinking_chars": self.thinking_chars,
            "thinking_tokens_est": self.thinking_tokens_est,
            "hit_num_predict": self.hit_num_predict,
            "eval_s": self.eval_s,
            "load_s": self.load_s,
            "tok_s": self.tok_s,
        }


def lab_chat_body(cfg: ProviderConfig, model: str, messages: list, tools: list | None) -> dict:
    """The lab's request body, field for field (ollama_chat.chat)."""
    body: dict = {
        "model": model,
        "stream": cfg.stream,
        "keep_alive": cfg.keep_alive,
        "options": {
            "num_ctx": cfg.num_ctx,
            "temperature": cfg.temperature,
            "num_predict": cfg.num_predict_for(model),  # = cfg.num_predict unless set per model (L-6)
        },
        "messages": messages,
    }
    if tools:
        body["tools"] = tools
    return body


def truncation_check(est_prompt: int, prompt_eval: int | None, num_ctx: int, first_turn: bool) -> tuple[bool, str]:
    """Possible silent truncation. Ollama cuts a prompt longer than num_ctx without an error.

    Flags (a *suspicion*, never a verdict): the estimate is over num_ctx; or, on the first
    turn only (later turns reuse the KV cache, so prompt_eval_count counts only new tokens),
    prompt_eval_count is under half the estimate.
    """
    if est_prompt > num_ctx:
        return True, f"estimated prompt {est_prompt} tok > num_ctx {num_ctx}"
    if first_turn and prompt_eval is not None and est_prompt >= 200 and prompt_eval < 0.5 * est_prompt:
        return True, f"prompt_eval_count {prompt_eval} < half of estimate {est_prompt}"
    return False, ""


def thinking_size(message: dict, eval_count: int | None) -> tuple[int, int]:
    """(thinking chars, estimated thinking tokens). Ollama reports one eval_count for thinking +
    content + tool calls together, so the thinking share is apportioned by characters."""
    thinking = str(message.get("thinking") or "")
    if not thinking:
        return 0, 0
    calls = message.get("tool_calls")
    rest = len(str(message.get("content") or "")) + (len(json.dumps(calls)) if calls else 0)
    if eval_count:
        share = len(thinking) / max(1, len(thinking) + rest)
        return len(thinking), int(round(eval_count * share))
    return len(thinking), int(len(thinking) / CHARS_PER_TOKEN)


class OllamaProvider:
    def __init__(self, cfg: ProviderConfig, transport: Transport | None = None,
                 clock: Callable[[], float] = time.perf_counter,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.cfg = cfg
        self.transport = transport or urllib_transport
        self.clock = clock
        self.sleep = sleep

    @property
    def chat_url(self) -> str:
        return f"{self.cfg.base_url}/api/chat"

    def chat(self, model: str, messages: list, tools: list | None, timeout: float,
             first_turn: bool = False) -> ChatResult:
        body = lab_chat_body(self.cfg, model, messages, tools)
        attempts, infra_retry, last_exc, last_text = 0, False, None, ""
        http500_left, conn_left = self.cfg.retries_on_http_500, self.cfg.retries_on_connection_error
        pause = self.cfg.retry_backoff_s
        t0 = self.clock()
        data = None
        while True:
            attempts += 1
            try:
                data = self.transport(self.chat_url, body, timeout)
                last_exc = None
                break
            except ProviderTimeout:
                raise
            except Exception as exc:
                last_exc, last_text = exc, _error_text(exc)
                if "500" in str(exc) and http500_left > 0:  # lab: once, at once, when it mentions 500
                    http500_left -= 1
                    infra_retry = True
                    continue
                if _is_connection_error(exc) and conn_left > 0:  # H1 P2: with a growing pause
                    conn_left -= 1
                    infra_retry = True
                    self.sleep(pause)
                    pause *= 2
                    continue
                break
        wall = self.clock() - t0
        if data is None:
            raise ProviderError(f"request failed: {last_text}") from last_exc
        if isinstance(data, dict) and data.get("error") and not data.get("message"):
            # H1 P2: a 200 that carries {"error": ...} is a failed call, not an empty answer
            raise ProviderError(f"provider error: {str(data['error'])[:500]}")
        msg = data.get("message") or {}
        pe, ev = data.get("prompt_eval_count"), data.get("eval_count")
        est = estimate_tokens(messages) + (estimate_tokens(tools) if tools else 0)
        suspect, why = truncation_check(est, pe, self.cfg.num_ctx, first_turn)
        t_chars, t_tok = thinking_size(msg, ev)
        return ChatResult(
            data=data, message=msg, wall_s=wall, attempts=attempts, infra_retry=infra_retry,
            prompt_eval_count=pe, eval_count=ev, done_reason=data.get("done_reason"),
            est_prompt_tokens=est, truncation_suspect=suspect, truncation_why=why,
            thinking_chars=t_chars, thinking_tokens_est=t_tok,
            hit_num_predict=bool(data.get("done_reason") == "length"
                                 or (ev is not None and ev >= self.cfg.num_predict_for(model))),
            eval_s=_ns_to_s(data.get("eval_duration")), load_s=_ns_to_s(data.get("load_duration")),
        )

    def ps(self) -> list[dict]:
        try:
            return self.transport(f"{self.cfg.base_url}/api/ps", None, 8).get("models") or []
        except Exception:
            return []

    def tags(self) -> list[dict]:
        try:
            return self.transport(f"{self.cfg.base_url}/api/tags", None, 8).get("models") or []
        except Exception:
            return []

    def version(self) -> str | None:
        """Ollama's reported version (``/api/version``), or None if it can't be read (S8f)."""
        try:
            return self.transport(f"{self.cfg.base_url}/api/version", None, 8).get("version")
        except Exception:
            return None

    def unload(self, model: str) -> None:
        """The lab's unload: /api/generate with keep_alive 0 for this one model."""
        try:
            self.transport(f"{self.cfg.base_url}/api/generate", {"model": model, "keep_alive": 0, "prompt": ""}, 30)
        except Exception:
            pass


# --- GPU fit ----------------------------------------------------------------------------------


def ps_row(models: list[dict], model: str) -> dict | None:
    """The lab's gpu_row matching (exact name, else same name before the ':')."""
    short = model.split(":")[0]
    for m in models:
        name = m.get("name") or m.get("model") or ""
        if name == model or name.startswith(short):
            return m
    return None


def disk_size(tags: list[dict], model: str) -> int | None:
    for m in tags:
        if (m.get("name") or m.get("model")) == model:
            return int(m.get("size") or 0) or None
    return None


@dataclass
class GpuFit:
    """Owner decision, S5: ``unknown`` (or ``spill``) keeps its reason and only takes the model's
    TIMING out of comparisons. Safety results are always run and counted; a model is never
    dropped from G5/G6 because of its lane."""

    lane: str  # "gpu" | "spill" | "unknown"
    ps_says: str
    reasons: list[str] = field(default_factory=list)

    @property
    def timing_comparable(self) -> bool:
        return self.lane == "gpu"

    def as_dict(self) -> dict:
        return {"lane": self.lane, "ps_says": self.ps_says, "reasons": list(self.reasons),
                "timing_comparable": self.timing_comparable, "safety_counted": True}


GIB = 1024**3


def gpu_fit(model: str, ps_models: list[dict], disk_bytes: int | None,
            vram_before: VramReading | None, vram_after: VramReading | None,
            ps_disk_ratio_min: float = 0.5, delta_ratio_min: float = 0.5) -> GpuFit:
    """Decide the lane. ``gpu`` only when /api/ps says 100% GPU AND nothing contradicts it.

    - /api/ps: size_vram < size -> spill (the lab's rule).
    - /api/ps size far below the on-disk size -> /api/ps cannot be trusted for this model.
    - nvidia-smi delta (used after load - used before) far below what /api/ps says is on the
      GPU -> the GPU did not actually take that much: not trusted.
    Any distrust -> ``unknown`` (timing-only, like spill, until a human looks).
    """
    row = ps_row(ps_models, model)
    if not row:
        return GpuFit("unknown", "not loaded", ["model not in /api/ps"])
    size, vram = int(row.get("size") or 0), int(row.get("size_vram") or 0)
    if size and not vram:  # lab bug: `size and vram and ...` called an all-CPU model 100% GPU
        return GpuFit("spill", f"SPILL {size / GIB:.2f}GB off VRAM (0.00GB on GPU)", ["size_vram is 0"])
    if size and vram and vram < size:
        return GpuFit("spill", f"SPILL {(size - vram) / GIB:.2f}GB off VRAM ({vram / GIB:.2f}GB on GPU)")
    fit = GpuFit("gpu", f"100% GPU ({vram / GIB:.2f}GB)")
    if disk_bytes and size and size < ps_disk_ratio_min * disk_bytes:
        fit.reasons.append(f"/api/ps size {size / GIB:.2f}GB < {ps_disk_ratio_min:.0%} of on-disk {disk_bytes / GIB:.2f}GB")
    if vram_before is not None and vram_after is not None and vram:
        delta_b = (vram_after.used_mib - vram_before.used_mib) * 1024 * 1024
        if delta_b < delta_ratio_min * vram:
            fit.reasons.append(
                f"nvidia-smi delta {delta_b / GIB:.2f}GB < {delta_ratio_min:.0%} of /api/ps size_vram {vram / GIB:.2f}GB")
    elif vram_before is None or vram_after is None:
        fit.reasons.append("no nvidia-smi reading: cross-check not done")
        return GpuFit("unknown", fit.ps_says, fit.reasons)
    if fit.reasons:
        fit.lane = "unknown"
    return fit
