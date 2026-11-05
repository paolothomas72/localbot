"""provider/ollama.py with recorded responses only — no live model calls (S4, spec §5.2).

Fixtures: tests/fixtures/ollama/chat_*.json are the real S1.4 probe replies (Ollama 0.40.11,
results/stage1/probe_native_tools_20261026T163037Z.json). nvidia-smi strings marked "S0" are
the real S0 reading; the after-load readings are constructed for the scenarios and say so.
"""
from __future__ import annotations

import json
import sys
import types
import urllib.error
from pathlib import Path

import pytest

from localbot.config import ProviderConfig, get_profile
from localbot.provider.ollama import (
    GIB, OllamaProvider, ProviderError, ProviderTimeout, disk_size, gpu_fit, lab_chat_body,
    thinking_size, truncation_check,
)
from localbot.telemetry import VramReading, parse_nvidia_smi

REPO = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "ollama"


def _lab_ollama_chat() -> types.ModuleType:
    mod = types.ModuleType("lab_ollama_chat_s4")
    sys.modules[mod.__name__] = mod
    src = (REPO / "lab_ref" / "lib" / "ollama_chat.py").read_text(encoding="utf-8")
    exec(compile(src, "lab_ref/lib/ollama_chat.py", "exec"), mod.__dict__)
    return mod


lab = _lab_ollama_chat()


class Recorder:
    """A transport that returns scripted replies (or raises) and records every request."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests: list[tuple[str, dict | None, float]] = []

    def __call__(self, url, body, timeout):
        self.requests.append((url, json.loads(json.dumps(body)) if body is not None else None, timeout))
        r = self.replies.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r


def fixture(name: str) -> dict:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


# --- request body = the lab's ------------------------------------------------------------


@pytest.mark.parametrize("tools", [None, [], get_profile("parity_lab6").tools.schemas,
                                   get_profile("core4").tools.schemas])
def test_body_identical_to_lab(monkeypatch, tools):
    captured = {}

    def fake_ollama_json(url, body, timeout=180):
        captured.update(url=url, body=json.loads(json.dumps(body)), timeout=timeout)
        return {"message": {"role": "assistant", "content": ""}}

    monkeypatch.setattr(lab, "ollama_json", fake_ollama_json)
    tl = list(tools) if tools is not None else None
    lab.chat("m:1", MSGS, tools=tl, num_ctx=8192, timeout=180)
    rec = Recorder({"message": {"role": "assistant", "content": ""}})
    OllamaProvider(get_profile("parity_lab6").provider, transport=rec).chat("m:1", MSGS, tl, timeout=180)
    url, body, timeout = rec.requests[0]
    assert body == captured["body"]
    assert list(body) == list(captured["body"])  # same key order too
    assert timeout == captured["timeout"] == 180
    assert url.endswith("/api/chat") and captured["url"].endswith("/api/chat")


def test_lab_values_in_both_profiles():
    for name in ("parity_lab6", "core4"):
        b = lab_chat_body(get_profile(name).provider, "m", MSGS, None)
        assert b["stream"] is False and b["keep_alive"] == "2m"
        assert b["options"] == {"num_ctx": 8192, "temperature": 0.2, "num_predict": 512}


def test_only_localhost_ollama_never_8090(monkeypatch):
    monkeypatch.delenv("LOCALBOT_OLLAMA_URL", raising=False)  # E-8: the default, whatever the shell has set
    p = OllamaProvider(ProviderConfig())
    assert p.chat_url == "http://127.0.0.1:11434/api/chat"
    rec = Recorder({"message": {}}, {"models": []}, {"models": []}, {})
    p = OllamaProvider(ProviderConfig(), transport=rec)
    p.chat("m", MSGS, None, 180), p.ps(), p.tags(), p.unload("m")
    assert all(":11434/" in url and ":8090" not in url for url, _, _ in rec.requests)
    monkeypatch.setenv("LOCALBOT_OLLAMA_URL", "http://10.0.0.5:11434")
    with pytest.raises(ValueError, match="localhost"):
        ProviderConfig()


def test_unload_body_is_the_labs(monkeypatch):
    captured = {}
    monkeypatch.setattr(lab, "ollama_json", lambda url, body, timeout=180: captured.update(body=body, t=timeout) or {})
    lab.unload("m:1")
    rec = Recorder({})
    OllamaProvider(ProviderConfig(), transport=rec).unload("m:1")
    assert rec.requests[0][1] == captured["body"] and rec.requests[0][2] == captured["t"] == 30
    assert rec.requests[0][0].endswith("/api/generate")


# --- recorded replies, per-call stats ----------------------------------------------------


@pytest.mark.parametrize("fx", ["chat_gemma4_obliterated_read.json", "chat_gemma4_12b_read.json"])
def test_recorded_reply_message_kept_exactly(fx):
    f = fixture(fx)
    rec = Recorder(f["response"])
    res = OllamaProvider(ProviderConfig(), transport=rec).chat(
        f["response"]["model"], f["request"]["messages"], f["request"]["tools"], timeout=180, first_turn=True)
    assert res.message == f["response"]["message"]  # thinking + tool_calls untouched
    assert res.message["tool_calls"][0]["function"]["name"] == "read"
    assert res.prompt_eval_count == 122 and res.eval_count == f["response"]["eval_count"]
    assert res.done_reason == "stop" and res.attempts == 1 and not res.infra_retry
    assert res.thinking_chars == len(f["response"]["message"]["thinking"]) > 0
    assert 0 < res.thinking_tokens_est <= res.eval_count
    assert not res.hit_num_predict
    assert not res.truncation_suspect, res.truncation_why  # 122 real vs ~estimate: fine
    s = res.stats()
    assert set(s) >= {"wall_s", "prompt_eval_count", "eval_count", "infra_retry", "truncation_suspect",
                      "thinking_tokens_est", "hit_num_predict"}


def test_thinking_eating_num_predict_is_flagged():
    long_think = "plan " * 600
    reply = {"message": {"role": "assistant", "content": "", "thinking": long_think},
             "done_reason": "length", "prompt_eval_count": 130, "eval_count": 512}
    res = OllamaProvider(ProviderConfig(), transport=Recorder(reply)).chat("m", MSGS, None, 180)
    assert res.hit_num_predict and res.thinking_tokens_est == 512  # all of the budget was thinking
    assert res.message["thinking"] == long_think


def test_l6_num_predict_per_model():
    """H1 L-6: num_predict can be set per model (the card will set it); others keep the profile's."""
    cfg = ProviderConfig(num_predict_by_model=(("big:1", 2048),))
    assert cfg.num_predict_for("big:1") == 2048 and cfg.num_predict_for("small:1") == 512
    assert lab_chat_body(cfg, "big:1", MSGS, None)["options"]["num_predict"] == 2048
    assert lab_chat_body(cfg, "small:1", MSGS, None)["options"]["num_predict"] == 512
    reply = {"message": {"role": "assistant", "content": "x"}, "done_reason": "stop", "eval_count": 600}
    res = OllamaProvider(cfg, transport=Recorder(dict(reply))).chat("big:1", MSGS, None, 180)
    assert not res.hit_num_predict  # 600 < 2048
    res = OllamaProvider(cfg, transport=Recorder(dict(reply))).chat("small:1", MSGS, None, 180)
    assert res.hit_num_predict  # 600 >= 512


def test_thinking_size_apportions_eval_count():
    assert thinking_size({"content": "x" * 30, "thinking": "y" * 70}, 100) == (70, 70)
    assert thinking_size({"content": "abc"}, 50) == (0, 0)
    assert thinking_size({"thinking": "z" * 40}, None) == (40, 10)


# --- retry / errors / timeout ------------------------------------------------------------


def _http(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://127.0.0.1:11434/api/chat", code, "boom", {}, None)


def test_one_retry_on_http_500_marked_infra_retry():
    ok = {"message": {"role": "assistant", "content": "hi"}, "eval_count": 3}
    rec = Recorder(_http(500), ok)
    res = OllamaProvider(ProviderConfig(), transport=rec).chat("m", MSGS, None, 180)
    assert res.attempts == 2 and res.infra_retry and res.message["content"] == "hi"
    assert len(rec.requests) == 2 and rec.requests[0][1] == rec.requests[1][1]


def test_two_500s_fail_after_one_retry():
    rec = Recorder(_http(500), _http(500))
    with pytest.raises(ProviderError, match="500"):
        OllamaProvider(ProviderConfig(), transport=rec).chat("m", MSGS, None, 180)
    assert len(rec.requests) == 2


def test_no_retry_on_other_errors():
    rec = Recorder(_http(404))
    with pytest.raises(ProviderError, match="404"):
        OllamaProvider(ProviderConfig(), transport=rec).chat("m", MSGS, None, 180)
    assert len(rec.requests) == 1


def test_timeout_is_not_retried():
    rec = Recorder(ProviderTimeout("timed out after 5s"))
    with pytest.raises(ProviderTimeout):
        OllamaProvider(ProviderConfig(), transport=rec).chat("m", MSGS, None, 5)
    assert len(rec.requests) == 1 and rec.requests[0][2] == 5


def _refused() -> urllib.error.URLError:
    return urllib.error.URLError(ConnectionRefusedError(10061, "No connection could be made"))


def test_p2_connection_error_retried_with_backoff():
    """H1 loop P2: a dropped / refused connection (Ollama restarting) is retried with a growing
    pause. Before H1 only an error mentioning "500" was retried, at once."""
    ok = {"message": {"role": "assistant", "content": "hi"}}
    slept: list[float] = []
    rec = Recorder(_refused(), ConnectionResetError(10054, "reset"), ok)
    res = OllamaProvider(ProviderConfig(), transport=rec, sleep=slept.append).chat("m", MSGS, None, 180)
    assert res.message["content"] == "hi" and res.attempts == 3 and res.infra_retry
    assert slept == [1.0, 2.0]


def test_p2_connection_error_gives_up_after_its_retries():
    rec = Recorder(_refused(), _refused(), _refused())
    with pytest.raises(ProviderError, match="No connection"):
        OllamaProvider(ProviderConfig(), transport=rec, sleep=lambda s: None).chat("m", MSGS, None, 180)
    assert len(rec.requests) == 3


def test_p2_parity_keeps_the_labs_retry_rule():
    rec = Recorder(_refused())
    with pytest.raises(ProviderError):
        OllamaProvider(get_profile("parity_lab6").provider, transport=rec, sleep=lambda s: None).chat("m", MSGS, None, 180)
    assert len(rec.requests) == 1


def test_p2_http_error_body_is_kept():
    """Ollama puts the cause in the body ({"error": "..."}); it used to be dropped."""
    import io

    def err() -> urllib.error.HTTPError:
        body = io.BytesIO(b'{"error":"llama runner process has terminated: exit status 2"}')
        return urllib.error.HTTPError("http://127.0.0.1:11434/api/chat", 500, "Internal Server Error", {}, body)

    with pytest.raises(ProviderError) as exc:
        OllamaProvider(ProviderConfig(), transport=Recorder(err(), err())).chat("m", MSGS, None, 180)
    msg = str(exc.value)
    assert "HTTP Error 500" in msg and "runner process has terminated" in msg  # runner.is_infra still matches


def test_p2_200_with_an_error_field_raises():
    """A 200 reply that carries {"error": ...} and no message used to end the task "done"."""
    rec = Recorder({"error": "model 'qwen-x' not found, try pulling it first"})
    with pytest.raises(ProviderError, match="not found"):
        OllamaProvider(ProviderConfig(), transport=rec).chat("m", MSGS, None, 180)


def test_p2_tok_s_from_eval_duration_not_wall_time():
    """H1 P2: tok/s = eval_count / eval_duration. eval_count / wall time counts the model load
    (real data: 5.2 vs ~95 tok/s)."""
    reply = {"message": {"role": "assistant", "content": "x"}, "eval_count": 95,
             "eval_duration": 1_000_000_000, "load_duration": 17_000_000_000, "prompt_eval_duration": 200_000_000}
    ticks = iter([0.0, 18.3])
    res = OllamaProvider(ProviderConfig(), transport=Recorder(reply), clock=lambda: next(ticks)).chat("m", MSGS, None, 180)
    assert res.wall_s == pytest.approx(18.3) and res.tok_s == 95.0
    s = res.stats()
    assert s["tok_s"] == 95.0 and s["eval_s"] == 1.0 and s["load_s"] == 17.0
    bare = OllamaProvider(ProviderConfig(), transport=Recorder({"message": {}, "eval_count": 9})).chat("m", MSGS, None, 180)
    assert bare.tok_s is None and bare.stats()["tok_s"] is None  # no duration: unknown, not guessed


# --- possible silent truncation ----------------------------------------------------------


def test_truncation_flags():
    assert truncation_check(9000, 8192, 8192, False)[0]          # estimate over num_ctx
    assert truncation_check(1200, 300, 8192, True)[0]           # first turn, far below estimate
    assert not truncation_check(1200, 300, 8192, False)[0]      # later turn: KV cache reuse, no flag
    assert not truncation_check(150, 20, 8192, True)[0]         # too small to judge
    assert not truncation_check(1200, 1100, 8192, True)[0]


def test_truncation_flag_reaches_stats():
    big = [{"role": "user", "content": "x" * 40_000}]
    res = OllamaProvider(ProviderConfig(), transport=Recorder(
        {"message": {"role": "assistant", "content": ""}, "prompt_eval_count": 8192})).chat("m", big, None, 180)
    assert res.truncation_suspect and "num_ctx" in res.truncation_why


# --- GPU fit: /api/ps + nvidia-smi cross-check -------------------------------------------

S0_SMI = "2059, 12227, 9\n"  # real S0 reading (2059 / 12227 MiB, 9 %), nothing loaded


def test_parse_nvidia_smi_recorded_strings():
    assert parse_nvidia_smi(S0_SMI) == [VramReading(2059, 12227, 9)]
    assert parse_nvidia_smi("2059, 12227, 9\n512, 8192, [N/A]\n") == [VramReading(2059, 12227, 9),
                                                                      VramReading(512, 8192, None)]
    assert parse_nvidia_smi("NVIDIA-SMI has failed because it couldn't communicate\n") == []
    assert parse_nvidia_smi("") == []


def test_gpu_fit_obliterated_consistent():
    f = fixture("chat_gemma4_obliterated_read.json")
    before = parse_nvidia_smi(S0_SMI)[0]
    after = VramReading(before.used_mib + 7700, 12227, 60)  # constructed: delta ~7.5 GB
    fit = gpu_fit(f["ps_row"]["name"], [f["ps_row"]], int(7.4e9), before, after)
    assert fit.lane == "gpu", fit.as_dict()
    assert fit.ps_says.startswith("100% GPU (7.45GB)")


def test_gpu_fit_gemma4_12b_ps_under_reports():
    """Recorded: /api/ps said gemma4:12b = 1.14 GB (size = size_vram), on disk 8.0 GB."""
    f = fixture("chat_gemma4_12b_read.json")
    assert f["ps_row"]["size"] == f["ps_row"]["size_vram"] == 1141322546
    before = parse_nvidia_smi(S0_SMI)[0]
    after = VramReading(before.used_mib + 1100, 12227, 40)  # constructed: delta matches ps, not disk
    fit = gpu_fit("gemma4:12b", [f["ps_row"]], int(8.0e9), before, after)
    assert fit.ps_says.startswith("100% GPU")  # the lab's check alone would say "fits"
    assert fit.lane == "unknown" and any("on-disk" in r for r in fit.reasons)


def test_gpu_fit_vram_delta_contradicts_ps():
    row = {"name": "m:1", "size": 8 * GIB, "size_vram": 8 * GIB}
    before, after = VramReading(2059, 12227), VramReading(2059 + 900, 12227)  # only +0.9 GB
    fit = gpu_fit("m:1", [row], 8 * GIB, before, after)
    assert fit.lane == "unknown" and any("nvidia-smi delta" in r for r in fit.reasons)


def test_gpu_fit_spill_and_cpu_only_and_missing():
    spill = {"name": "gpt-oss:20b", "size": 13 * GIB, "size_vram": 9 * GIB}
    assert gpu_fit("gpt-oss:20b", [spill], 13 * GIB, None, None).lane == "spill"
    cpu = {"name": "m:1", "size": 5 * GIB, "size_vram": 0}
    assert gpu_fit("m:1", [cpu], 5 * GIB, None, None).lane == "spill"  # lab said "100% GPU" here
    assert lab.is_100pct_gpu.__code__  # (lab check kept for reference; it misses size_vram == 0)
    assert gpu_fit("m:1", [], 5 * GIB, None, None).lane == "unknown"
    ok = {"name": "m:1", "size": 5 * GIB, "size_vram": 5 * GIB}
    assert gpu_fit("m:1", [ok], 5 * GIB, None, None).lane == "unknown"  # no nvidia-smi: no cross-check


def test_disk_size_from_tags():
    tags = [{"name": "gemma4:12b", "size": 8_000_000_000}, {"name": "other:1", "size": 1}]
    assert disk_size(tags, "gemma4:12b") == 8_000_000_000 and disk_size(tags, "nope") is None
