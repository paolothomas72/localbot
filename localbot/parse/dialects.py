"""Tool-name aliases and the plug-in point for model-specific text-call dialects.

The loop reads native ``tool_calls`` first. Only when a reply has none does it parse the
text, first with the vendored LocalHarness parser (fn_call.py: Qwen ``<function=>``,
Hermes ``<tool_call>{json}``, legacy), then with any *implemented* dialect below.

Dialects are scaffolded only (docs/STAGE1_SPEC.md §5.3). Each one can already *detect* its
marker, so a missed call is counted as ``text_call_unparsed`` and not as "model made no
call". ``extract`` stays empty until a roster model needs it; fill it with recorded
strings as tests.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

from localbot.parse.fn_call import (
    _FENCED_BLOCK_PATTERN, _FENCED_CALL_BODY_PATTERN, _QWEN_PARAM_PATTERN, _QWEN_TOOL_PATTERN,
    _RUNTIME_NATIVE_CALL_PATTERN, FnCallConverter, _drop_spurious_toolcall_closers,
    has_tool_call_attempt, strip_thinking_tags,
)
from localbot.parse.types import ToolCall

# Names models reach for from other harnesses -> Local Bot tool names.
ALIASES: dict[str, str] = {
    "read_file": "read",
    "write_file": "write",
    "patch": "edit",
    "terminal": "bash",
    "bash_exec": "bash",
    "search_files": "grep",
}


def canonical_name(name: str, allowed: Iterable[str]) -> str:
    """Map an alias onto an allowed tool. Never rewrites a name the profile already has.

    Parity keeps ``bash_exec`` (it is a lab tool there); core4 maps it to ``bash``. If the
    alias target is not allowed either, the original name is returned and the loop reports
    an unknown tool.
    """
    allowed = set(allowed)
    if name in allowed:
        return name
    target = ALIASES.get(name)
    return target if target in allowed else name


@dataclass(frozen=True)
class Dialect:
    """One model family's text tool-call syntax."""

    name: str
    markers: tuple[str, ...]
    implemented: bool = False
    _extract: Callable[[str], list[ToolCall]] = field(default=lambda text: [], repr=False)
    notes: str = ""

    def detect(self, text: str) -> bool:
        return bool(text) and any(m in text for m in self.markers)

    def extract(self, text: str) -> list[ToolCall]:
        return self._extract(text) if self.implemented else []


DIALECTS: tuple[Dialect, ...] = (
    Dialect(
        name="gemma",
        markers=("```tool_code",),
        notes=(
            "Stub. Probe 2026-10-26 (results/stage1/probe_native_tools_20261026T163037Z.json): "
            "hf.co/OBLITERATUS/Gemma-4-12B-OBLITERATED:Q4_K_M and gemma4:12b both returned "
            "native tool_calls on Ollama 0.40.11, so no text parser is needed for them now."
        ),
    ),
    Dialect(name="mistral", markers=("[TOOL_CALLS]",), notes="Stub."),
    Dialect(name="llama", markers=("<|python_tag|>",), notes="Stub."),
    Dialect(name="deepseek", markers=("<｜DSML｜", "<｜tool▁calls▁begin｜>", "<｜tool▁call▁begin｜>"), notes="Stub."),
)

_CONVERTER = FnCallConverter()


def _string_params(tools: Iterable[dict]) -> dict[str, set[str]]:
    """{tool name: names of its string-typed parameters}, from the advertised schemas."""
    out: dict[str, set[str]] = {}
    for t in tools:
        fn = t.get("function") or {}
        props = (fn.get("parameters") or {}).get("properties") or {}
        out[fn.get("name") or ""] = {p for p, spec in props.items() if (spec or {}).get("type") == "string"}
    return out


def _keep_raw_strings(text: str, calls: list[ToolCall], tools: Iterable[dict]) -> list[ToolCall]:
    """H1 L-5: the vendored Qwen-XML extractor JSON-decodes every ``<parameter=…>`` value, so
    ``<parameter=content>21</parameter>`` becomes the int 21 and a write of a number or a JSON
    body is refused by the arg gate. For string-typed parameters, put the raw text back.

    Only when the Qwen extractor is the one that produced ``calls``: it runs first, on the
    think-stripped text (or, if nothing parsed, on the closer-normalised text), and its matches
    line up one to one with the calls it returns. Other formats carry real JSON types; untouched.
    """
    strings = _string_params(tools)
    names = tuple(strings)
    cleaned = strip_thinking_tags(text)
    for form in (cleaned, _drop_spurious_toolcall_closers(cleaned)):
        matches = list(_QWEN_TOOL_PATTERN.finditer(form))
        if not matches:
            continue
        if [m.group(1) for m in matches] != [c.name for c in calls]:
            return calls  # some other extractor won: leave it
        out = []
        for m, c in zip(matches, calls):
            keep = strings.get(canonical_name(c.name, names), set())
            args = dict(c.arguments)
            for pm in _QWEN_PARAM_PATTERN.finditer(m.group(2)):
                if pm.group(1) in keep:
                    args[pm.group(1)] = pm.group(2).strip()  # the extractor's own strip, no decode
            out.append(ToolCall(name=c.name, arguments=args, id=c.id))
        return out
    return calls


def parse_text_calls(text: str, tools: Iterable[dict] | None = None) -> tuple[list[ToolCall], str | None]:
    """Calls found in reply text, and which parser found them (None if no calls).

    Order: vendored fn_call formats, then implemented dialects. With ``tools`` (the advertised
    schemas), Qwen-XML string parameters keep their raw text (L-5).
    """
    calls = _CONVERTER.extract_tool_calls(text or "")
    if calls:
        return (_keep_raw_strings(text or "", calls, tools) if tools else calls), "fn_call"
    for d in DIALECTS:
        if d.implemented and d.detect(text):
            calls = d.extract(text)
            if calls:
                return calls, d.name
    return [], None


def unparsed_dialect(text: str) -> str | None:
    """Name of a known-but-unimplemented dialect whose marker is in the text, if any."""
    for d in DIALECTS:
        if not d.implemented and d.detect(text):
            return d.name
    return None


EXAMPLE_HERMES = '<tool_call>{"name": "read", "arguments": {"path": "a.txt"}}</tool_call>'
EXAMPLE_QWEN = "<tool_call>\n<function=read>\n<parameter=path>a.txt</parameter>\n</function>\n</tool_call>"
EXAMPLE_LEGACY = '<tool_call>\n<name>read</name>\n<parameters>{"path": "a.txt"}</parameters>\n</tool_call>'
_LEGACY_OPEN = re.compile(r"<tool_call>\s*<name[>=]", re.IGNORECASE)


def call_example(text: str) -> str:
    """An example call written in the syntax the model's own failed reply used (H1 P2), so the
    nudge teaches its dialect instead of switching it to another: Qwen XML, legacy, else Hermes."""
    if "<function=" in (text or ""):
        return EXAMPLE_QWEN
    if _LEGACY_OPEN.search(text or ""):
        return EXAMPLE_LEGACY
    return EXAMPLE_HERMES


_JSON_NAME = re.compile(r'"name"\s*:\s*"([^"]*)"')


def call_attempt(text: str, allowed: Iterable[str]) -> bool:
    """``has_tool_call_attempt``, except that JSON shaped like ``{"name": …, "arguments": …}`` with
    no call syntax around it counts only when the name is a tool (or an alias of one) the profile
    has (H1 L-7): a JSON-only answer such as ``{"name": "Alice", "arguments": [...]}`` is an answer.
    The taught syntax (``<tool_call>``, ``<function=…>``), runtime-native markers and
    ```` ```tool_code ```` / ```` ```tool_call ```` fences count whatever the name."""
    if not has_tool_call_attempt(text):
        return False
    if "<tool_call" in text or ("<function=" in text and "<parameter=" in text):
        return True
    if _RUNTIME_NATIVE_CALL_PATTERN.search(text):
        return True
    for tag, body in _FENCED_BLOCK_PATTERN.findall(text):
        if tag.lower() != "json" and _FENCED_CALL_BODY_PATTERN.search(body):
            return True
    allowed = tuple(allowed)
    return any(canonical_name(n, allowed) in allowed for n in _JSON_NAME.findall(text))


# --- H1 L-1: which text calls may run ------------------------------------------------------------

_FENCE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)  # an unclosed fence runs to the end
_READ_NUMBER = re.compile(r"(?m)^\d+\t")  # read() output: "N\tline"


def strip_code_fences(text: str) -> str:
    """The text outside ``` code blocks. A call written inside one is an example, not a call."""
    return _FENCE.sub("", text or "")


def call_key(name: str, args: dict, allowed: Iterable[str]) -> str:
    """One call as a comparable key: the canonical tool name + its arguments (key order and JSON
    layout do not matter)."""
    return f"{canonical_name(name, allowed)}:{json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)}"


def seen_calls(text: str, allowed: Iterable[str], tools: Iterable[dict] | None = None) -> set[str]:
    """Keys of every call block in a tool result (file text, command output), so the loop never
    runs one of them when the model repeats it (L-1). The read tool's line numbers break a block
    that spans lines, and the model drops them when it quotes, so both forms are parsed. Pass the
    same ``tools`` the reply is parsed with, so both sides get the same argument types (L-5)."""
    allowed = tuple(allowed)
    keys: set[str] = set()
    for form in {text or "", _READ_NUMBER.sub("", text or "")}:
        for c in parse_text_calls(form, tools)[0]:
            keys.add(call_key(c.name, dict(c.arguments), allowed))
    return keys
