"""Output caps, eviction, compaction (docs/STAGE1_SPEC.md §5.8, step S4).

All of it is driven by ``ContextConfig``; ``parity_lab6`` turns every part off (the lab loop
had no caps, no eviction, no compaction), so there this module does nothing.

- ``cap_output``: a tool result over the cap is cut and says ``[truncated N chars]``.
- ``evict_old_results``: tool results over N characters, outside the first K and last M
  messages, are replaced by a short placeholder (the bot can re-run the tool).
- ``compact``: when the estimated prompt passes ``trigger`` × num_ctx, drop the oldest middle
  messages until it is under ``target`` × num_ctx (LocalHarness-style hysteresis), keeping the
  first K messages (system + ticket) and never starting the kept tail on an orphan tool result.
  A note is inserted in place of what was dropped. S1-X X3b: in a task with several user messages,
  the current one (the loop names it) is kept too; in a one-message task it is the ticket.
- ``RereadGate``: after a compaction, a write/edit to an existing file is refused until that
  file — and ``TASK_MANIFEST.json``, if the workspace has one — was read again. So "re-read
  before writing; no overwrite" is a harness property, not a hope about the model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from localbot.config import ContextConfig

CHARS_PER_TOKEN = 4.0  # rough; the trigger and target are fractions of num_ctx, not exact
MANIFEST = "TASK_MANIFEST.json"
COMPACT_NOTE = (
    "[harness: context compacted, {n} earlier messages removed to fit num_ctx. "
    "Before any write: read TASK_MANIFEST.json (if it exists) and the target file again.]"
)
EVICTED = "[harness: evicted {n} chars of old {name} output. Run the tool again if you need it.]"


def cap_output(text: str, cap: int | None) -> str:
    if cap is None or len(text) <= cap:
        return text
    return f"{text[:cap]}\n[truncated {len(text) - cap} chars]"


def estimate_tokens(messages: list[dict]) -> int:
    return int(len(json.dumps(messages, ensure_ascii=False)) / CHARS_PER_TOKEN)


def evict_old_results(messages: list[dict], over: int | None, keep_first: int, keep_last: int) -> int:
    """In place. Returns how many tool results were evicted."""
    if over is None:
        return 0
    n = 0
    last_start = max(keep_first, len(messages) - keep_last)
    for i in range(keep_first, last_start):
        m = messages[i]
        content = m.get("content")
        if m.get("role") == "tool" and isinstance(content, str) and len(content) > over and not content.startswith("[harness: evicted"):
            messages[i] = {**m, "content": EVICTED.format(n=len(content), name=m.get("name") or "tool")}
            n += 1
    return n


def compact(messages: list[dict], num_ctx: int, trigger: float, target: float,
            keep_first: int, reserve: int = 0, keep: dict | None = None) -> tuple[list[dict], int]:
    """Return (messages, dropped). ``dropped == 0`` means nothing was done.

    ``reserve`` (H1 P2): tokens of num_ctx the messages cannot use — the tool schemas sent with
    every request and the reply's num_predict. Trigger and target apply to messages + reserve.

    ``keep`` (S1-X X3b): the task's current user message (the loop passes it), never dropped, like the
    first ``keep_first``. The rest goes oldest first: what came before it, then its own oldest steps.
    When it is in the head (a one-message task: it is message 1) or not in the list, nothing changes.
    When the head and it alone are over the budget, every other message goes and nothing more is done,
    as for a head alone over the budget."""
    if estimate_tokens(messages) + reserve <= trigger * num_ctx:
        return messages, 0
    head = messages[:keep_first]
    tail = list(messages[keep_first:])
    budget = target * num_ctx - reserve
    pin = next((i for i, m in enumerate(tail) if m is keep), None) if keep is not None else None
    before, kept, after = (tail, [], []) if pin is None else (tail[:pin], [tail[pin]], tail[pin + 1:])
    dropped = 0
    for part in (before, after):  # without ``keep``: one part, today's two loops
        n0 = dropped
        while part and estimate_tokens(head + [{"role": "user", "content": COMPACT_NOTE}] + before + kept + after) > budget:
            part.pop(0)
            dropped += 1
        # never keep a result without its call (the job's own steps: only once one of them went)
        while part and (part is before or dropped > n0) and part[0].get("role") == "tool":
            part.pop(0)
            dropped += 1
    if not dropped:
        return messages, 0
    note = {"role": "user", "content": COMPACT_NOTE.format(n=dropped)}
    return head + [note] + before + kept + after, dropped


def _norm(path: str) -> str:
    """H1 P2: drop leading "./" prefixes only. ``lstrip("./")`` stripped characters: ".env" -> "env"."""
    p = (path or "").replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p.casefold()


@dataclass
class RereadGate:
    """After a compaction: existing files must be read again before they are written."""

    active: bool = False
    read_since: set[str] = field(default_factory=set)
    compactions: int = 0

    def compacted(self) -> None:
        self.active = True
        self.read_since.clear()
        self.compactions += 1

    def note_read(self, path: str) -> None:
        self.read_since.add(_norm(path))

    def check_write(self, path: str, target_exists: bool, manifest_exists: bool) -> str | None:
        if not self.active:
            return None
        need = []
        if manifest_exists and _norm(MANIFEST) not in self.read_since:
            need.append(MANIFEST)
        if target_exists and _norm(path) not in self.read_since and _norm(path) != _norm(MANIFEST):
            need.append(path)
        if not need:
            return None
        return f"harness: context was compacted. Read {' and '.join(need)} again before writing {path}."


class ContextManager:
    """One per task."""

    def __init__(self, cfg: ContextConfig, num_ctx: int, tools: list[dict] | None = None,
                 num_predict: int = 0) -> None:
        self.cfg = cfg
        self.num_ctx = num_ctx
        # H1 P2: what every request spends besides the messages (schemas + the reply's budget)
        self.reserve = (estimate_tokens(list(tools)) if tools else 0) + num_predict
        self.gate = RereadGate()
        self.evicted = 0
        self.compactions: list[int] = []  # messages dropped per compaction

    def cap(self, text: str) -> str:
        return cap_output(text, self.cfg.tool_output_cap_chars)

    def after_turn(self, messages: list[dict], keep: dict | None = None) -> list[dict]:
        """Evict, then compact if needed. Returns the (possibly new) message list. ``keep`` (X3b): the
        task's current user message, never compacted away."""
        self.evicted += evict_old_results(messages, self.cfg.evict_tool_results_over_chars,
                                          self.cfg.keep_first_messages, self.cfg.keep_last_messages)
        if not self.cfg.compaction:
            return messages
        out, dropped = compact(messages, self.num_ctx, self.cfg.compaction_trigger,
                               self.cfg.compaction_target, self.cfg.keep_first_messages, self.reserve, keep=keep)
        if dropped:
            self.compactions.append(dropped)
            self.gate.compacted()
        return out
