"""Step cap, wall clocks, stuck, repetition, provider timeout (docs/STAGE1_SPEC.md §5.9, S4).

Every task ends with exactly one reason code (``END_REASONS``). Nothing ends silently.

Time: ``TrialClock`` holds the whole-trial deadline; ``TaskWatch`` adds the optional per-task
wall clock. Each provider call gets the time that is left, and ``call_with_deadline`` runs it
in a worker thread and stops waiting at that time + ``hard_overrun_s``, so a hung or
slow-dripping provider can never push a trial past its deadline + 10 s (G7) — the socket
timeout alone can't promise that (it is per read, not per call).
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar

from localbot.config import LoopConfig, WatchdogConfig
from localbot.provider.ollama import ProviderTimeout

END_REASONS = ("done", "max_steps", "task_timeout", "trial_deadline", "stuck", "repetition",
               "provider_error", "parse_fail", "undo_unsafe", "git_changed",
               "work_quota", "exec_failed")  # H1-I item 7: the sandbox stopped a step (undone first)

STUCK_NUDGE = ("harness: you made the same {name} call {n} times and got the same result each time. "
               "Do something different, or finish.")

T = TypeVar("T")
Clock = Callable[[], float]


class TrialClock:
    """The whole trial's deadline (lab: 900 s for the 12-task trial)."""

    def __init__(self, deadline_s: float, clock: Clock = time.monotonic) -> None:
        self.clock = clock
        self.start = clock()
        self.deadline_s = deadline_s

    def elapsed(self) -> float:
        return self.clock() - self.start

    def remaining(self) -> float:
        return self.deadline_s - self.elapsed()


def call_signature(name: str, args: dict) -> str:
    try:
        return f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
    except (TypeError, ValueError):
        return f"{name}:{args!r}"


def unique_line_ratio(text: str) -> tuple[float, int]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return 1.0, 0
    return len(set(lines)) / len(lines), len(lines)


@dataclass
class TaskWatch:
    wcfg: WatchdogConfig
    lcfg: LoopConfig
    trial: TrialClock
    clock: Clock = time.monotonic
    start: float = field(init=False)
    calls: dict[str, int] = field(default_factory=dict)
    nudges: int = 0
    min_repetition_lines: int = 8

    def __post_init__(self) -> None:
        self.start = self.clock()

    # -- time ------------------------------------------------------------------------------

    def task_remaining(self) -> float | None:
        if self.wcfg.task_wall_clock_s is None:
            return None
        return self.wcfg.task_wall_clock_s - (self.clock() - self.start)

    def time_left(self) -> tuple[float, str]:
        """(seconds left, reason code if it runs out)."""
        trial = self.trial.remaining()
        task = self.task_remaining()
        if task is not None and task < trial:
            return task, "task_timeout"
        return trial, "trial_deadline"

    def time_up(self) -> str | None:
        left, why = self.time_left()
        return why if left <= 0 else None

    def call_timeout(self) -> float:
        """Socket timeout for the next provider call: the lab's per-call value, or less."""
        left, _ = self.time_left()
        return max(0.1, min(float(self.lcfg.task_timeout_s), left))

    # -- behaviour -------------------------------------------------------------------------

    def note_call(self, name: str, args: dict, result: str | None = None) -> str | None:
        """None, "nudge" (at stuck_nudge_after) or "stuck" (at stuck_end_after).

        H1 L-3: counted per (call, result). The same call with a new result (tests re-run after
        each edit) is progress; only the same call getting the same result counts toward stuck."""
        if self.wcfg.stuck_nudge_after is None and self.wcfg.stuck_end_after is None:
            return None
        sig = call_signature(name, args)
        if result is not None:
            sig += "|" + hashlib.sha256(result.encode("utf-8", "surrogatepass")).hexdigest()
        n = self.calls[sig] = self.calls.get(sig, 0) + 1
        if self.wcfg.stuck_end_after is not None and n >= self.wcfg.stuck_end_after:
            return "stuck"
        if self.wcfg.stuck_nudge_after is not None and n == self.wcfg.stuck_nudge_after:
            self.nudges += 1
            return "nudge"
        return None

    def note_text(self, text: str) -> str | None:
        ratio_min = self.wcfg.repetition_min_unique_line_ratio
        if ratio_min is None:
            return None
        ratio, n = unique_line_ratio(text)
        if n >= self.min_repetition_lines and ratio < ratio_min:
            return "repetition"
        return None


def call_with_deadline(fn: Callable[[], T], wait_s: float) -> T:
    """Run ``fn`` in a daemon thread; give up waiting after ``wait_s`` seconds.

    A call that is given up on keeps running in the background until its own socket timeout;
    its result is discarded. Raises ProviderTimeout when the wait runs out.
    """
    box: dict = {}

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # handed to the caller
            box["error"] = exc

    t = threading.Thread(target=run, name="localbot-provider", daemon=True)
    t.start()
    t.join(max(0.0, wait_s))
    if t.is_alive():
        raise ProviderTimeout(f"no reply within {wait_s:.1f}s")
    if "error" in box:
        raise box["error"]
    return box["value"]
