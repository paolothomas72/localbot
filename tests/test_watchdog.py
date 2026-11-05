"""watchdog.py (S4, spec §5.9): step cap, wall clocks, stuck, repetition, provider timeout.
The hard rule — no trial past its deadline + hard_overrun — is tested with a provider that
never answers in time (real clock, small deadlines)."""
from __future__ import annotations

import dataclasses
import time

import pytest

from localbot.config import LoopConfig, WatchdogConfig, get_profile
from localbot.loop import run_task
from localbot.provider.ollama import ProviderTimeout
from localbot.watchdog import END_REASONS, TaskWatch, TrialClock, call_with_deadline, unique_line_ratio
from tests.fakes import SlowProvider


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_reason_codes_are_the_spec_list():
    assert END_REASONS == ("done", "max_steps", "task_timeout", "trial_deadline", "stuck",
                           "repetition", "provider_error", "parse_fail", "undo_unsafe", "git_changed",
                           "work_quota", "exec_failed")


def test_time_left_picks_the_binding_clock():
    clk = FakeClock()
    trial = TrialClock(900, clk)
    w = TaskWatch(WatchdogConfig(task_wall_clock_s=180), LoopConfig(), trial, clk)
    assert w.time_left() == (180, "task_timeout")
    clk.t += 800  # trial has 100 s left, task 180 - 800 < 0
    assert w.time_up() == "task_timeout"
    w2 = TaskWatch(WatchdogConfig(task_wall_clock_s=180), LoopConfig(), trial, clk)
    assert w2.time_left() == (100, "trial_deadline") and w2.time_up() is None
    clk.t += 100
    assert w2.time_up() == "trial_deadline"


def test_call_timeout_is_lab_value_or_less():
    clk = FakeClock()
    w = TaskWatch(WatchdogConfig(task_wall_clock_s=None), LoopConfig(), TrialClock(900, clk), clk)
    assert w.call_timeout() == 180  # the lab's per-call timeout
    clk.t += 850
    assert w.call_timeout() == 50


def test_stuck_nudge_at_3_end_at_5_and_off_in_parity():
    w = TaskWatch(WatchdogConfig(), LoopConfig(), TrialClock(900))
    seq = [w.note_call("read", {"path": "a"}) for _ in range(5)]
    assert seq == [None, None, "nudge", None, "stuck"]
    assert w.note_call("read", {"path": "b"}) is None  # different args, different signature
    p = get_profile("parity_lab6").watchdog
    wp = TaskWatch(p, LoopConfig(), TrialClock(900))
    assert [wp.note_call("read", {"path": "a"}) for _ in range(6)] == [None] * 6


def test_l3_same_call_with_a_new_result_is_progress_not_stuck():
    """H1 L-3: the stuck count keys on (call, result). A call whose result keeps changing (tests
    after each edit) is progress; the same call with the same result is stuck, as before."""
    w = TaskWatch(WatchdogConfig(), LoopConfig(), TrialClock(900))
    assert [w.note_call("bash", {"command": "pytest"}, f"{i} passed") for i in range(8)] == [None] * 8
    w2 = TaskWatch(WatchdogConfig(), LoopConfig(), TrialClock(900))
    assert [w2.note_call("bash", {"command": "pytest"}, "1 failed") for _ in range(5)] == \
        [None, None, "nudge", None, "stuck"]


def test_repetition():
    w = TaskWatch(WatchdogConfig(), LoopConfig(), TrialClock(900))
    assert w.note_text("same\n" * 20) == "repetition"
    assert w.note_text("\n".join(f"line {i}" for i in range(20))) is None
    assert w.note_text("same\n" * 5) is None  # too short to judge
    assert unique_line_ratio("a\na\nb\nb\n") == (0.5, 4)
    assert TaskWatch(get_profile("parity_lab6").watchdog, LoopConfig(), TrialClock(900)).note_text("same\n" * 20) is None


def test_call_with_deadline():
    assert call_with_deadline(lambda: 42, 1.0) == 42
    with pytest.raises(ValueError):
        call_with_deadline(lambda: (_ for _ in ()).throw(ValueError("x")), 1.0)
    t0 = time.monotonic()
    with pytest.raises(ProviderTimeout):
        call_with_deadline(lambda: time.sleep(5), 0.3)
    assert time.monotonic() - t0 < 1.0


def _fast(profile_name: str, trial_s: float, task_s: float | None, overrun: float):
    base = get_profile(profile_name)
    return dataclasses.replace(
        base,
        loop=dataclasses.replace(base.loop, trial_deadline_s=trial_s),
        watchdog=dataclasses.replace(base.watchdog, hard_overrun_s=overrun, task_wall_clock_s=task_s),
    )


@pytest.mark.parametrize("profile_name", ["parity_lab6", "core4"])
def test_slow_provider_never_pushes_trial_past_deadline(disk, profile_name):
    """Trial deadline 1.0 s, overrun 0.5 s, provider sleeps 30 s per call. Three tasks in one
    trial: the whole trial must end within deadline + overrun (+ scheduling slack)."""
    prof = _fast(profile_name, trial_s=1.0, task_s=None, overrun=0.5)
    ws = disk.ws({"a.txt": "A"})
    trial = TrialClock(prof.loop.trial_deadline_s)
    t0 = time.monotonic()
    reasons = [run_task(prof, SlowProvider(30), "slow:1", "do it", ws, trial=trial).reason for _ in range(3)]
    took = time.monotonic() - t0
    assert reasons == ["trial_deadline"] * 3
    assert took <= 1.0 + 0.5 + 0.4, took


def test_task_wall_clock_core4(disk):
    prof = _fast("core4", trial_s=100, task_s=0.5, overrun=0.3)
    ws = disk.ws({"a.txt": "A"})
    t0 = time.monotonic()
    res = run_task(prof, SlowProvider(30), "slow:1", "do it", ws)
    assert res.reason == "task_timeout" and "no reply" in res.error
    assert time.monotonic() - t0 <= 0.5 + 0.3 + 0.4


def test_parity_has_no_task_wall_clock_like_lab():
    assert get_profile("parity_lab6").watchdog.task_wall_clock_s is None
    assert get_profile("core4").watchdog.task_wall_clock_s == 180
    assert get_profile("parity_lab6").loop.task_timeout_s == 180  # per provider call, as the lab
