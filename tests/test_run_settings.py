"""S1-X X3: run settings as runner flags, several user messages in one task, pack deadlines, and the
hooks the long-task pack (X4) plugs into. Scripted providers only: no model, no GPU, no container.

- Flags ``--num-ctx`` / ``--max-turns`` / ``--num-predict`` (defaults = the profile's: 8192 / 8 / 512)
  and ``--tasks``. No flag = the very same profile (the first request stays byte-identical, G3 parity).
  A flag reaches the request body, the compaction budget, the truncation check and the result file.
- ``turns: [...]`` in a local-pack task: the messages go in one after another, each with its own
  ``max_turns``; a message that runs out of turns lets the next one go; a guard end ends the task.
- ``deadline_s`` / ``trial_deadline_s`` from a local pack; lab packs keep 180 / 900.
- ``generator`` / ``long`` blocks call ``localbot.exam.long_seed.build`` / ``long_judge.verdict``
  (OpenCode's, X4), imported only when a task has the field. Tested with fake modules.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import types
from pathlib import Path

import pytest

from localbot import loop
from localbot.config import ContextConfig, get_profile, with_run_settings
from localbot.context import ContextManager, estimate_tokens
from localbot.exam import runner
from localbot.exam.runner import load_pack, run_pack
from localbot.provider import ollama
from localbot.telemetry import TrialSampler, VramReading
from localbot.watchdog import TrialClock
from tests.fakes import FakeShell, ReactiveTransport, RoutedTransport, ScriptedTransport, call, provider, reply

P6, C4 = get_profile("parity_lab6"), get_profile("core4")
LAB_TASK = "e_read"  # the first work task: one scripted "done" reply ends it


def _sampler():
    return TrialSampler(0.05, cpu=lambda: 1.0, ram=lambda: 1, vram=lambda: VramReading(1, 2, 0))


def _fixed(disk) -> dict:
    return dict(vram_reader=lambda: VramReading(1, 2, 0), sampler_factory=_sampler,
                results_dir=disk.base / "results", trials_root=disk.base / "_trials", log=lambda s: None)


def _pack(disk, profile, chat, tasks, pack="work", **kw):
    rt = RoutedTransport(chat)
    shell = FakeShell({}) if profile.tools.shell_mode == "container" else None
    res, dest = run_pack(pack, profile, "fake:1", provider(rt, profile.provider), n=1, tasks=tasks, shell=shell,
                         **_fixed(disk), **kw)
    return res, dest


def _main(disk, monkeypatch, argv, script):
    """``runner.main`` end to end: the provider it builds talks to a scripted model, its run_pack writes into
    the test's folders. Returns (transport, [result dicts], [ContextManager], [truncation_check num_ctx])."""
    chat = ScriptedTransport(script)
    rt = RoutedTransport(chat)
    monkeypatch.setattr(ollama, "OllamaProvider", lambda cfg: provider(rt, cfg))
    real_pack, results = runner.run_pack, []

    def pack(name, prof, model, prov, **kw):
        shell = FakeShell({}) if prof.tools.shell_mode == "container" else None
        res, dest = real_pack(name, prof, model, prov, **{**kw, **_fixed(disk), "shell": shell})
        results.append(res)
        return res, dest

    monkeypatch.setattr(runner, "run_pack", pack)
    managers: list[ContextManager] = []

    class Recording(ContextManager):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            managers.append(self)

    monkeypatch.setattr(loop, "ContextManager", Recording)
    checked: list[int] = []
    real_check = ollama.truncation_check
    monkeypatch.setattr(ollama, "truncation_check", lambda est, pe, n, first: (checked.append(n), real_check(est, pe, n, first))[1])
    assert runner.main(argv) == 0
    return chat, results, managers, checked


# --- X3.1 flags ---------------------------------------------------------------------------------


def test_x3_no_flag_is_the_very_same_profile_and_the_file_says_default(disk, monkeypatch):
    """Test 1: no flag = the profile object itself (no ``dataclasses.replace``), so the first request is the
    one ``tests/test_first_request.py`` compares with the lab byte for byte. The file says ``default``."""
    assert with_run_settings(P6) is P6 and with_run_settings(C4, num_ctx=None, max_turns=None, num_predict=None) is C4
    chat, results, managers, checked = _main(disk, monkeypatch, ["--pack", "work", "--tasks", LAB_TASK, "--profile",
                                                                 "parity_lab6", "--n", "1", "m:1"], [reply("done")])
    opts = chat.bodies[0]["options"]
    assert (opts["num_ctx"], opts["num_predict"]) == (8192, 512)
    env = results[0]["run_env"]
    assert env["run_settings"] == {"num_ctx": 8192, "max_turns": 8, "num_predict": 512,
                                   "num_predict_by_model": [], "source": "default"}
    assert env["context"] == {"tool_output_cap_chars": None, "evict_over_chars": None,
                              "compaction_trigger": None, "compaction_target": None}  # parity: all off
    assert [m.num_ctx for m in managers] == [8192] and checked == [8192]


@pytest.mark.parametrize("flags,want", [
    (["--num-ctx", "32768"], (32768, 8, 512)),
    (["--max-turns", "24"], (8192, 24, 512)),
    (["--num-predict", "1024"], (8192, 8, 1024)),
    (["--num-ctx", "16384", "--max-turns", "64", "--num-predict", "2048"], (16384, 64, 2048)),
], ids=["num_ctx", "max_turns", "num_predict", "all_three"])
def test_x3_a_flag_reaches_the_request_the_budget_the_check_and_the_file(disk, monkeypatch, flags, want):
    """Test 2: each flag alone, then all three, through ``runner.main``: the request's ``options``, the
    ContextManager's ``num_ctx`` and reserve (schemas + the reply budget), the truncation check and the
    file's ``run_settings`` all carry the same numbers."""
    num_ctx, max_turns, num_predict = want
    chat, results, managers, checked = _main(disk, monkeypatch, ["--pack", "work", "--tasks", LAB_TASK, "--profile",
                                                                 "core4", "--n", "1", *flags, "m:1"], [reply("done")])
    opts = chat.bodies[0]["options"]
    assert (opts["num_ctx"], opts["num_predict"]) == (num_ctx, num_predict)
    (cm,) = managers
    schemas = ContextManager(C4.context, 1, tools=list(C4.tools.schemas)).reserve
    assert cm.num_ctx == num_ctx and cm.reserve == schemas + num_predict
    assert checked == [num_ctx]
    env = results[0]["run_env"]
    assert env["run_settings"] == {"num_ctx": num_ctx, "max_turns": max_turns, "num_predict": num_predict,
                                   "num_predict_by_model": [], "source": "cli"}
    assert env["context"] == {"tool_output_cap_chars": 4000, "evict_over_chars": 2500,
                              "compaction_trigger": 0.8, "compaction_target": 0.6}


def test_x3_max_turns_is_the_loop_limit(disk):
    """``--max-turns 3``: the loop stops after 3 replies (``max_steps``), not 8."""
    ws = disk.ws({**{f"f{i}.txt": f"{i}\n" for i in range(5)}})
    script = [reply(calls=[call("read", path=f"f{i}.txt")]) for i in range(5)]
    tr = ScriptedTransport(script)
    prof = with_run_settings(P6, max_turns=3)
    res = loop.run_task(prof, provider(tr, prof.provider), "fake:1", "read them", ws)
    assert res.reason == "max_steps" and len(res.turns) == 3 and len(tr.bodies) == 3


def test_x3_num_predict_flag_clears_the_per_model_table():
    prof = dataclasses.replace(C4, provider=dataclasses.replace(C4.provider, num_predict_by_model=(("m:1", 4096),)))
    assert prof.provider.num_predict_for("m:1") == 4096
    new = with_run_settings(prof, num_predict=1024)
    assert new.provider.num_predict_by_model == () and new.provider.num_predict_for("m:1") == 1024


def _filler(n: int) -> dict[str, str]:
    """``n`` files of ~2,200 characters each (with the read tool's line numbers, still under the eviction
    size of 2,500), so every read stays in the context and only compaction can drop it."""
    return {f"d{i}.txt": "\n".join(f"line {i}-{j} " + "x" * 50 for j in range(36)) + "\n" for i in range(n)}


@pytest.mark.parametrize("num_ctx,compacts", [(8192, True), (32768, False)])
def test_x3_compaction_starts_at_the_flag_s_num_ctx(disk, num_ctx, compacts):
    """Test 3: 14 reads of ~2,200 characters: the conversation (plus the schemas and the reply budget) ends
    over 0.8 × 8192 and under 0.8 × 32768. With ``--num-ctx 32768`` nothing is compacted; at the default
    it is."""
    files = _filler(14)
    ws = disk.ws(files)
    script = [reply(calls=[call("read", path=name)]) for name in files] + [reply("done")]
    prof = with_run_settings(C4, num_ctx=num_ctx, max_turns=16)
    res = loop.run_task(prof, provider(ScriptedTransport(script), prof.provider), "fake:1", "read them all", ws)
    assert res.reason == "done" and len(res.tool_calls) == 14
    assert bool(res.compactions) is compacts, res.compactions
    if not compacts:  # the whole conversation is still there: its size is between the two triggers
        reserve = ContextManager(prof.context, num_ctx, tools=list(prof.tools.schemas),
                                 num_predict=prof.provider.num_predict).reserve
        assert 0.8 * 8192 < estimate_tokens(res.messages) + reserve < 0.8 * 32768


@pytest.mark.parametrize("argv,says", [
    (["--num-ctx", "0"], "num_ctx must be a whole number above 0"),
    (["--max-turns", "-3"], "max_turns must be a whole number above 0"),
    (["--num-predict", "-1"], "num_predict must be a whole number above 0"),
    (["--num-ctx", "lots"], "invalid int value"),
    (["--max-turns", "2.5"], "invalid int value"),
    (["--num-ctx", "4096", "--num-predict", "4096"], "num_predict (4096) must be smaller than num_ctx (4096)"),
    (["--num-predict", "8192"], "num_predict (8192) must be smaller than num_ctx (8192)"),
    (["--num-ctx", "256"], "num_predict (512) must be smaller than num_ctx (256)"),
], ids=["ctx_zero", "turns_negative", "predict_negative", "ctx_word", "turns_float", "predict_eq_ctx",
        "predict_eq_default_ctx", "ctx_under_default_predict"])
def test_x3_bad_values_are_refused_with_a_plain_message(monkeypatch, capsys, argv, says):
    """Test 4: 0, negative, non-numbers and ``--num-predict`` >= ``--num-ctx`` are refused before anything
    runs (argparse exit 2 with the message)."""
    monkeypatch.setattr(runner, "run_pack", lambda *a, **k: pytest.fail("ran with a bad setting"))
    with pytest.raises(SystemExit) as e:
        runner.main(["--pack", "work", "--profile", "core4", *argv, "m:1"])
    assert e.value.code == 2 and says in capsys.readouterr().err


# --- X3.1 --tasks ---------------------------------------------------------------------------------


def _ran_tasks(monkeypatch, argv) -> list[tuple[str, list[str]]]:
    ran: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(runner, "run_pack", lambda pack, *a, tasks=None, **k: ran.append((pack, [t["id"] for t in tasks])))
    assert runner.main(argv) == 0
    return ran


def test_x3_tasks_by_id_and_by_pattern(monkeypatch):
    """Test 10: an id runs that task only; a pattern runs every match, in pack order; with ``--pack all`` a
    pack none of the patterns names is skipped."""
    assert _ran_tasks(monkeypatch, ["--pack", "work", "--tasks", "e_calc", "m:1"]) == [("work", ["e_calc"])]
    ran = _ran_tasks(monkeypatch, ["--pack", "injection", "--profile", "core4", "--tasks", "i_quoted_*", "m:1"])
    assert ran == [("injection", [t["id"] for t in load_pack("injection") if t["id"].startswith("i_quoted_")])]
    assert len(ran[0][1]) == 5
    ran = _ran_tasks(monkeypatch, ["--pack", "all", "--tasks", "e_read,s_no_secret", "m:1"])
    assert ran == [("work", ["e_read"]), ("trust", ["s_no_secret"])]


def test_x3_an_unknown_task_pattern_stops_with_the_list_of_ids(monkeypatch, capsys):
    monkeypatch.setattr(runner, "run_pack", lambda *a, **k: pytest.fail("ran with an unknown task"))
    with pytest.raises(SystemExit) as e:
        runner.main(["--pack", "work", "--tasks", "e_read,l_log_*", "m:1"])
    err = capsys.readouterr().err
    assert e.value.code == 2 and "l_log_*" in err and all(t["id"] in err for t in load_pack("work"))


# --- X3.2 several user messages in one task ---------------------------------------------------------

TURNS = ["Rule: every answer goes in out.txt.", "Job 2: read a.txt.", "Job 3: read b.txt."]


def _turns_run(disk, script, profile=P6, turns=TURNS, seed=None):
    ws = disk.ws(seed or {"a.txt": "A\n", "b.txt": "B\n", "keep.txt": "keep\n"})
    tr = ScriptedTransport(script)
    res = loop.run_task(profile, provider(tr, profile.provider), "fake:1", turns, ws)
    return res, tr


def _users(messages) -> list[str]:
    return [m["content"] for m in messages if m.get("role") == "user"]


def test_x3_three_messages_each_done(disk):
    """Test 5: each message runs until the model answers with no call; then the next goes in. All three
    user messages are in the conversation, in order; ``said`` has every reply; three segments."""
    script = [reply("ok, noted."), reply(calls=[call("read", path="a.txt")]), reply("a.txt says A."),
              reply(calls=[call("read", path="b.txt")]), reply("b.txt says B.")]
    res, tr = _turns_run(disk, script)
    assert res.reason == "done" and _users(res.messages) == TURNS
    assert res.said == ["ok, noted.", "", "a.txt says A.", "", "b.txt says B."]
    assert [(s["turn"], s["end_reason"], s["steps"]) for s in res.segments] == [(1, "done", 1), (2, "done", 2),
                                                                                (3, "done", 2)]
    assert all(s["wall_s"] >= 0 for s in res.segments) and res.not_sent == []
    assert _users(tr.bodies[-1]["messages"]) == TURNS  # the model saw them all in one conversation


def test_x3_a_message_that_runs_out_of_turns_lets_the_next_one_go(disk):
    """Test 6: ``max_turns`` is per message. Message 2 uses both its turns (``max_steps``, the loop's own
    code for it); message 3 is still sent."""
    script = [reply("ok."), reply(calls=[call("read", path="a.txt")]), reply(calls=[call("read", path="b.txt")]),
              reply("done 3.")]
    res, _tr = _turns_run(disk, script, profile=with_run_settings(P6, max_turns=2))
    assert [(s["turn"], s["end_reason"], s["steps"]) for s in res.segments] == [(1, "done", 1), (2, "max_steps", 2),
                                                                                (3, "done", 1)]
    assert res.reason == "done" and _users(res.messages) == TURNS and res.not_sent == []


def test_x3_a_guard_end_in_message_2_ends_the_task(disk):
    """Test 7: message 2 repeats one call until the stuck guard ends the task (core4 answers a third read
    of one file "already read", so that answer repeats 5 times by the 7th call); message 3 is never sent
    and is recorded ``not_sent``."""
    stuck = [reply(calls=[call("read", path="a.txt")]) for _ in range(7)]
    res, tr = _turns_run(disk, [reply("ok."), *stuck], profile=C4)
    assert res.reason == "stuck" and res.not_sent == [3]
    assert [(s["turn"], s["end_reason"]) for s in res.segments] == [(1, "done"), (2, "stuck")]
    assert TURNS[2] not in json.dumps(tr.bodies)


def test_x3_stuck_counts_start_again_with_each_message(disk):
    """The same call with the same result once per message is not "stuck": the stuck guard counts within
    one message (a new message is a new job). 7 jobs that each read ``keep.txt``: from job 3 on the guard
    answers "already read" (its own per-path rule, unchanged), so counted over the task that answer would
    be "stuck" at job 7 (and nudged at job 5)."""
    turns = [f"Job {i}: check keep.txt." for i in range(1, 8)]
    script = [r for _ in turns for r in (reply(calls=[call("read", path="keep.txt")]), reply("keep.txt is there."))]
    res, _tr = _turns_run(disk, script, profile=C4, turns=turns)
    assert res.reason == "done" and [s["end_reason"] for s in res.segments] == ["done"] * 7
    assert res.nudges["stuck"] == 0


def test_x3_compaction_keeps_the_first_message_with_the_rules(disk):
    """Test 8, pinned: with a tiny ``num_ctx`` the conversation is compacted, and the first user message (the
    rules) survives: ``keep_first_messages=2`` keeps the system prompt and it. An old job's message can be
    compacted away like any other middle message (X3b keeps only the current one too)."""
    files = _filler(6)
    turns = ["RULES: never touch keep.txt; every answer goes in out.txt."] + [f"Job {i}: read d{i}.txt." for i in range(6)]
    script = [reply("ok.")] + [r for i in range(6) for r in (reply(calls=[call("read", path=f"d{i}.txt")]),
                                                              reply(f"d{i} read."))]
    prof = with_run_settings(C4, num_ctx=4096)
    res, tr = _turns_run(disk, script, profile=prof, turns=turns, seed={**files, "keep.txt": "keep\n"})
    assert res.compactions, "precondition: the conversation was compacted"
    assert res.messages[1] == {"role": "user", "content": turns[0]}  # the rules, right after the system prompt
    assert all(b["messages"][1]["content"] == turns[0] for b in tr.bodies)  # in every request
    assert not all(t in _users(res.messages) for t in turns[1:-1])  # an old job's message was dropped


def test_x3b_compaction_keeps_the_current_job_s_message(disk):
    """X3b red/green (the shape measured for the X3 report): at ``--num-ctx 4096`` a job that reads five files
    of ~2,200 characters is compacted mid-job. Before X3b its own message went too (requests 5 and 6 did not
    carry it); now every request from the one that first carried it holds it, and message 1 (the rules) too."""
    files = _filler(5)
    job = "Job 2: read d0..d4, then say done."
    turns = ["RULES: every answer goes in out.txt.", job]
    script = [reply("ok.")] + [reply(calls=[call("read", path=f"d{i}.txt")]) for i in range(5)] + [reply("done")]
    res, tr = _turns_run(disk, script, profile=with_run_settings(C4, num_ctx=4096), turns=turns, seed=files)
    assert res.compactions and res.reason == "done", "precondition: compacted mid-job"
    carry = [job in [m["content"] for m in b["messages"]] for b in tr.bodies]
    assert carry == [False] + [True] * (len(tr.bodies) - 1), carry  # request 0 is message 1's (the rules)
    assert all(b["messages"][1]["content"] == turns[0] for b in tr.bodies)
    assert [m["content"] for m in res.messages].count(job) == 1


def test_x3b_a_harness_note_is_not_the_job(disk):
    """The loop's own notes go in as user messages too (stuck, parse, cut, too many calls, the compaction
    note). The kept message is the task's latest user message, never one of those notes."""
    files = _filler(5)
    job = "Job 2: read d0..d4 (twice each), then say done."
    turns = ["RULES: every answer goes in out.txt.", job]
    reads = [reply(calls=[call("read", path=f"d{i}.txt"), call("read", path=f"d{i}.txt")]) for i in range(5)]
    prof = dataclasses.replace(with_run_settings(C4, num_ctx=4096),
                               loop=dataclasses.replace(C4.loop, max_calls_per_turn=1))  # a "too many calls" note each turn
    res, tr = _turns_run(disk, [reply("ok."), *reads, reply("done")], profile=prof, turns=turns, seed=files)
    assert res.compactions and "only the first 1 ran" in json.dumps(tr.bodies), "precondition: notes and a compaction"
    assert all(job in [m["content"] for m in b["messages"]] for b in tr.bodies[1:])


def test_x3_a_string_and_a_one_item_list_send_the_same_first_request(disk):
    """Test 12: ``"text"`` and ``["text"]`` are the same task."""
    bodies = []
    for user in ("Read a.txt.", ["Read a.txt."]):
        ws = disk.ws({"a.txt": "A\n"})
        tr = ScriptedTransport([reply("A.")])
        res = loop.run_task(P6, provider(tr, P6.provider), "fake:1", user, ws)
        assert res.reason == "done" and [s["turn"] for s in res.segments] == [1]
        bodies.append(tr.bodies[0])
    assert json.dumps(bodies[0], sort_keys=True) == json.dumps(bodies[1], sort_keys=True)


@pytest.mark.parametrize("bad", [[], [""], ["ok", 3], None])
def test_x3_turns_must_be_a_list_of_texts(disk, bad):
    ws = disk.ws({"a.txt": "A\n"})
    if bad == [""]:  # an empty message is still a text: allowed
        res = loop.run_task(P6, provider(ScriptedTransport([reply("ok")]), P6.provider), "fake:1", bad, ws)
        assert res.reason == "done"
        return
    with pytest.raises(ValueError, match="string or a non-empty list of strings"):
        loop.run_task(P6, provider(ScriptedTransport([]), P6.provider), "fake:1", bad, ws)


def test_x3_a_turns_task_through_the_runner_records_its_segments(disk):
    spec = {"id": "t_turns", "seed": {"a.txt": "A\n"}, "turns": ["one", "two"], "judge": lambda f, t, x: (True, "ok")}
    res, _dest = _pack(disk, P6, ScriptedTransport([reply("1."), reply("2.")]), [spec])
    row = res["tasks"]["t_turns"]["trials"][0]
    assert [(s["turn"], s["end_reason"]) for s in row["segments"]] == [(1, "done"), (2, "done")]
    assert row["not_sent"] == [] and row["pass"] is True


# --- X3.3 deadlines from the pack ------------------------------------------------------------------


def _long_pack(tmp_path, monkeypatch, tasks, **meta) -> Path:
    path = tmp_path / "long.json"
    path.write_text(json.dumps({"tasks": tasks, **meta}), encoding="utf-8")
    monkeypatch.setitem(runner.LOCAL_PACKS, "long", path)
    return path


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _clocked(monkeypatch, clock):
    """Run each task on ``clock``: the task's own watch and its trial clock (same deadline as the runner's)."""
    real = runner.run_task
    seen = []

    def run(prof, prov, model, user, ws, **kw):
        seen.append((prof.watchdog.task_wall_clock_s, kw["trial"].deadline_s))
        kw["trial"] = TrialClock(kw["trial"].deadline_s, clock)
        return real(prof, prov, model, user, ws, clock=clock, **kw)

    monkeypatch.setattr(runner, "run_task", run)
    return seen


def test_x3_a_pack_deadline_cuts_the_task_at_that_time(disk, tmp_path, monkeypatch):
    """Test 9: a local pack's ``deadline_s`` (task) and ``trial_deadline_s`` (pass) replace 180 / 900. On a
    fake clock that moves 100 s per model reply, a 250 s task ends ``task_timeout`` after its third reply."""
    clock = FakeClock()
    task = {"id": "l_slow", "prompt": "go", "seed": {"a.txt": "A\n"}, "deadline_s": 250, "long": {"kind": "fake"}}
    _long_pack(tmp_path, monkeypatch, [task], trial_deadline_s=3600)
    fake_judge(monkeypatch, lambda block, seed, files_end, res: {"ok": True, "why": "fake", "info": {}})
    seen = _clocked(monkeypatch, clock)

    def model(messages):
        clock.t += 100
        return reply(calls=[call("read", path="a.txt")])

    res, _dest = _pack(disk, C4, ReactiveTransport(model), load_pack("long"), pack="long")
    row = res["tasks"]["l_slow"]["trials"][0]
    assert seen == [(250, 3600)] and row["end_reason"] == "task_timeout"
    assert row["deadlines"] == {"task_s": 250, "trial_s": 3600, "source": "pack"}
    assert len(row["calls"]) == 2  # replies at t+100, t+200 ran; the one at t+300 is past the 250 s


def test_x3_lab_packs_keep_180_and_900(disk, monkeypatch):
    seen = _clocked(monkeypatch, FakeClock())
    res, _dest = _pack(disk, C4, ScriptedTransport([reply("done")]), [t for t in load_pack("work") if t["id"] == LAB_TASK])
    assert seen == [(180, 900)]
    assert res["tasks"][LAB_TASK]["trials"][0]["deadlines"] == {"task_s": 180, "trial_s": 900, "source": "profile"}


# --- X3.4 hooks for the long pack (fake modules: OpenCode builds the real ones in X4) ---------------


def fake_module(monkeypatch, name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def fake_judge(monkeypatch, fn):
    return fake_module(monkeypatch, "localbot.exam.long_judge", verdict=fn)


def test_x3_a_generator_s_files_land_in_work_with_the_task_s_key(disk, tmp_path, monkeypatch):
    """Test 11a: ``generator: {name, params}`` → ``long_seed.build(name, params, secret)``; its files are merged
    into the seed and written to ``work/``; ``secret`` is the task's own fresh fake key, so it is redacted."""
    calls = []

    def build(name, params, secret):
        calls.append((name, params, secret))
        return {"logs/big.log": "\n".join(f"line {i}" for i in range(50)) + "\n", "notes.txt": f"key={secret}\n"}

    fake_module(monkeypatch, "localbot.exam.long_seed", build=build)
    seen_files = []
    fake_judge(monkeypatch, lambda block, seed, files_end, res: (seen_files.append(sorted(files_end)),
                                                                 {"ok": True, "why": "fine", "info": {"n": 1}})[1])
    _long_pack(tmp_path, monkeypatch, [{"id": "l_gen", "prompt": "summarise", "seed": {"a.txt": "A\n"},
                                        "generator": {"name": "log", "params": {"lines": 50}}, "long": {"want": 3}}])
    tasks = load_pack("long")
    (name, params, secret), = calls
    assert (name, params) == ("log", {"lines": 50}) and secret == tasks[0]["secret_values"][0]
    assert sorted(tasks[0]["seed"]) == ["a.txt", "logs/big.log", "notes.txt"]
    res, dest = _pack(disk, C4, ScriptedTransport([reply("done")]), tasks, pack="long")
    assert seen_files == [["a.txt", "logs/big.log", "notes.txt"]]
    row = res["tasks"]["l_gen"]["trials"][0]
    assert row["pass"] is True and row["long"] == {"ok": True, "why": "fine", "info": {"n": 1}}
    assert secret not in dest.read_text(encoding="utf-8")


def test_x3_a_generator_file_may_not_replace_a_seed_file(tmp_path, monkeypatch):
    fake_module(monkeypatch, "localbot.exam.long_seed", build=lambda n, p, s: {"a.txt": "other\n"})
    _long_pack(tmp_path, monkeypatch, [{"id": "l_clash", "prompt": "x", "seed": {"a.txt": "A\n"},
                                        "generator": {"name": "x", "params": {}}}])
    with pytest.raises(ValueError, match="a.txt"):
        load_pack("long")


@pytest.mark.parametrize("judge,why", [
    (lambda block, seed, files_end, res: {"ok": False, "why": "found.txt has 2 of 3 lines", "info": {}}, "2 of 3"),
    (lambda block, seed, files_end, res: 1 / 0, "long judge error: ZeroDivisionError"),
    (lambda block, seed, files_end, res: {"ok": "yes", "why": "not a bool"}, "not a bool"),
], ids=["judge_says_no", "judge_crashes", "ok_is_not_true"])
def test_x3_a_long_judge_that_says_no_or_crashes_fails_the_try(disk, tmp_path, monkeypatch, judge, why):
    """Test 11b: ``long: {...}`` → ``long_judge.verdict(block, seed, files_end, res)``, used like the injection
    verdict. ``ok`` other than ``True`` fails; a crash fails, never a pass."""
    fake_judge(monkeypatch, judge)
    _long_pack(tmp_path, monkeypatch, [{"id": "l_j", "prompt": "x", "seed": {"a.txt": "A\n"}, "long": {"k": 1}}])
    res, _dest = _pack(disk, C4, ScriptedTransport([reply("done")]), load_pack("long"), pack="long")
    row = res["tasks"]["l_j"]["trials"][0]
    assert row["pass"] is False and row["core4_judge_pass"] is False and why in row["judge_reason"], row["judge_reason"]


def test_x3_a_long_judge_s_info_is_redacted(disk, tmp_path, monkeypatch):
    """Test 11c: ``info`` is saved names and numbers only; a fake key in it is redacted in the file."""
    keys = []
    fake_judge(monkeypatch, lambda block, seed, files_end, res: {"ok": True, "why": "fine",
                                                                 "info": {"tokens_read": 120, "note": f"saw {keys[0]}"}})
    _long_pack(tmp_path, monkeypatch, [{"id": "l_r", "prompt": "x", "seed": {".env": "K={SECRET}\n"}, "long": {}}])
    tasks = load_pack("long")
    keys.append(tasks[0]["secret_values"][0])
    res, dest = _pack(disk, C4, ScriptedTransport([reply("done")]), tasks, pack="long")
    text = dest.read_text(encoding="utf-8")
    assert keys[0] not in text and res["tasks"]["l_r"]["trials"][0]["long"]["info"]["tokens_read"] == 120


def test_x3_a_task_without_the_fields_never_imports_the_long_modules(disk, monkeypatch):
    """The two imports are lazy: the injection pack runs with both modules unimportable."""
    monkeypatch.setitem(sys.modules, "localbot.exam.long_seed", None)  # an import would raise ImportError
    monkeypatch.setitem(sys.modules, "localbot.exam.long_judge", None)
    tasks = [t for t in load_pack("injection") if t["id"] == "i_quoted_cat_readme"]
    res, _dest = _pack(disk, C4, ScriptedTransport([reply("ok")]), tasks, pack="injection")
    assert res["tasks"]["i_quoted_cat_readme"]["trials"][0]["end_reason"] == "done"


def test_x3_the_long_pack_file_is_needed_only_when_chosen(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(runner.LOCAL_PACKS, "long", tmp_path / "missing.json")
    assert "long" in runner.LOCAL_PACKS and _ran_tasks(monkeypatch, ["--pack", "work", "--tasks", "e_read", "m:1"])
    with pytest.raises(SystemExit) as e:
        runner.main(["--pack", "long", "--profile", "core4", "m:1"])
    assert e.value.code == 2 and "missing.json" in capsys.readouterr().err


# --- X3: docs/s10_recompute.py never adds files at different run settings together ----------------

W2S_SRC = "privacy_core4_llama3.2_1b_20261102T034106Z.json"  # run "W2S 6a58d2d" (the first W2S run printed)
INJ_SRC = "injection_core4_qwen3-8b-ft-agentic-v1_latest_20261102T134003Z.json"  # run "INJ 6a58d2d"
G6_SRC = "privacy_core4_gpt-oss_20b_20261031T170443Z.json"  # run "G6 S8h" (an exam gate)
WIDE = {"num_ctx": 32768, "max_turns": 8, "num_predict": 512, "num_predict_by_model": [], "source": "cli"}
WIDE_LABEL = "(num_ctx 32768 / max_turns 8 / num_predict 512)"


def _s10_with(tmp_path, run_name: str, src: str, edit):
    """s10 over a copy of today's results plus one copy of ``src`` run at ``WIDE``, listed in ``run_name``,
    with ``edit`` applied to it (a planted count, so a sum would show)."""
    from tests.test_result_tools import _manifest, _root, _run_s10, _variant

    name, d = _variant(src, "20261103T000000Z")
    d["run_env"] = {**(d.get("run_env") or {}), "run_settings": dict(WIDE)}
    edit(d)
    man = _manifest()
    next(r for r in man["runs"] if r["run"] == run_name)["files"].append(name)
    root = _root(tmp_path, man)
    (root / "results" / "stage1" / name).write_text(json.dumps(d), encoding="utf-8")
    return name, _run_s10(root)


def _first_try(d: dict) -> dict:
    return next(iter(d["tasks"].values()))["trials"][0]


def test_x3_s10_inj_rows_at_other_settings_stand_apart(tmp_path):
    """A second qwen-ft INJ file in one run, at num_ctx 32768, with one planted ``executed`` call: its own §16
    row (1 executed, bar NO); the default qwen-ft row keeps its own numbers (1 executed, from its own file)."""
    from tests.test_result_tools import row, section

    name, r = _s10_with(tmp_path, "INJ 6a58d2d", INJ_SRC,
                        lambda d: _first_try(d)["injection"]["executed"].append({"i": 0, "call": "bash"}))
    assert r.returncode == 0, r.stderr[-2000:]
    sec = section(r.stdout, "16")
    wide = row(sec, f"| qwen-ft {WIDE_LABEL} |")
    assert wide[1].startswith("20261103T000000Z") and wide[3] == "2" and wide[-1] == "NO", wide  # 1 saved + 1 planted
    base = row(sec, "| qwen-ft | 20261102T134003Z")
    assert base[3] == "1" and base[-1] == "NO", base  # its own file only: not 3


def test_x3_s10_w2s_rows_at_other_settings_stand_apart(tmp_path):
    """A second llama3.2:1b privacy core4 file in a W2S run, at num_ctx 32768, with a planted host change: its own
    §15.1 row and §15.2 bar row (NO); the default model row still holds its 4 pack runs and its bar."""
    from tests.test_result_tools import row, section

    name, r = _s10_with(tmp_path, "W2S 6a58d2d", W2S_SRC, lambda d: _first_try(d)["host_change"].append("planted"))
    assert r.returncode == 0, r.stderr[-2000:]
    assert row(section(r.stdout, "15.1"), f"| llama3.2:1b {WIDE_LABEL} | privacy | core4 |")[3] == "20261103T000000Z"
    bar = section(r.stdout, "15.2")
    wide = row(bar, f"| llama3.2:1b {WIDE_LABEL} |")
    assert wide[1] == "1 of 1" and wide[-2] == "NO", wide
    base = row(bar, "| llama3.2:1b |")
    assert base[1] == "4 of 4" and base[-2] == "yes", base


def test_x3_s10_an_exam_gate_file_at_other_settings_stops(tmp_path):
    """The exam gates (G3–G6) are at the lab's settings: a file run at others, listed in one, stops the script
    with a plain message instead of entering a lab-settings number."""
    name, r = _s10_with(tmp_path, "G6 S8h", G6_SRC, lambda d: None)
    assert r.returncode == 2 and r.stdout == ""
    assert "exam gate G6 (run G6 S8h) lists a file run at num_ctx 32768" in r.stderr and name in r.stderr, r.stderr
