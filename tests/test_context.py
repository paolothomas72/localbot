"""context.py (S4, spec §5.8): caps, eviction, compaction, and the post-compaction reread gate,
including the fake-model test: after compaction the bot re-reads TASK_MANIFEST.json and the
target file before writing, and nothing is overwritten."""
from __future__ import annotations

import dataclasses
import json

from localbot.config import ContextConfig, UndoConfig, get_profile
from localbot.context import (
    COMPACT_NOTE, ContextManager, RereadGate, cap_output, compact, estimate_tokens, evict_old_results,
)
from localbot.guard.undo import UndoStore
from localbot.loop import run_task
from tests.fakes import ReactiveTransport, call, provider, reply


def test_cap_output():
    assert cap_output("abc", 4000) == "abc"
    out = cap_output("x" * 4100, 4000)
    assert out.startswith("x" * 4000) and out.endswith("[truncated 100 chars]")
    assert cap_output("x" * 9000, None) == "x" * 9000


def _msgs(n_tool: int, size: int) -> list[dict]:
    m = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    for i in range(n_tool):
        m.append({"role": "assistant", "content": "", "tool_calls": [call("read", path=f"f{i}")]})
        m.append({"role": "tool", "content": "y" * size, "name": "read"})
    return m


def test_evict_keeps_first_2_and_last_6():
    m = _msgs(6, 3000)  # 2 + 12 messages
    n = evict_old_results(m, 2500, 2, 6)
    tools = [i for i, x in enumerate(m) if x["role"] == "tool"]
    evicted = [i for i in tools if x_ev(m[i])]
    assert n == 3 and evicted == [3, 5, 7]  # the last 6 messages (8..13) untouched
    assert all(not x_ev(m[i]) for i in tools if i >= len(m) - 6)
    assert evict_old_results(m, 2500, 2, 6) == 0  # idempotent
    assert evict_old_results(_msgs(6, 3000), None, 2, 6) == 0  # off


def x_ev(msg: dict) -> bool:
    return msg["content"].startswith("[harness: evicted")


def test_small_results_never_evicted():
    m = _msgs(6, 2400)
    assert evict_old_results(m, 2500, 2, 6) == 0


def test_compaction_hysteresis_and_shape():
    m = _msgs(10, 1000)
    est = estimate_tokens(m)
    num_ctx = int(est / 0.85)  # above the 0.80 trigger
    out, dropped = compact(m, num_ctx, 0.80, 0.60, 2)
    assert dropped > 0
    assert out[:2] == m[:2] and out[2]["content"] == COMPACT_NOTE.format(n=dropped)
    assert estimate_tokens(out) <= 0.60 * num_ctx
    assert out[3]["role"] != "tool"  # no orphan tool result
    same, d0 = compact(m, int(est / 0.75), 0.80, 0.60, 2)  # under the trigger: untouched
    assert d0 == 0 and same is m


def test_p2_compaction_budget_reserves_schemas_and_num_predict():
    """H1 P2: num_ctx holds the prompt AND the tool schemas AND the reply (num_predict). Before H1
    only the messages were counted, so a prompt that "fit" left no room for the answer."""
    m = _msgs(10, 1000)
    est = estimate_tokens(m)
    num_ctx = int(est / 0.75)  # the messages alone are under the 0.80 trigger
    same, d0 = compact(m, num_ctx, 0.80, 0.60, 2)
    assert d0 == 0
    reserve = int(0.10 * num_ctx)  # schemas + num_predict: messages + reserve > 0.80
    out, dropped = compact(m, num_ctx, 0.80, 0.60, 2, reserve=reserve)
    assert dropped > 0 and estimate_tokens(out) + reserve <= 0.60 * num_ctx


def test_p2_context_manager_reserves_the_profiles_schemas_and_num_predict():
    p = get_profile("core4")
    cm = ContextManager(p.context, p.provider.num_ctx, tools=list(p.tools.schemas), num_predict=512)
    assert cm.reserve == estimate_tokens(list(p.tools.schemas)) + 512 > 512


def test_context_manager_off_in_parity():
    cm = ContextManager(get_profile("parity_lab6").context, 8192)
    m = _msgs(10, 5000)
    before = json.dumps(m)
    assert cm.after_turn(m) is m and json.dumps(m) == before
    assert cm.cap("z" * 9000) == "z" * 9000


def test_p2_reread_gate_strips_only_a_leading_dot_slash():
    """H1 P2: ``lstrip("./")`` strips characters, not a prefix: ".env" became "env" and "../x"
    became "x", so reading "env" counted as having re-read ".env"."""
    g = RereadGate()
    g.compacted()
    g.note_read("env")
    assert g.check_write(".env", target_exists=True, manifest_exists=False) is not None  # never re-read
    g.note_read("./.env")
    assert g.check_write(".env", target_exists=True, manifest_exists=False) is None
    g.note_read(".\\sub\\a.txt")
    assert g.check_write("sub/a.txt", target_exists=True, manifest_exists=False) is None


def test_reread_gate():
    g = RereadGate()
    assert g.check_write("a.txt", True, True) is None  # no compaction yet
    g.compacted()
    assert g.check_write("a.txt", True, True) == \
        "harness: context was compacted. Read TASK_MANIFEST.json and a.txt again before writing a.txt."
    g.note_read("TASK_MANIFEST.json")
    assert g.check_write("a.txt", True, True) == \
        "harness: context was compacted. Read a.txt again before writing a.txt."
    g.note_read("./a.txt")
    assert g.check_write("a.txt", True, True) is None
    assert g.check_write("new.txt", False, True) is None  # new file: manifest read is enough
    g.compacted()
    assert g.check_write("new.txt", False, False) is None  # no manifest, new file


# --- fake-model test through the real loop ----------------------------------------------------

MANIFEST = json.dumps({"goal": "append the line 'three' to notes.txt", "paths": ["notes.txt"],
                       "last_verified_bytes": 10})
NOTES = "one\ntwo\n"


def _careless_then_careful(messages: list[dict]) -> dict:
    """Reads the manifest and notes, then a big log (which forces compaction). Then, having
    'forgotten', it first tries to overwrite notes.txt; refused, it re-reads both files and
    appends with an edit. Decides from the conversation only, like a model."""
    tool_texts = [m["content"] for m in messages if m.get("role") == "tool"]
    compacted = any(m.get("role") == "user" and "context compacted" in m.get("content", "") for m in messages)
    last = tool_texts[-1] if tool_texts else ""
    if not compacted:
        if not tool_texts:
            return reply(calls=[call("read", path="TASK_MANIFEST.json")])
        if len(tool_texts) == 1:
            return reply(calls=[call("read", path="notes.txt")])
        return reply(calls=[call("read", path="big.log")])
    if "context was compacted" not in " ".join(tool_texts) and not any("Wrote" in t or "Updated" in t for t in tool_texts):
        return reply(calls=[call("write", path="notes.txt", content="three\n")])  # careless overwrite
    reread = [m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")]
    names = [(c["function"]["name"], c["function"]["arguments"].get("path")) for m in reread for c in m["tool_calls"]]
    after_refusal = names[names.index(("write", "notes.txt")) + 1:] if ("write", "notes.txt") in names else []
    if ("read", "TASK_MANIFEST.json") not in after_refusal:
        return reply(calls=[call("read", path="TASK_MANIFEST.json")])
    if ("read", "notes.txt") not in after_refusal:
        return reply(calls=[call("read", path="notes.txt")])
    if "Updated notes.txt" not in last:
        return reply(calls=[call("edit", path="notes.txt", old_string="two\n", new_string="two\nthree\n")])
    return reply("Appended 'three' to notes.txt.")


def test_after_compaction_bot_rereads_before_writing_no_overwrite(disk):
    base = get_profile("core4")
    prof = dataclasses.replace(base, provider=dataclasses.replace(base.provider, num_ctx=1500))
    ws = disk.ws({"TASK_MANIFEST.json": MANIFEST, "notes.txt": NOTES, "big.log": "log line\n" * 600})
    tr = ReactiveTransport(_careless_then_careful)
    store = UndoStore(UndoConfig(store_dir=disk.base / "state"), "bot1", disk.root)
    res = run_task(prof, provider(tr, prof.provider), "fake:1", "Do the task in TASK_MANIFEST.json.", ws,
                   undo=store)
    assert res.reason == "done", res.summary()
    assert res.compactions and res.compactions[0] > 0
    refusal = "ERROR: harness: context was compacted. Read TASK_MANIFEST.json and notes.txt again before writing notes.txt."
    assert refusal in res.tool_results
    # order after the refusal: manifest read, notes read, then the edit
    i = res.tool_results.index(refusal)
    later = res.tool_results[i + 1:]
    assert later[0].startswith("1\t{") and later[1] == "1\tone\n2\ttwo" and later[2] == "Updated notes.txt"
    assert (disk.root / "notes.txt").read_text(encoding="utf-8") == "one\ntwo\nthree\n"  # no overwrite
    assert res.messages[2]["content"].startswith("[harness: context compacted")
    assert all(b["options"]["num_ctx"] == 1500 for b in tr.bodies)


# --- S1-X X3b: compaction keeps the current job's user message too ----------------------------------


def _jobs(n_before: int, n_after: int, size: int) -> tuple[list[dict], dict]:
    """system, the rules, ``n_before`` reads, the current job's user message, ``n_after`` reads of that job."""
    m = _msgs(n_before, size)
    m[1] = {"role": "user", "content": "RULES"}
    job = {"role": "user", "content": "JOB 2"}
    m.append(job)
    for i in range(n_after):
        m.append({"role": "assistant", "content": "", "tool_calls": [call("read", path=f"g{i}")]})
        m.append({"role": "tool", "content": "z" * size, "name": "read"})
    return m, job


def test_x3b_the_current_job_message_is_never_dropped():
    """X3b: ``keep`` (the loop's current user message) goes like message 1: never dropped. The others go
    oldest first (everything before the job, then the job's own oldest steps); order is kept."""
    msgs, job = _jobs(6, 6, 800)  # the job's own 6 steps alone are over the budget
    old, n_old = compact(msgs, 2000, 0.8, 0.6, keep_first=2)
    assert n_old and job not in old  # today: the job's message goes with the old middle
    out, n = compact(msgs, 2000, 0.8, 0.6, keep_first=2, keep=job)
    assert n and out[:2] == msgs[:2] and out[2]["content"] == COMPACT_NOTE.format(n=n)
    assert sum(m is job for m in out) == 1
    after = out[out.index(job) + 1:]
    assert after == msgs[len(msgs) - len(after):]  # the job's newest steps, unchanged, in order
    assert all(m in msgs[msgs.index(job):] for m in after)
    assert not after or after[0]["role"] != "tool"  # never a result without its call
    assert estimate_tokens(out) <= 0.6 * 2000 or after == []


def test_x3b_keep_is_message_1_or_unknown_then_nothing_changes():
    """A one-message task: the current message IS message 1 (in the head). Kept once, and the output is
    byte for byte today's, on every path (a cut, an orphan result first, a head over the budget). A message
    that is not in the list changes nothing either."""
    shapes = [(_msgs(12, 800), 2000, 0), (_msgs(1, 50) + [{"role": "tool", "content": "o" * 900, "name": "read"}]
                                          + _msgs(8, 900)[2:], 2000, 0), (_msgs(6, 800), 2000, 1500)]
    shapes[2][0][1] = {"role": "user", "content": "u" * 9000}  # message 1 alone is over the budget
    for msgs, num_ctx, reserve in shapes:
        today = compact(msgs, num_ctx, 0.8, 0.6, 2, reserve)
        for keep in (msgs[1], {"role": "user", "content": "not in the list"}):
            got = compact(msgs, num_ctx, 0.8, 0.6, 2, reserve, keep=keep)
            assert json.dumps(got) == json.dumps(today)
        assert sum(m is msgs[1] for m in got[0]) == 1


def test_x3b_message_1_and_the_job_over_the_budget_take_today_s_path():
    """When message 1 and the job alone are over the budget, the same path as today's "message 1 alone does
    not fit": every other message is dropped and nothing more is done (still over; no error, no cut)."""
    msgs, job = _jobs(3, 3, 400)
    job["content"] = "JOB " + "j" * 6000
    out, n = compact(msgs, 2000, 0.8, 0.6, keep_first=2, keep=job)
    assert out == msgs[:2] + [{"role": "user", "content": COMPACT_NOTE.format(n=n)}, job]
    assert n == len(msgs) - 3 and estimate_tokens(out) > 0.6 * 2000
    msgs1 = _msgs(3, 400)
    msgs1[1] = {"role": "user", "content": "u" * 9000}  # today's path, for comparison: message 1 alone
    out1, n1 = compact(msgs1, 2000, 0.8, 0.6, keep_first=2)
    assert out1 == msgs1[:2] + [{"role": "user", "content": COMPACT_NOTE.format(n=n1)}] and n1 == len(msgs1) - 2
