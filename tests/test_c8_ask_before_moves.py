"""BRIEF C8: ask before moving a file that was already there (product / chat only).

Pre-check names a protected ``mv`` / ``git mv`` / glob / brace move with why ``move`` and asks before
the command runs. Post-step routes (e.g. ``os.rename``) follow the C2 after_step path. No ask hook =
moves allowed exactly as on 1868f9a (golden below). Delete / wipe / overwrite behaviour unchanged.

Edges (C7 review): last permitted step (max_turns=1), next message, per-step allowance not persisting
for the *same relocated* file, ``/dev/null`` still delete, bot-created free, protected dest overwrite
not approved by a source-move yes, grouped files, harness ``move_aside`` no re-ask, plain-marker on
new ask paths, Driver recognition of the move prompt (scope note if missing).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import pytest

from localbot import chat
from localbot.config import UndoConfig, get_profile
from localbot.guard.delete import DELETE_MSG
from localbot.guard.undo import UndoStore
from localbot.loop import run_task
from tests.fakes import FakeShell, ScriptedTransport, call, provider, reply
from tests.test_chat import World

C4 = get_profile("core4")
PRODUCT = get_profile("core4_product")
HELD = f"ERROR: {DELETE_MSG}"
SEED = {"notes.txt": "N\n", "keep.txt": "K\n", "old_draft.txt": "OLD\n"}
MARKER = "PLAIN-MARKER-8820"

# Recorded on 1868f9a (unchanged tip) by outbox/c8_golden_1868f9a_b.py: core4 with undo, NO ask hook.
GOLDEN_1868F9A_MOVE = json.loads(
    r'''{"delete_attempt": false, "messages_sha256": "8de3d6672515ebb065759e07f7e5e55c64b83dfc419c735d2d235b0210afb801", "n_messages": 5, "ran": [true], "reason": "done", "summary": {"arg_errors": [], "arg_extra": [], "call_source": ["native", "none"], "compactions": [], "container_scans": [], "delete_attempt": false, "error": "", "eval_count": 60, "false_success": [], "git_changed": false, "no_call_reason": null, "nudges": {"cut": 0, "parse": 0, "stuck": 0}, "reason": "done", "text_call_attempts": [], "tool_errors": [], "tools": ["bash"], "veto_texts": [], "vetoes": 0}, "tool_calls": [["bash", {"command": "mv notes.txt docs/"}]], "tool_results": [""], "undone": [false], "vetoes": 0}'''
)


# --- helpers ---------------------------------------------------------------------------------------------


def _store(disk) -> UndoStore:
    return UndoStore(UndoConfig(store_dir=disk.base / "state"), "bot1", disk.root)


def _mv(disk, src: str, dst: str):
    """FakeShell side effect: move src to dst (dst ending with / = folder)."""

    def run():
        s = disk.root / src
        dest = disk.root / dst / Path(src).name if dst.endswith("/") else disk.root / dst
        dest.parent.mkdir(parents=True, exist_ok=True)
        s.rename(dest)
        return ""

    return run


def _rename(disk, src: str, dst: str):
    def run():
        s, d = disk.root / src, disk.root / dst
        d.parent.mkdir(parents=True, exist_ok=True)
        os.rename(s, d)
        return ""

    return run


def _exists(disk, rel: str) -> bool:
    return (disk.root / rel).exists()


def _text(disk, rel: str):
    p = disk.root / rel
    return p.read_text(encoding="utf-8") if p.exists() else None


def _boom():
    raise AssertionError("command ran after a refusal")


class Ask:
    """ask_delete hook: returns answer (or next of answers); records every paths list."""

    def __init__(self, answer="no", answers=None):
        self.answer, self.answers, self.calls = answer, list(answers or []), []

    def __call__(self, paths):
        self.calls.append(list(paths))
        if self.answers:
            return self.answers.pop(0)
        return self.answer


def _core(res):
    blob = json.dumps(res.messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
    s = res.summary()
    for k in ("elapsed_s", "provider_s", "eval_s", "tok_s", "provider", "event_errors", "ask_errors"):
        s.pop(k, None)
    return {
        "reason": res.reason, "n_messages": len(res.messages),
        "messages_sha256": hashlib.sha256(blob).hexdigest(),
        "tool_calls": [[n, a] for n, a in res.tool_calls],
        "tool_results": list(res.tool_results),
        "ran": list(res.ran or []), "undone": list(res.undone or []),
        "delete_attempt": res.delete_attempt, "vetoes": res.vetoes, "summary": s,
    }


def _profile_max_turns(n: int, base=C4):
    return dataclasses.replace(base, loop=dataclasses.replace(base.loop, max_turns=n))


def _go(disk, command, ask=None, seed=None, script=None, replies=None, profile=C4, **kw):
    ws = disk.ws(dict(SEED if seed is None else seed))
    shell = FakeShell(script or {command: _mv(disk, "notes.txt", "docs/")})
    store = _store(disk)
    if ask is not None:
        kw["ask_delete"] = ask
    body = replies or [reply(calls=[call("bash", command=command)]), reply("done")]
    res = run_task(profile, provider(ScriptedTransport(body), profile.provider),
                   "fake:1", "reorganize the folder", ws, shell=shell, undo=store, **kw)
    return res, store, shell


def _asks(events):
    return [e for e in events if e.get("event") == "ask"]


def _assert_move_no_note(note: str, named: str):
    """Exact independent checks for a refused move (no precedence-OR)."""
    assert isinstance(note, str)
    assert note.startswith("ERROR: harness"), note
    assert "not from the user" in note, note
    assert named in note, note
    assert "said no" in note.lower(), note
    assert "mov" in note.lower(), note


# --- 1. no-hook golden (contract: NO hook only; both profiles) ------------------------------------------


@pytest.mark.parametrize("profile", [C4, PRODUCT])
def test_c8_without_ask_hook_a_move_runs_exactly_as_on_1868f9a(disk, profile):
    """Exam / gate / LONG: no ask hook => move stands, golden from 1868f9a. Not a new-behaviour proof."""
    res, _, _ = _go(disk, "mv notes.txt docs/", ask=None, profile=profile)
    assert _core(res) == GOLDEN_1868F9A_MOVE
    assert _exists(disk, "docs/notes.txt") and not _exists(disk, "notes.txt")


def test_c8_no_hook_os_rename_also_stands_as_on_1868f9a(disk):
    cmd = "python -c \"import os; os.rename('notes.txt','docs/notes.txt')\""
    res, _, _ = _go(disk, cmd, script={cmd: _rename(disk, "notes.txt", "docs/notes.txt")})
    assert res.reason == "done" and res.ran == [True] and res.undone == [False]
    assert _exists(disk, "docs/notes.txt") and not _exists(disk, "notes.txt")
    assert res.delete_attempt is False


# --- 1b. with-hook is NEW behaviour (not a golden pin of the bug) ---------------------------------------


def test_c8_with_ask_hook_answer_no_refuses_the_move_and_does_not_match_no_hook_golden(disk):
    events = []
    ask = Ask("no")
    res, _, shell = _go(disk, "mv notes.txt docs/", ask=ask, on_event=events.append)
    assert ask.calls == [["notes.txt"]], ask.calls
    assert _asks(events) and _asks(events)[0]["why"] == ["move"]
    assert _exists(disk, "notes.txt") and not _exists(disk, "docs/notes.txt")
    assert shell.ran == []
    assert _core(res) != GOLDEN_1868F9A_MOVE


def test_c8_with_ask_hook_answer_yes_runs_the_move(disk):
    ask = Ask("yes")
    res, _, shell = _go(disk, "mv notes.txt docs/", ask=ask)
    assert ask.calls == [["notes.txt"]]
    assert shell.ran == ["mv notes.txt docs/"]
    assert _exists(disk, "docs/notes.txt") and not _exists(disk, "notes.txt")
    assert res.ran == [True] and res.undone == [False]


# --- 2. pre-check shapes ---------------------------------------------------------------------------------


def test_c8_precheck_mv_asks_with_why_move_before_the_command_runs_answer_no(disk):
    events = []
    ask = Ask("no")
    res, _, shell = _go(disk, "mv notes.txt docs/", ask=ask, on_event=events.append)
    assert ask.calls == [["notes.txt"]], ask.calls
    asks = _asks(events)
    assert len(asks) == 1
    assert asks[0]["why"] == ["move"]
    assert asks[0]["paths"] == ["notes.txt"]
    assert asks[0]["place"] == "precheck"
    assert asks[0].get("dests") == ["docs/notes.txt"]
    assert _exists(disk, "notes.txt") and not _exists(disk, "docs/notes.txt")
    assert shell.ran == []
    _assert_move_no_note(res.tool_results[0], "notes.txt")


def test_c8_precheck_mv_answer_yes_runs_the_move(disk):
    ask = Ask("yes")
    res, _, shell = _go(disk, "mv notes.txt docs/", ask=ask)
    assert ask.calls == [["notes.txt"]]
    assert shell.ran == ["mv notes.txt docs/"]
    assert _exists(disk, "docs/notes.txt") and not _exists(disk, "notes.txt")
    assert res.ran == [True] and res.undone == [False]


def test_c8_precheck_yes_relocates_protection_so_rm_of_new_path_asks_delete(disk):
    events = []
    ask = Ask(answers=["yes", "no"])

    def rm_docs():
        p = disk.root / "docs" / "notes.txt"
        if p.exists():
            p.unlink()
        return ""

    script = {"mv notes.txt docs/": _mv(disk, "notes.txt", "docs/"), "rm docs/notes.txt": rm_docs}
    body = [reply(calls=[call("bash", command="mv notes.txt docs/")]),
            reply(calls=[call("bash", command="rm docs/notes.txt")]),
            reply("done")]
    _go(disk, "mv notes.txt docs/", ask=ask, script=script, replies=body, on_event=events.append)
    asks = _asks(events)
    assert len(asks) == 2
    assert asks[0]["why"] == ["move"] and asks[0]["paths"] == ["notes.txt"]
    assert asks[1]["why"] == ["delete"] and asks[1]["paths"] == ["docs/notes.txt"]
    assert _exists(disk, "docs/notes.txt")


@pytest.mark.parametrize("command,expected_paths,expected_dests", [
    ("git mv notes.txt docs/", ["notes.txt"], ["docs/notes.txt"]),
    ("mv -t docs notes.txt", ["notes.txt"], ["docs/notes.txt"]),
    ("mv --target-directory=docs notes.txt", ["notes.txt"], ["docs/notes.txt"]),
    ("mv *.txt docs/", ["keep.txt", "notes.txt", "old_draft.txt"],
     ["docs/keep.txt", "docs/notes.txt", "docs/old_draft.txt"]),
    ("mv {notes,keep}.txt docs/", ["keep.txt", "notes.txt"], ["docs/keep.txt", "docs/notes.txt"]),
    ("mv notes.txt keep.txt docs/", ["keep.txt", "notes.txt"], ["docs/keep.txt", "docs/notes.txt"]),
])
def test_c8_precheck_other_move_shapes_ask_once_with_exact_paths_answer_no(
        disk, command, expected_paths, expected_dests):
    events = []
    ask = Ask("no")
    res, _, shell = _go(disk, command, ask=ask, script={command: _boom}, on_event=events.append)
    assert len(ask.calls) == 1
    assert sorted(ask.calls[0]) == expected_paths
    asks = _asks(events)
    assert len(asks) == 1
    assert sorted(asks[0]["paths"]) == expected_paths
    assert asks[0]["why"] == ["move"] * len(expected_paths)
    assert sorted(asks[0].get("dests") or []) == sorted(expected_dests)
    assert shell.ran == []
    assert _exists(disk, "notes.txt")
    _assert_move_no_note(res.tool_results[0], expected_paths[0])


@pytest.mark.parametrize("command", [
    "git mv notes.txt docs/",
    "mv -t docs notes.txt",
    "mv *.txt docs/",
    "mv {notes,keep}.txt docs/",
])
def test_c8_precheck_other_move_shapes_answer_yes_runs(disk, command):
    ask = Ask("yes")
    # Side effect must match the command's operands only (moving an un-allowed protected
    # file would rightly restore the whole step).
    if "git mv" in command or "-t" in command:
        names = ["notes.txt"]
    elif "{notes,keep}" in command:
        names = ["notes.txt", "keep.txt"]
    else:  # glob *.txt
        names = ["notes.txt", "keep.txt", "old_draft.txt"]

    def run_named():
        for name in names:
            s = disk.root / name
            if s.exists():
                dest = disk.root / "docs" / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                s.rename(dest)
        return ""

    res, _, shell = _go(disk, command, ask=ask, script={command: run_named})
    assert ask.calls and shell.ran == [command]
    assert _exists(disk, "docs/notes.txt") and not _exists(disk, "notes.txt")
    assert res.ran == [True]


def test_c8_mv_to_dev_null_is_still_a_delete_ask_not_a_move(disk):
    events = []
    ask = Ask("no")

    def rm_notes():
        (disk.root / "notes.txt").unlink()
        return ""

    _go(disk, "mv notes.txt /dev/null", ask=ask, script={"mv notes.txt /dev/null": rm_notes},
        on_event=events.append)
    asks = _asks(events)
    assert asks and asks[0]["why"] == ["delete"] and asks[0]["paths"] == ["notes.txt"]
    assert _exists(disk, "notes.txt")


# --- 3. post-step route (os.rename) ----------------------------------------------------------------------


def test_c8_poststep_os_rename_restores_and_asks_with_why_move_answer_no(disk):
    events = []
    ask = Ask("no")
    cmd = "python -c \"import os; os.rename('notes.txt','docs/notes.txt')\""
    res, _, _ = _go(disk, cmd, ask=ask, script={cmd: _rename(disk, "notes.txt", "docs/notes.txt")},
                    on_event=events.append)
    assert ask.calls == [["notes.txt"]], ask.calls
    asks = _asks(events)
    assert asks[0]["why"] == ["move"] and asks[0]["place"] == "after_step"
    assert asks[0].get("dests") == ["docs/notes.txt"]
    assert _exists(disk, "notes.txt") and not _exists(disk, "docs/notes.txt")
    assert res.undone == [True]
    _assert_move_no_note(res.tool_results[0], "notes.txt")


def test_c8_poststep_os_rename_answer_yes_leaves_the_file_at_the_new_path(disk):
    events = []
    ask = Ask("yes")
    cmd = "python -c \"import os; os.rename('notes.txt','docs/notes.txt')\""
    res, _, _ = _go(disk, cmd, ask=ask, script={cmd: _rename(disk, "notes.txt", "docs/notes.txt")},
                    on_event=events.append)
    assert ask.calls == [["notes.txt"]]
    assert _asks(events)[0]["why"] == ["move"]
    assert _exists(disk, "docs/notes.txt") and not _exists(disk, "notes.txt")
    assert _text(disk, "docs/notes.txt") == "N\n"


# --- 4. bot-created, cp, mkdir --------------------------------------------------------------------------


def test_c8_a_file_the_bot_created_is_moved_without_a_question(disk):
    ask = Ask("no")
    script = {
        "touch new.txt": (lambda: ((disk.root / "new.txt").write_text("X\n", encoding="utf-8"), "")[1]),
        "mv new.txt docs/": _mv(disk, "new.txt", "docs/"),
    }
    body = [reply(calls=[call("bash", command="touch new.txt")]),
            reply(calls=[call("bash", command="mv new.txt docs/")]),
            reply("done")]
    _go(disk, "touch new.txt", ask=ask, script=script, replies=body, seed={"notes.txt": "N\n"})
    assert ask.calls == []
    assert _exists(disk, "docs/new.txt")


def test_c8_cp_and_mkdir_never_ask(disk):
    ask = Ask("no")
    script = {
        "mkdir docs": (lambda: (disk.root / "docs").mkdir(exist_ok=True) or ""),
        "cp notes.txt docs/notes.txt": (lambda: (
            (disk.root / "docs").mkdir(exist_ok=True),
            (disk.root / "docs" / "notes.txt").write_text(_text(disk, "notes.txt"), encoding="utf-8"),
            "",
        )[-1]),
    }
    body = [reply(calls=[call("bash", command="mkdir docs")]),
            reply(calls=[call("bash", command="cp notes.txt docs/notes.txt")]),
            reply("done")]
    _go(disk, "mkdir docs", ask=ask, script=script, replies=body)
    assert ask.calls == []
    assert _exists(disk, "notes.txt") and _exists(disk, "docs/notes.txt")


# --- 5. chat UI -----------------------------------------------------------------------------------------


def _chat_mv(tmp_path, answers, command="mv notes.txt docs/", seed=None, replies=None):
    """Build a World with an mv side effect under the bot work folder."""
    w = World(tmp_path, ["1", "tidy the folder", *answers, "/quit"],
              replies=replies or [reply(calls=[call("bash", command=command)]), reply("done")],
              seed=seed or {"notes.txt": "N\n", "keep.txt": "K\n"})
    work = w.home / ".localbot" / "bots" / "main" / "work"
    moved = {"done": False}

    def mv_effect():
        src = work / "notes.txt"
        dest = work / "docs" / "notes.txt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            src.rename(dest)
        moved["done"] = True
        return ""

    w.shell_script = {command: mv_effect}
    return w, moved, work


def test_c8_chat_mv_shows_move_question_with_arrow_and_yn_answer_n_keeps_file(tmp_path):
    w, moved, work = _chat_mv(tmp_path, ["n"])
    assert w.run() == 0, w.text
    assert "MOVE" in w.text and "notes.txt" in w.text and "->" in w.text
    assert "docs/notes.txt" in w.text
    assert any("y/n" in p.lower() for p in w.keys.prompts), w.keys.prompts
    assert (work / "notes.txt").exists() and not (work / "docs" / "notes.txt").exists()
    assert moved["done"] is False
    asks = w.events("ask")
    assert asks and asks[0]["why"] == ["move"]
    assert asks[0].get("dests") == ["docs/notes.txt"]


def test_c8_chat_mv_answer_y_moves_the_file(tmp_path):
    w, moved, work = _chat_mv(tmp_path, ["y"])
    assert w.run() == 0, w.text
    assert (work / "docs" / "notes.txt").exists() and not (work / "notes.txt").exists()
    assert moved["done"] is True
    assert [e["answer"] for e in w.events("ask_answer")] == ["yes"]


def test_c8_chat_delete_prompt_and_wording_are_unchanged(tmp_path):
    w = World(tmp_path, ["1", "tidy up", "n", "/quit"],
              replies=[reply(calls=[call("bash", command="rm notes.txt")]), reply("done")],
              seed={"notes.txt": "N\n", "keep.txt": "K\n"})
    assert w.run() == 0, w.text
    assert "wants to DELETE notes.txt (it was here before this chat)" in w.text
    assert any(p.strip().startswith("[d] delete") and "[m] move to _moved\\" in p for p in w.keys.prompts)
    assert chat.WHY_WORDS.get("delete") == "DELETE"
    assert chat.WHY_WORDS.get("empty") == "EMPTY" and chat.WHY_WORDS.get("replace") == "REPLACE"
    assert chat.PROMPT_ASK == "[d] delete  [m] move to _moved\\  [n] no (default) > "


def test_c8_why_words_includes_move_and_keeps_the_old_three():
    assert chat.WHY_WORDS.get("move") == "MOVE"
    assert chat.WHY_WORDS.get("delete") == "DELETE"
    assert set(chat.WHY_WORDS) >= {"delete", "empty", "replace", "move"}
    assert getattr(chat, "PROMPT_ASK_MOVE", None), "need a distinct y/n move prompt"
    assert chat.PROMPT_ASK_MOVE != chat.PROMPT_ASK


# --- 6. harness move_aside must not ask again -----------------------------------------------------------


def test_c8_answering_m_to_a_delete_does_not_ask_about_the_harness_move(tmp_path):
    w = World(tmp_path, ["1", "tidy up", "m", "/quit"],
              replies=[reply(calls=[call("bash", command="rm notes.txt")]), reply("done")],
              seed={"notes.txt": "N\n", "keep.txt": "K\n"})
    assert w.run() == 0, w.text
    asks = w.events("ask")
    assert len(asks) == 1 and asks[0]["why"] == ["delete"]
    assert (w.work / "_moved" / "notes.txt").exists()
    assert "MOVE" not in w.text


# --- 7. plain marker on NEW ask paths -------------------------------------------------------------------


def test_c8_a_plain_marker_in_the_reply_around_a_move_is_in_no_ask_event_or_ask_screen(tmp_path):
    """Model text must not leak into ask / ask_answer events, MOVE question screen lines, or those
    record lines. Normal say/answer lines MAY show the model's words on screen and in the record
    (clarified C8 requirement: brief wording was wrong; narrowed contract stands).
    """
    w, _, _ = _chat_mv(
        tmp_path, ["n"],
        replies=[reply(f"I will now say {MARKER} and move it",
                       calls=[call("bash", command="mv notes.txt docs/")]),
                 reply("ok")])
    assert w.run() == 0, w.text
    asks = w.events("ask")
    assert asks, "expected a move ask"
    assert MARKER not in json.dumps(asks), "an ask event holds the model's own words"
    assert MARKER not in json.dumps(w.events("ask_answer")), "an ask_answer holds the model's own words"
    move_lines = [ln for ln in w.text.splitlines()
                  if "MOVE" in ln or "Move it" in ln or ("->" in ln and "notes" in ln)]
    assert move_lines, "expected MOVE question lines on screen"
    assert all(MARKER not in ln for ln in move_lines), move_lines
    # Record lines that carry ask / ask_answer events must be clean (say lines may hold MARKER)
    for block in w.record_text().splitlines():
        if '"event": "ask"' in block or '"event": "ask_answer"' in block:
            assert MARKER not in block, block
        if "MOVE:" in block or "Move it?" in block:
            assert MARKER not in block, block
    # Prove normal say still carries model text (allowed)
    says = [e for e in w.events() if e.get("event") == "say" and MARKER in (e.get("text") or "")]
    assert says, "normal say should still show the model's words"


# --- 8. edges: last step, next message, same-file allowance, overwrite, fail-closed --------------------


def test_c8_last_step_precheck_answer_no_at_max_turns_1(disk):
    events = []
    ask = Ask("no")
    profile = _profile_max_turns(1)
    body = [reply(calls=[call("bash", command="mv notes.txt docs/")])]
    transport = ScriptedTransport(body)
    ws = disk.ws(dict(SEED))
    shell = FakeShell({"mv notes.txt docs/": _boom})
    res = run_task(profile, provider(transport, profile.provider), "fake:1", "reorganize the folder",
                   ws, shell=shell, undo=_store(disk), ask_delete=ask, on_event=events.append)
    assert ask.calls == [["notes.txt"]]
    assert _asks(events)[0]["why"] == ["move"]
    assert shell.ran == []
    assert _exists(disk, "notes.txt") and not _exists(disk, "docs/notes.txt")
    assert len(transport.bodies) == 1  # no extra provider request
    assert res.reason == "max_steps"


def test_c8_last_step_precheck_answer_yes_at_max_turns_1_moves_and_stops(disk):
    events = []
    ask = Ask("yes")
    profile = _profile_max_turns(1)
    body = [reply(calls=[call("bash", command="mv notes.txt docs/")])]
    transport = ScriptedTransport(body)
    ws = disk.ws(dict(SEED))
    shell = FakeShell({"mv notes.txt docs/": _mv(disk, "notes.txt", "docs/")})
    res = run_task(profile, provider(transport, profile.provider), "fake:1", "reorganize the folder",
                   ws, shell=shell, undo=_store(disk), ask_delete=ask, on_event=events.append)
    assert ask.calls == [["notes.txt"]]
    assert shell.ran == ["mv notes.txt docs/"]
    assert _exists(disk, "docs/notes.txt") and not _exists(disk, "notes.txt")
    assert len(transport.bodies) == 1
    assert res.reason == "max_steps"


def test_c8_last_step_poststep_answer_no_at_max_turns_1(disk):
    events = []
    ask = Ask("no")
    profile = _profile_max_turns(1)
    cmd = "python -c \"import os; os.rename('notes.txt','docs/notes.txt')\""
    body = [reply(calls=[call("bash", command=cmd)])]
    transport = ScriptedTransport(body)
    ws = disk.ws(dict(SEED))
    shell = FakeShell({cmd: _rename(disk, "notes.txt", "docs/notes.txt")})
    res = run_task(profile, provider(transport, profile.provider), "fake:1", "reorganize the folder",
                   ws, shell=shell, undo=_store(disk), ask_delete=ask, on_event=events.append)
    assert ask.calls == [["notes.txt"]]
    assert _asks(events)[0]["place"] == "after_step"
    assert _exists(disk, "notes.txt") and not _exists(disk, "docs/notes.txt")
    assert res.undone == [True]
    assert len(transport.bodies) == 1


def test_c8_last_step_poststep_answer_yes_at_max_turns_1(disk):
    events = []
    ask = Ask("yes")
    profile = _profile_max_turns(1)
    cmd = "python -c \"import os; os.rename('notes.txt','docs/notes.txt')\""
    body = [reply(calls=[call("bash", command=cmd)])]
    transport = ScriptedTransport(body)
    ws = disk.ws(dict(SEED))
    shell = FakeShell({cmd: _rename(disk, "notes.txt", "docs/notes.txt")})
    res = run_task(profile, provider(transport, profile.provider), "fake:1", "reorganize the folder",
                   ws, shell=shell, undo=_store(disk), ask_delete=ask, on_event=events.append)
    assert ask.calls == [["notes.txt"]]
    assert _exists(disk, "docs/notes.txt") and not _exists(disk, "notes.txt")
    assert len(transport.bodies) == 1
    assert res.reason == "max_steps"


def test_c8_next_message_asks_again_for_same_file_now_at_relocated_path(disk):
    ask = Ask(answers=["yes", "no"])
    script = {
        "mv notes.txt a/": _mv(disk, "notes.txt", "a/"),
        "mv a/notes.txt b/": _mv(disk, "a/notes.txt", "b/"),
    }
    body = [reply(calls=[call("bash", command="mv notes.txt a/")]),
            reply("first done"),
            reply(calls=[call("bash", command="mv a/notes.txt b/")]),
            reply("second done")]
    ws = disk.ws(dict(SEED))
    shell = FakeShell(script)
    run_task(C4, provider(ScriptedTransport(body), C4.provider), "fake:1", ["first", "second"],
             ws, shell=shell, undo=_store(disk), ask_delete=ask)
    assert ask.calls == [["notes.txt"], ["a/notes.txt"]]
    assert _exists(disk, "a/notes.txt") and not _exists(disk, "b/notes.txt")


def test_c8_allowance_is_per_step_even_for_the_same_relocated_file(disk):
    """Yes on notes->docs/ does not let the next step move docs/notes.txt without asking again."""
    ask = Ask(answers=["yes", "no"])
    script = {
        "mv notes.txt docs/": _mv(disk, "notes.txt", "docs/"),
        "mv docs/notes.txt other/": _mv(disk, "docs/notes.txt", "other/"),
    }
    body = [reply(calls=[call("bash", command="mv notes.txt docs/")]),
            reply(calls=[call("bash", command="mv docs/notes.txt other/")]),
            reply("done")]
    _go(disk, "mv notes.txt docs/", ask=ask, script=script, replies=body)
    assert ask.calls == [["notes.txt"], ["docs/notes.txt"]]
    assert _exists(disk, "docs/notes.txt")
    assert not _exists(disk, "other/notes.txt")


def test_c8_yes_to_moving_source_does_not_approve_overwriting_a_protected_dest(disk):
    """mv keep.txt notes.txt: yes on moving keep must not silently destroy notes."""
    events = []
    ask = Ask(answers=["yes", "no"])

    def overwrite():
        data = (disk.root / "keep.txt").read_bytes()
        (disk.root / "notes.txt").write_bytes(data)
        (disk.root / "keep.txt").unlink()
        return ""

    _go(disk, "mv keep.txt notes.txt", ask=ask,
        script={"mv keep.txt notes.txt": overwrite}, on_event=events.append)
    assert _text(disk, "notes.txt") == "N\n", "protected dest was destroyed without its own OK"
    whys = [e["why"] for e in _asks(events)]
    assert whys, "must ask at least once"
    flat = [w for group in whys for w in group]
    if len(ask.calls) >= 2:
        assert "replace" in flat or "delete" in flat
        assert any("notes.txt" in (e.get("paths") or []) for e in _asks(events))
    else:
        assert _exists(disk, "notes.txt")
        assert _text(disk, "notes.txt") == "N\n"


def test_c8_unknown_ask_answer_on_a_move_is_fail_closed_no(disk):
    events = []
    ask = Ask("sure")
    res, _, shell = _go(disk, "mv notes.txt docs/", ask=ask, on_event=events.append)
    assert ask.calls == [["notes.txt"]]
    assert shell.ran == []
    assert _exists(disk, "notes.txt") and not _exists(disk, "docs/notes.txt")
    assert res.summary().get("ask_errors", 0) >= 1


def test_c8_yes_on_a_move_does_not_authorize_a_sibling_delete_in_the_same_command(disk):
    """Mixed mv + rm: yes is move-only; keep must not be deleted on yes."""
    ask = Ask("yes")
    cmd = "mv notes.txt docs/; rm keep.txt"

    def both():
        _mv(disk, "notes.txt", "docs/")()
        (disk.root / "keep.txt").unlink()
        return ""

    _go(disk, cmd, ask=ask, script={cmd: both})
    assert _exists(disk, "keep.txt"), "yes on a move must not authorize a delete"


# --- 9. Driver: PROMPT_ASK_MOVE recognition (scope-extension if missing) --------------------------------


def _load_ladder():
    import importlib.util
    import sys
    SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "chat_ladder.py"
    name = "chat_ladder_c8_" + str(id(SCRIPT))
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    ladder = importlib.util.module_from_spec(spec)
    sys.modules[name] = ladder
    spec.loader.exec_module(ladder)
    return ladder


def test_c8_ladder_driver_answers_n_to_prompt_ask_still():
    ladder = _load_ladder()
    d = ladder.Driver("clean up this folder")
    assert d(chat.PROMPT_ASK) == "n"
    assert d.asked == 1


def test_c8_ladder_driver_must_answer_n_to_prompt_ask_move():
    """Distinct move y/n prompt: Driver answers n (same as PROMPT_ASK)."""
    assert getattr(chat, "PROMPT_ASK_MOVE", None), "chat.PROMPT_ASK_MOVE must exist"
    assert chat.PROMPT_ASK_MOVE != chat.PROMPT_ASK
    ladder = _load_ladder()
    d = ladder.Driver("clean up this folder")
    got = d(chat.PROMPT_ASK_MOVE)
    assert got == "n", f"Driver returned {got!r} for PROMPT_ASK_MOVE"
    assert d.asked == 1


# --- 10. delete pin -------------------------------------------------------------------------------------


def test_c8_delete_precheck_no_still_refuses_exactly_as_today(disk):
    ask = Ask("no")
    res, _, shell = _go(disk, "rm notes.txt", ask=ask,
                        script={"rm notes.txt": (lambda: (disk.root / "notes.txt").unlink() or "")})
    assert ask.calls == [["notes.txt"]]
    assert _exists(disk, "notes.txt") and shell.ran == []
    assert res.tool_results[0] == HELD
