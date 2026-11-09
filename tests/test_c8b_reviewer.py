"""C8b (reviewer chat 26): a yes to a move keeps the file guarded; protection follows its real new path.

R1, R7: `mv notes.txt docs` (no slash, folder exists or made in the same command): the shown destination
was `docs`, so protection went to `docs` and `rm docs/notes.txt` was not asked. R2: yes to move, the same
command then deleted it: no question. R3: the file landed somewhere else: unprotected. R4: a folder rename
holding a protected file IS asked after the step. R8: a yes is for one step only.
"""
from __future__ import annotations
import pytest
from tests.test_c8_ask_before_moves import (Ask, _go, _asks, _mv, _rename, _exists, _text, SEED, HELD, C4)
from tests.fakes import reply, call


def _rm(disk, rel):
    def run():
        p = disk.root / rel
        if p.exists():
            p.unlink()
        return ""
    return run


def _both(*fs):
    def run():
        for f in fs:
            f()
        return ""
    return run


SEED_DOCS = dict(SEED, **{"docs/readme.md": "R\n"})


# R1: mv into an EXISTING folder without a trailing slash; yes; then rm the file at its real new path.
def test_r1_existing_dir_no_slash_then_rm_new_path_asks(disk):
    events = []
    ask = Ask(answers=["yes", "no"])
    script = {"mv notes.txt docs": _mv(disk, "notes.txt", "docs/"),
              "rm docs/notes.txt": _rm(disk, "docs/notes.txt")}
    body = [reply(calls=[call("bash", command="mv notes.txt docs")]),
            reply(calls=[call("bash", command="rm docs/notes.txt")]), reply("done")]
    _go(disk, "mv notes.txt docs", ask=ask, seed=SEED_DOCS, script=script, replies=body,
        on_event=events.append)
    asks = _asks(events)
    print("ASKS", asks)
    assert len(asks) == 2, asks
    assert _exists(disk, "docs/notes.txt"), "person's file deleted without a delete question"


# R2: yes to a move, but the same command also deletes the moved file at its destination.
def test_r2_yes_to_move_then_same_command_deletes_it(disk):
    events = []
    ask = Ask(answers=["yes", "no"])
    cmd = "mv notes.txt docs/ && rm docs/notes.txt"
    script = {cmd: _both(_mv(disk, "notes.txt", "docs/"), _rm(disk, "docs/notes.txt"))}
    res, _, _ = _go(disk, cmd, ask=ask, script=script, on_event=events.append)
    print("ASKS", _asks(events), res.tool_results)
    assert _exists(disk, "notes.txt") or _exists(disk, "docs/notes.txt"), \
        "person said yes to MOVE, file is gone with no delete question"


# R3: yes to `mv notes.txt docs/` but the command (or a python line in it) puts the file somewhere else.
def test_r3_yes_but_file_lands_elsewhere_stays_protected(disk):
    events = []
    ask = Ask(answers=["yes", "no"])
    cmd = "mv notes.txt docs/ ; true"
    script = {cmd: _rename(disk, "notes.txt", "other/n.txt"),
              "rm other/n.txt": _rm(disk, "other/n.txt")}
    body = [reply(calls=[call("bash", command=cmd)]),
            reply(calls=[call("bash", command="rm other/n.txt")]), reply("done")]
    _go(disk, cmd, ask=ask, script=script, replies=body, on_event=events.append)
    print("ASKS", _asks(events))
    assert _exists(disk, "other/n.txt") or _exists(disk, "notes.txt"), "file lost without a delete question"


# R4: folder rename holding a protected file (lokal listed it as 'suspected not asked').
def test_r4_folder_rename_is_asked_post_step(disk):
    events = []
    ask = Ask("no")
    script = {"mv docs docs2": _rename(disk, "docs", "docs2")}
    res, _, _ = _go(disk, "mv docs docs2", ask=ask, seed=SEED_DOCS, script=script, on_event=events.append)
    asks = _asks(events)
    print("ASKS", asks, res.tool_results)
    assert asks and asks[0]["why"] == ["move"]
    assert _exists(disk, "docs/readme.md")


# R5: exam rules (no hook): mixed move + delete is held exactly as before (no command run).
def test_r5_no_hook_mixed_move_and_delete_is_held(disk):
    cmd = "mv notes.txt docs/ && rm keep.txt"
    res, _, shell = _go(disk, cmd, script={cmd: lambda: (_ for _ in ()).throw(AssertionError("ran"))})
    assert res.tool_results[0] == HELD
    assert shell.ran == []


# R6: chat-level mixed delete+move question answered delete -> what happens to the move?
def test_r6_mixed_delete_and_move_answer_delete(disk):
    events = []
    ask = Ask(answers=["delete", "no"])
    cmd = "rm old_draft.txt && mv notes.txt docs/"
    script = {cmd: _both(_rm(disk, "old_draft.txt"), _mv(disk, "notes.txt", "docs/"))}
    res, _, _ = _go(disk, cmd, ask=ask, script=script, on_event=events.append)
    asks = _asks(events)
    print("ASKS", [(a["place"], a["paths"], a["why"]) for a in asks], res.tool_results)
    assert _exists(disk, "notes.txt"), "move happened without a yes"


# R7: the likely real pattern: make the folder and move into it in one command (no trailing slash).
def test_r7_mkdir_and_mv_into_new_folder_then_rm_asks(disk):
    events = []
    ask = Ask(answers=["yes", "no"])
    cmd = "mkdir docs && mv notes.txt docs"
    script = {cmd: _mv(disk, "notes.txt", "docs/"), "rm docs/notes.txt": _rm(disk, "docs/notes.txt")}
    body = [reply(calls=[call("bash", command=cmd)]),
            reply(calls=[call("bash", command="rm docs/notes.txt")]), reply("done")]
    _go(disk, cmd, ask=ask, script=script, replies=body, on_event=events.append)
    asks = _asks(events)
    print("ASKS", [(a["place"], a["paths"], a["why"], a.get("dests")) for a in asks])
    assert [a["why"] for a in asks] == [["move"], ["delete"]]
    assert _exists(disk, "docs/notes.txt"), "deleted without a delete question"


# R8: a yes is for that one step: the move did not happen (command did nothing), the NEXT step moves the
# same file another way -> asked again.
def test_r8_move_yes_does_not_carry_to_the_next_step(disk):
    events = []
    ask = Ask(answers=["yes", "no"])
    py = "python -c \"import os; os.rename('notes.txt','docs/notes.txt')\""
    script = {"mv notes.txt docs/": lambda: "", py: _rename(disk, "notes.txt", "docs/notes.txt")}
    body = [reply(calls=[call("bash", command="mv notes.txt docs/")]),
            reply(calls=[call("bash", command=py)]), reply("done")]
    _go(disk, "mv notes.txt docs/", ask=ask, script=script, replies=body, on_event=events.append)
    asks = _asks(events)
    assert [(a["place"], a["why"]) for a in asks] == [("precheck", ["move"]), ("after_step", ["move"])]
    assert _exists(disk, "notes.txt") and not _exists(disk, "docs/notes.txt")
