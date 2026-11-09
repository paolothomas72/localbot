"""BRIEF C10: close the last ways to change the person's files without asking (C8 clean-up; product / chat only).

1. Mixed note: a step with both a delete and a move question tells the model what happened to every file
   (deleted / moved aside / moved / kept / not moved), for every answer, at the pre-check and after the step.
2. Real destination: the move question shows where ``mv`` would really put the file (an existing folder: inside
   it; ``-T``: a rename; ``-t`` / several sources / a trailing slash: a folder; an existing file: a rename over
   it; nothing there yet: said so plainly, "docs (a folder or a new name)").
3. A yes is for the shown place only: a file that lands elsewhere is asked about again after the step.
4. A link put at an existing file's name (``ln -sf``), an unpack (``tar x``, ``unzip``) or ``dd`` over it: the
   overwrite ask (``replace``), before the command where its text names the file, after the step otherwise.
5. ``chmod`` of an existing file: a ``permissions`` ask (y/n), before the command; a file the bot made: no ask.
No hook (exam / gate / LONG): exactly as on 0c8009f (goldens below). C2 / C5 / C8 wording unchanged.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import stat
from pathlib import Path

import pytest

from localbot import chat
from localbot.guard.workspace_fs import set_case_sensitive
from localbot.loop import run_task
from tests.fakes import FakeShell, ScriptedTransport, call, provider, reply
from tests.test_c8_ask_before_moves import (C4, HELD, MARKER, PRODUCT, SEED, Ask, _asks, _assert_move_no_note,
                                            _boom, _core, _exists, _go, _load_ladder, _mv, _profile_max_turns,
                                            _rename, _store, _text)
from tests.test_chat import World

PROFILES = {"core4": C4, "core4_product": PRODUCT}
C10_SEED = dict(SEED, **{"other.txt": "O\n", "run.sh": "#!/bin/sh\necho hi\n", "docs/readme.md": "R\n"})
AFTER = ("notes.txt", "docs/notes.txt", "newname.txt", "old_draft.txt", "run.sh")
CHMOD = "chmod +x run.sh"
TAR = "tar -xf a.tar"
UNZIP = "unzip -o a.zip"
LN = "ln -sf other.txt notes.txt"
DD = "dd of=notes.txt if=other.txt bs=1 count=2"  # the guard refuses "dd if=" first (harm rule); this order gets past it
MIXED = "rm old_draft.txt && mv notes.txt docs/"
MIXED_PY = "python -c \"import os; os.remove('old_draft.txt'); os.rename('notes.txt', 'docs/notes.txt')\""
UNKNOWN = " (a folder or a new name)"

# Recorded on 0c8009f (this worktree's tip before any C10 change, `git status` clean) by the scratch script
# c10_golden_0c8009f.py (2026-11-05): core4 / core4_product with undo, NO ask hook, the FakeShell effects of
# ``C10_CASES`` below. ``core`` = ``_core(res)``, ``ran`` = the shell's commands, ``after`` = (exists, is a link)
# for every path in ``AFTER``.
GOLDEN_0C8009F = json.loads(r'''{"chmod":{"command":"chmod +x run.sh","core4":{"after":{"docs/notes.txt":[false,false],"newname.txt":[false,false],"notes.txt":[true,false],"old_draft.txt":[true,false],"run.sh":[true,false]},"core":{"delete_attempt":false,"messages_sha256":"13f7b42d81c117395970b797816e7c2f8c832b5d420f8d9a2a41cad21b968159","n_messages":5,"ran":[true],"reason":"done","summary":{"arg_errors":[],"arg_extra":[],"call_source":["native","none"],"compactions":[],"container_scans":[],"delete_attempt":false,"error":"","eval_count":60,"false_success":[],"git_changed":false,"no_call_reason":null,"nudges":{"cut":0,"parse":0,"stuck":0},"reason":"done","text_call_attempts":[],"tool_errors":[],"tools":["bash"],"veto_texts":[],"vetoes":0},"tool_calls":[["bash",{"command":"chmod +x run.sh"}]],"tool_results":[""],"undone":[false],"vetoes":0},"ran":["chmod +x run.sh"]},"core4_product":{"after":{"docs/notes.txt":[false,false],"newname.txt":[false,false],"notes.txt":[true,false],"old_draft.txt":[true,false],"run.sh":[true,false]},"core":{"delete_attempt":false,"messages_sha256":"13f7b42d81c117395970b797816e7c2f8c832b5d420f8d9a2a41cad21b968159","n_messages":5,"ran":[true],"reason":"done","summary":{"arg_errors":[],"arg_extra":[],"call_source":["native","none"],"compactions":[],"container_scans":[],"delete_attempt":false,"error":"","eval_count":60,"false_success":[],"git_changed":false,"no_call_reason":null,"nudges":{"cut":0,"parse":0,"stuck":0},"reason":"done","text_call_attempts":[],"tool_errors":[],"tools":["bash"],"veto_texts":[],"vetoes":0},"tool_calls":[["bash",{"command":"chmod +x run.sh"}]],"tool_results":[""],"undone":[false],"vetoes":0},"ran":["chmod +x run.sh"]}},"dd_of":{"command":"dd of=notes.txt if=other.txt bs=1 count=2","core4":{"after":{"docs/notes.txt":[false,false],"newname.txt":[false,false],"notes.txt":[true,false],"old_draft.txt":[true,false],"run.sh":[true,false]},"core":{"delete_attempt":false,"messages_sha256":"e84482e469a0660cf8b6d559700bb6756c12b065030211e5bf547b6dd9ec3683","n_messages":5,"ran":[true],"reason":"done","summary":{"arg_errors":[],"arg_extra":[],"call_source":["native","none"],"compactions":[],"container_scans":[],"delete_attempt":false,"error":"","eval_count":60,"false_success":[],"git_changed":false,"no_call_reason":null,"nudges":{"cut":0,"parse":0,"stuck":0},"reason":"done","text_call_attempts":[],"tool_errors":[],"tools":["bash"],"veto_texts":[],"vetoes":0},"tool_calls":[["bash",{"command":"dd of=notes.txt if=other.txt bs=1 count=2"}]],"tool_results":[""],"undone":[false],"vetoes":0},"ran":["dd of=notes.txt if=other.txt bs=1 count=2"]},"core4_product":{"after":{"docs/notes.txt":[false,false],"newname.txt":[false,false],"notes.txt":[true,false],"old_draft.txt":[true,false],"run.sh":[true,false]},"core":{"delete_attempt":false,"messages_sha256":"e84482e469a0660cf8b6d559700bb6756c12b065030211e5bf547b6dd9ec3683","n_messages":5,"ran":[true],"reason":"done","summary":{"arg_errors":[],"arg_extra":[],"call_source":["native","none"],"compactions":[],"container_scans":[],"delete_attempt":false,"error":"","eval_count":60,"false_success":[],"git_changed":false,"no_call_reason":null,"nudges":{"cut":0,"parse":0,"stuck":0},"reason":"done","text_call_attempts":[],"tool_errors":[],"tools":["bash"],"veto_texts":[],"vetoes":0},"tool_calls":[["bash",{"command":"dd of=notes.txt if=other.txt bs=1 count=2"}]],"tool_results":[""],"undone":[false],"vetoes":0},"ran":["dd of=notes.txt if=other.txt bs=1 count=2"]}},"ln_sf":{"command":"ln -sf other.txt notes.txt","core4":{"after":{"docs/notes.txt":[false,false],"newname.txt":[false,false],"notes.txt":[true,false],"old_draft.txt":[true,false],"run.sh":[true,false]},"core":{"delete_attempt":true,"messages_sha256":"d9d5a2e3527f2ec5dc28e2b1fb45b3e012522058b8fe77ec13d6ffa752bfce21","n_messages":5,"ran":[true],"reason":"done","summary":{"arg_errors":[],"arg_extra":[],"call_source":["native","none"],"compactions":[],"container_scans":[{"command":"ln -sf other.txt notes.txt","hard_links":[],"refused_names":[],"reparse":["notes.txt"]}],"delete_attempt":true,"error":"","eval_count":60,"false_success":[],"git_changed":false,"no_call_reason":"vetoed","nudges":{"cut":0,"parse":0,"stuck":0},"reason":"done","text_call_attempts":[],"tool_errors":[],"tools":["bash"],"veto_texts":["ERROR: harness: Deleting existing files needs the user's OK. Don't delete. Tell the user which files you would remove and ask whether to delete or move them."],"vetoes":1},"tool_calls":[["bash",{"command":"ln -sf other.txt notes.txt"}]],"tool_results":["ERROR: harness: Deleting existing files needs the user's OK. Don't delete. Tell the user which files you would remove and ask whether to delete or move them."],"undone":[true],"vetoes":1},"ran":["ln -sf other.txt notes.txt"]},"core4_product":{"after":{"docs/notes.txt":[false,false],"newname.txt":[false,false],"notes.txt":[true,false],"old_draft.txt":[true,false],"run.sh":[true,false]},"core":{"delete_attempt":true,"messages_sha256":"d9d5a2e3527f2ec5dc28e2b1fb45b3e012522058b8fe77ec13d6ffa752bfce21","n_messages":5,"ran":[true],"reason":"done","summary":{"arg_errors":[],"arg_extra":[],"call_source":["native","none"],"compactions":[],"container_scans":[{"command":"ln -sf other.txt notes.txt","hard_links":[],"refused_names":[],"reparse":["notes.txt"]}],"delete_attempt":true,"error":"","eval_count":60,"false_success":[],"git_changed":false,"no_call_reason":"vetoed","nudges":{"cut":0,"parse":0,"stuck":0},"reason":"done","text_call_attempts":[],"tool_errors":[],"tools":["bash"],"veto_texts":["ERROR: harness: Deleting existing files needs the user's OK. Don't delete. Tell the user which files you would remove and ask whether to delete or move them."],"vetoes":1},"tool_calls":[["bash",{"command":"ln -sf other.txt notes.txt"}]],"tool_results":["ERROR: harness: Deleting existing files needs the user's OK. Don't delete. Tell the user which files you would remove and ask whether to delete or move them."],"undone":[true],"vetoes":1},"ran":["ln -sf other.txt notes.txt"]}},"mixed":{"command":"rm old_draft.txt && mv notes.txt docs/","core4":{"after":{"docs/notes.txt":[false,false],"newname.txt":[false,false],"notes.txt":[true,false],"old_draft.txt":[true,false],"run.sh":[true,false]},"core":{"delete_attempt":true,"messages_sha256":"244338656d440f6089ad4487a0323d6c1356fff3cb39fb87d8d6de0279ec9d17","n_messages":5,"ran":[false],"reason":"done","summary":{"arg_errors":[],"arg_extra":[],"call_source":["native","none"],"compactions":[],"container_scans":[],"delete_attempt":true,"error":"","eval_count":60,"false_success":[],"git_changed":false,"no_call_reason":"vetoed","nudges":{"cut":0,"parse":0,"stuck":0},"reason":"done","text_call_attempts":[],"tool_errors":[],"tools":["bash"],"veto_texts":["ERROR: harness: Deleting existing files needs the user's OK. Don't delete. Tell the user which files you would remove and ask whether to delete or move them."],"vetoes":1},"tool_calls":[["bash",{"command":"rm old_draft.txt && mv notes.txt docs/"}]],"tool_results":["ERROR: harness: Deleting existing files needs the user's OK. Don't delete. Tell the user which files you would remove and ask whether to delete or move them."],"undone":[false],"vetoes":1},"ran":[]},"core4_product":{"after":{"docs/notes.txt":[false,false],"newname.txt":[false,false],"notes.txt":[true,false],"old_draft.txt":[true,false],"run.sh":[true,false]},"core":{"delete_attempt":true,"messages_sha256":"244338656d440f6089ad4487a0323d6c1356fff3cb39fb87d8d6de0279ec9d17","n_messages":5,"ran":[false],"reason":"done","summary":{"arg_errors":[],"arg_extra":[],"call_source":["native","none"],"compactions":[],"container_scans":[],"delete_attempt":true,"error":"","eval_count":60,"false_success":[],"git_changed":false,"no_call_reason":"vetoed","nudges":{"cut":0,"parse":0,"stuck":0},"reason":"done","text_call_attempts":[],"tool_errors":[],"tools":["bash"],"veto_texts":["ERROR: harness: Deleting existing files needs the user's OK. Don't delete. Tell the user which files you would remove and ask whether to delete or move them."],"vetoes":1},"tool_calls":[["bash",{"command":"rm old_draft.txt && mv notes.txt docs/"}]],"tool_results":["ERROR: harness: Deleting existing files needs the user's OK. Don't delete. Tell the user which files you would remove and ask whether to delete or move them."],"undone":[false],"vetoes":1},"ran":[]}},"mv_into_dir":{"command":"mv notes.txt docs","core4":{"after":{"docs/notes.txt":[true,false],"newname.txt":[false,false],"notes.txt":[false,false],"old_draft.txt":[true,false],"run.sh":[true,false]},"core":{"delete_attempt":false,"messages_sha256":"192c5f6cfef24cff746b97d6762260cc65c64b306abae089a9ce8d45e59db1e5","n_messages":5,"ran":[true],"reason":"done","summary":{"arg_errors":[],"arg_extra":[],"call_source":["native","none"],"compactions":[],"container_scans":[],"delete_attempt":false,"error":"","eval_count":60,"false_success":[],"git_changed":false,"no_call_reason":null,"nudges":{"cut":0,"parse":0,"stuck":0},"reason":"done","text_call_attempts":[],"tool_errors":[],"tools":["bash"],"veto_texts":[],"vetoes":0},"tool_calls":[["bash",{"command":"mv notes.txt docs"}]],"tool_results":[""],"undone":[false],"vetoes":0},"ran":["mv notes.txt docs"]},"core4_product":{"after":{"docs/notes.txt":[true,false],"newname.txt":[false,false],"notes.txt":[false,false],"old_draft.txt":[true,false],"run.sh":[true,false]},"core":{"delete_attempt":false,"messages_sha256":"192c5f6cfef24cff746b97d6762260cc65c64b306abae089a9ce8d45e59db1e5","n_messages":5,"ran":[true],"reason":"done","summary":{"arg_errors":[],"arg_extra":[],"call_source":["native","none"],"compactions":[],"container_scans":[],"delete_attempt":false,"error":"","eval_count":60,"false_success":[],"git_changed":false,"no_call_reason":null,"nudges":{"cut":0,"parse":0,"stuck":0},"reason":"done","text_call_attempts":[],"tool_errors":[],"tools":["bash"],"veto_texts":[],"vetoes":0},"tool_calls":[["bash",{"command":"mv notes.txt docs"}]],"tool_results":[""],"undone":[false],"vetoes":0},"ran":["mv notes.txt docs"]}},"mv_rename":{"command":"mv notes.txt newname.txt","core4":{"after":{"docs/notes.txt":[false,false],"newname.txt":[true,false],"notes.txt":[false,false],"old_draft.txt":[true,false],"run.sh":[true,false]},"core":{"delete_attempt":false,"messages_sha256":"ad42244b31f8b96af0b5c2ff10b3bdb6be1e4687bff1f7aff5ff72cf80eeb980","n_messages":5,"ran":[true],"reason":"done","summary":{"arg_errors":[],"arg_extra":[],"call_source":["native","none"],"compactions":[],"container_scans":[],"delete_attempt":false,"error":"","eval_count":60,"false_success":[],"git_changed":false,"no_call_reason":null,"nudges":{"cut":0,"parse":0,"stuck":0},"reason":"done","text_call_attempts":[],"tool_errors":[],"tools":["bash"],"veto_texts":[],"vetoes":0},"tool_calls":[["bash",{"command":"mv notes.txt newname.txt"}]],"tool_results":[""],"undone":[false],"vetoes":0},"ran":["mv notes.txt newname.txt"]},"core4_product":{"after":{"docs/notes.txt":[false,false],"newname.txt":[true,false],"notes.txt":[false,false],"old_draft.txt":[true,false],"run.sh":[true,false]},"core":{"delete_attempt":false,"messages_sha256":"ad42244b31f8b96af0b5c2ff10b3bdb6be1e4687bff1f7aff5ff72cf80eeb980","n_messages":5,"ran":[true],"reason":"done","summary":{"arg_errors":[],"arg_extra":[],"call_source":["native","none"],"compactions":[],"container_scans":[],"delete_attempt":false,"error":"","eval_count":60,"false_success":[],"git_changed":false,"no_call_reason":null,"nudges":{"cut":0,"parse":0,"stuck":0},"reason":"done","text_call_attempts":[],"tool_errors":[],"tools":["bash"],"veto_texts":[],"vetoes":0},"tool_calls":[["bash",{"command":"mv notes.txt newname.txt"}]],"tool_results":[""],"undone":[false],"vetoes":0},"ran":["mv notes.txt newname.txt"]}},"tar_x":{"command":"tar -xf a.tar","core4":{"after":{"docs/notes.txt":[false,false],"newname.txt":[false,false],"notes.txt":[true,false],"old_draft.txt":[true,false],"run.sh":[true,false]},"core":{"delete_attempt":false,"messages_sha256":"c2d9c84b6178e7f0cfb5e97c0cb28033db78152b1754ef6090141ffeb764cd28","n_messages":5,"ran":[true],"reason":"done","summary":{"arg_errors":[],"arg_extra":[],"call_source":["native","none"],"compactions":[],"container_scans":[],"delete_attempt":false,"error":"","eval_count":60,"false_success":[],"git_changed":false,"no_call_reason":null,"nudges":{"cut":0,"parse":0,"stuck":0},"reason":"done","text_call_attempts":[],"tool_errors":[],"tools":["bash"],"veto_texts":[],"vetoes":0},"tool_calls":[["bash",{"command":"tar -xf a.tar"}]],"tool_results":[""],"undone":[false],"vetoes":0},"ran":["tar -xf a.tar"]},"core4_product":{"after":{"docs/notes.txt":[false,false],"newname.txt":[false,false],"notes.txt":[true,false],"old_draft.txt":[true,false],"run.sh":[true,false]},"core":{"delete_attempt":false,"messages_sha256":"c2d9c84b6178e7f0cfb5e97c0cb28033db78152b1754ef6090141ffeb764cd28","n_messages":5,"ran":[true],"reason":"done","summary":{"arg_errors":[],"arg_extra":[],"call_source":["native","none"],"compactions":[],"container_scans":[],"delete_attempt":false,"error":"","eval_count":60,"false_success":[],"git_changed":false,"no_call_reason":null,"nudges":{"cut":0,"parse":0,"stuck":0},"reason":"done","text_call_attempts":[],"tool_errors":[],"tools":["bash"],"veto_texts":[],"vetoes":0},"tool_calls":[["bash",{"command":"tar -xf a.tar"}]],"tool_results":[""],"undone":[false],"vetoes":0},"ran":["tar -xf a.tar"]}},"unzip_o":{"command":"unzip -o a.zip","core4":{"after":{"docs/notes.txt":[false,false],"newname.txt":[false,false],"notes.txt":[true,false],"old_draft.txt":[true,false],"run.sh":[true,false]},"core":{"delete_attempt":false,"messages_sha256":"442a26e21e56fdc84dcc348fb12a2f34af08be7a075d96d45f7a730a2d445898","n_messages":5,"ran":[true],"reason":"done","summary":{"arg_errors":[],"arg_extra":[],"call_source":["native","none"],"compactions":[],"container_scans":[],"delete_attempt":false,"error":"","eval_count":60,"false_success":[],"git_changed":false,"no_call_reason":null,"nudges":{"cut":0,"parse":0,"stuck":0},"reason":"done","text_call_attempts":[],"tool_errors":[],"tools":["bash"],"veto_texts":[],"vetoes":0},"tool_calls":[["bash",{"command":"unzip -o a.zip"}]],"tool_results":[""],"undone":[false],"vetoes":0},"ran":["unzip -o a.zip"]},"core4_product":{"after":{"docs/notes.txt":[false,false],"newname.txt":[false,false],"notes.txt":[true,false],"old_draft.txt":[true,false],"run.sh":[true,false]},"core":{"delete_attempt":false,"messages_sha256":"442a26e21e56fdc84dcc348fb12a2f34af08be7a075d96d45f7a730a2d445898","n_messages":5,"ran":[true],"reason":"done","summary":{"arg_errors":[],"arg_extra":[],"call_source":["native","none"],"compactions":[],"container_scans":[],"delete_attempt":false,"error":"","eval_count":60,"false_success":[],"git_changed":false,"no_call_reason":null,"nudges":{"cut":0,"parse":0,"stuck":0},"reason":"done","text_call_attempts":[],"tool_errors":[],"tools":["bash"],"veto_texts":[],"vetoes":0},"tool_calls":[["bash",{"command":"unzip -o a.zip"}]],"tool_results":[""],"undone":[false],"vetoes":0},"ran":["unzip -o a.zip"]}}}''')


# --- helpers ---------------------------------------------------------------------------------------------


def _write(disk, rel: str, data: bytes):
    def run():
        (disk.root / rel).write_bytes(data)
        return ""
    return run


def _symlink_swap(disk, rel: str, target: str):
    def run():
        p = disk.root / rel
        p.unlink()
        os.symlink(target, p)
        return ""
    return run


def _rm(disk, rel: str):
    def run():
        (disk.root / rel).unlink()
        return ""
    return run


def _both(*fs):
    def run():
        for f in fs:
            f()
        return ""
    return run


def _read_only(path):
    """A chmod the host can see: the read-only flag (the one mode bit Windows keeps)."""
    def run():
        os.chmod(path, stat.S_IREAD)
        return ""
    return run


def _writable(path) -> None:
    os.chmod(path, stat.S_IREAD | stat.S_IWRITE)


def _mode(disk, rel: str) -> int:
    return stat.S_IMODE((disk.root / rel).stat().st_mode)


C10_CASES = {
    "ln_sf": (LN, lambda d: _symlink_swap(d, "notes.txt", "other.txt")),
    "tar_x": (TAR, lambda d: _write(d, "notes.txt", b"T\n")),
    "unzip_o": (UNZIP, lambda d: _write(d, "notes.txt", b"T\n")),
    "dd_of": (DD, lambda d: _write(d, "notes.txt", b"\0\0")),
    "chmod": (CHMOD, lambda d: (lambda: "")),
    "mv_into_dir": ("mv notes.txt docs", lambda d: _mv(d, "notes.txt", "docs/")),
    "mv_rename": ("mv notes.txt newname.txt", lambda d: _rename(d, "notes.txt", "newname.txt")),
    "mixed": (MIXED, lambda d: _both(_rm(d, "old_draft.txt"), _mv(d, "notes.txt", "docs/"))),
}


def _outcomes(note: str) -> dict[str, str]:
    """The per-file outcomes a C10 note names: ``What happened: path: outcome; path: outcome.``"""
    m = re.search(r"What happened: (.*?)\.?$", note.strip())
    assert m, note
    return dict(tuple(c.rsplit(": ", 1)) for c in m.group(1).split("; "))


def _disk_agrees(disk, want: dict[str, str]) -> None:
    for p, o in want.items():
        if o.startswith("kept") or o in ("not moved", "permissions unchanged"):
            assert _exists(disk, p), (p, o)
        elif o.startswith("moved aside to ") or o.startswith("moved to "):
            new = o.split(" to ", 1)[1]
            assert not _exists(disk, p) and _exists(disk, new), (p, o)
        elif o == "deleted":
            assert not _exists(disk, p), (p, o)
        else:
            raise AssertionError(f"unknown outcome {o!r} for {p}")


def _places(events):
    return [(a["place"], a["why"]) for a in _asks(events)]


def _chat(tmp_path, keys, command, effect, seed=None, replies=None):
    """A World whose shell runs ``effect(work)`` for ``command``; the person types ``keys`` at the questions."""
    w = World(tmp_path, ["1", "tidy up", *keys, "/quit"],
              replies=replies or [reply(calls=[call("bash", command=command)]), reply("done")],
              seed=seed or C10_SEED)
    work = w.home / ".localbot" / "bots" / "main" / "work"
    w.shell_script = {command: effect(work)}
    return w, work


# --- 0. no hook: every route runs exactly as on 0c8009f ------------------------------------------------


@pytest.mark.parametrize("pname", sorted(PROFILES))
@pytest.mark.parametrize("key", sorted(C10_CASES))
def test_c10_without_ask_hook_every_route_runs_exactly_as_on_0c8009f(disk, key, pname):
    """Exam / gate / LONG: not a proof of new behaviour, a pin of the old one (both profiles, no hook)."""
    cmd, effect = C10_CASES[key]
    res, _, shell = _go(disk, cmd, ask=None, seed=C10_SEED, script={cmd: effect(disk)}, profile=PROFILES[pname])
    want = GOLDEN_0C8009F[key][pname]
    assert _core(res) == want["core"]
    assert list(shell.ran) == want["ran"]
    assert {rel: [_exists(disk, rel), os.path.islink(disk.root / rel)] for rel in AFTER} == want["after"]


# --- 1. the mixed note -----------------------------------------------------------------------------------


@pytest.mark.parametrize("answers,want,ran", [
    (["no"], {"old_draft.txt": "kept (not deleted)", "notes.txt": "not moved"}, []),
    (["move"], {"old_draft.txt": "moved aside to _moved/old_draft.txt", "notes.txt": "not moved"}, []),
    (["delete", "no"], {"old_draft.txt": "kept (not deleted)", "notes.txt": "not moved"}, [MIXED]),
    (["delete", "yes"], {"old_draft.txt": "kept (not deleted)", "notes.txt": "moved to docs/notes.txt"}, [MIXED]),
])
def test_c10_mixed_precheck_note_names_every_file_for_every_answer(disk, answers, want, ran):
    events = []
    ask = Ask(answers=list(answers))
    _, effect = C10_CASES["mixed"]
    res, _, shell = _go(disk, MIXED, ask=ask, seed=C10_SEED, script={MIXED: effect(disk)}, on_event=events.append)
    note = res.tool_results[0]
    assert note.startswith("ERROR: harness"), note
    assert _outcomes(note) == want, note
    assert shell.ran == ran
    assert _places(events)[0] == ("precheck", ["delete", "move"])
    _disk_agrees(disk, want)


def test_c10_mixed_precheck_delete_then_yes_asks_twice_and_the_second_is_the_move(disk):
    events = []
    _, effect = C10_CASES["mixed"]
    _go(disk, MIXED, ask=Ask(answers=["delete", "yes"]), seed=C10_SEED, script={MIXED: effect(disk)},
        on_event=events.append)
    assert _places(events) == [("precheck", ["delete", "move"]), ("after_step", ["move"])]
    assert _exists(disk, "old_draft.txt") and _exists(disk, "docs/notes.txt") and not _exists(disk, "notes.txt")


def test_c10_mixed_precheck_a_yes_is_refused_and_the_note_still_names_every_file(disk):
    _, effect = C10_CASES["mixed"]
    res, _, shell = _go(disk, MIXED, ask=Ask("yes"), seed=C10_SEED, script={MIXED: effect(disk)})
    assert shell.ran == []
    assert _outcomes(res.tool_results[0]) == {"old_draft.txt": "kept (not deleted)", "notes.txt": "not moved"}
    assert res.summary().get("ask_errors", 0) >= 1


@pytest.mark.parametrize("answer,want", [
    ("no", {"old_draft.txt": "kept (not deleted)", "notes.txt": "not moved"}),
    ("delete", {"old_draft.txt": "deleted", "notes.txt": "not moved"}),
    ("move", {"old_draft.txt": "moved aside to _moved/old_draft.txt", "notes.txt": "not moved"}),
])
def test_c10_mixed_after_step_note_names_every_file_for_every_answer(disk, answer, want):
    events = []
    script = {MIXED_PY: _both(_rm(disk, "old_draft.txt"), _rename(disk, "notes.txt", "docs/notes.txt"))}
    res, _, _ = _go(disk, MIXED_PY, ask=Ask(answer), seed=C10_SEED, script=script, on_event=events.append)
    note = res.tool_results[0]
    assert _places(events) == [("after_step", ["move", "delete"])]
    assert res.undone == [True]
    assert note.startswith("ERROR: harness"), note
    assert _outcomes(note) == want, note
    _disk_agrees(disk, want)


def test_c10_single_kind_notes_are_exactly_as_before(disk):
    res, _, _ = _go(disk, "rm notes.txt", ask=Ask("no"), seed=C10_SEED, script={"rm notes.txt": _boom})
    assert res.tool_results[0] == HELD
    res, _, _ = _go(disk, "mv notes.txt docs/", ask=Ask("no"), seed=C10_SEED, script={"mv notes.txt docs/": _boom})
    _assert_move_no_note(res.tool_results[0], "notes.txt")
    assert "What happened" not in res.tool_results[0]


# --- 2. the real destination ----------------------------------------------------------------------------


@pytest.mark.parametrize("command,seed,expected", [
    ("mv notes.txt docs", C10_SEED, {"notes.txt": "docs/notes.txt"}),                  # an existing folder: inside
    ("mv notes.txt docs", SEED, {"notes.txt": "docs" + UNKNOWN}),                      # nothing there yet
    ("mv notes.txt notes.bak", SEED, {"notes.txt": "notes.bak" + UNKNOWN}),            # a plain rename: the same
    ("mv notes.txt keep.txt", SEED, {"notes.txt": "keep.txt"}),                        # an existing file: over it
    ("mv -T notes.txt docs", C10_SEED, {"notes.txt": "docs"}),                         # -T: always a rename
    ("mv -t docs notes.txt", SEED, {"notes.txt": "docs/notes.txt"}),                   # -t: always a folder
    ("mv notes.txt keep.txt docs", SEED, {"notes.txt": "docs/notes.txt", "keep.txt": "docs/keep.txt"}),
    ("mv notes.txt docs/", SEED, {"notes.txt": "docs/notes.txt"}),                     # a slash: a folder
    ("mkdir docs && mv notes.txt docs", SEED, {"notes.txt": "docs" + UNKNOWN}),        # made by the command
])
def test_c10_the_move_question_shows_where_mv_would_really_put_the_file(disk, command, seed, expected):
    events = []
    _go(disk, command, ask=Ask("no"), seed=seed, script={command: _boom}, on_event=events.append)
    asks = _asks(events)
    assert len(asks) == 1 and asks[0]["why"] == ["move"] * len(expected)
    assert dict(zip(asks[0]["paths"], asks[0]["dests"])) == expected


@pytest.mark.parametrize("command,seed,lands", [
    ("mkdir docs && mv notes.txt docs", SEED, "docs/notes.txt"),   # "docs (a folder or a new name)": the folder
    ("mv notes.txt docs", SEED, "docs"),                           # the same question: the new name
])
def test_c10_a_yes_to_the_shown_place_is_one_question_when_the_file_lands_there(disk, command, seed, lands):
    events = []
    ask = Ask(answers=["yes", "no"])
    _go(disk, command, ask=ask, seed=seed, script={command: _rename(disk, "notes.txt", lands)},
        on_event=events.append)
    assert _places(events) == [("precheck", ["move"])]
    assert _exists(disk, lands) and not _exists(disk, "notes.txt")


# C10b (review fix 1): `mv notes.txt Docs` with `docs` on disk is one folder on NTFS and another name on a
# case-sensitive disk (Linux, or a Windows work folder with the flag on, which Stage 1 uses). The expectation follows
# the test's own folder: the reviewer's probe (make ``a``, look for ``A``).


def _case_sensitive(folder: Path) -> bool:
    probe = folder / "case_probe_a"
    probe.write_text("x", encoding="utf-8")
    try:
        return not (folder / "CASE_PROBE_A").exists()
    finally:
        probe.unlink()


def _case_rule(disk, sensitive: bool) -> bool:
    """Make the trial folder case-sensitive when asked (Windows: the flag on the empty work folder; Linux: it already
    is), then say what the disk really does."""
    if sensitive and os.name == "nt":
        set_case_sensitive(disk.root)
    return _case_sensitive(disk.root)


@pytest.mark.parametrize("sensitive", [False, True])
def test_c10b_the_move_question_follows_the_disks_case_rule(disk, sensitive):
    expect = "Docs" + UNKNOWN if _case_rule(disk, sensitive) else "Docs/notes.txt"
    events = []
    _go(disk, "mv notes.txt Docs", ask=Ask("no"), seed=C10_SEED, script={"mv notes.txt Docs": _boom},
        on_event=events.append)
    asks = _asks(events)
    assert len(asks) == 1 and asks[0]["why"] == ["move"]
    assert dict(zip(asks[0]["paths"], asks[0]["dests"])) == {"notes.txt": expect}


@pytest.mark.parametrize("sensitive", [False, True])
def test_c10b_a_yes_to_docs_in_another_case_is_one_question_where_the_disk_puts_it(disk, sensitive):
    lands = "Docs" if _case_rule(disk, sensitive) else "docs/notes.txt"  # a new name, or inside the one folder
    events = []
    cmd = "mv notes.txt Docs"
    _go(disk, cmd, ask=Ask(answers=["yes", "no"]), seed=C10_SEED, script={cmd: _rename(disk, "notes.txt", lands)},
        on_event=events.append)
    assert _places(events) == [("precheck", ["move"])]
    assert _exists(disk, lands) and not _exists(disk, "notes.txt")


# --- 3. a yes is for the shown place only -------------------------------------------------------------


def test_c10_a_yes_to_docs_then_the_file_lands_elsewhere_is_an_unasked_move(disk):
    events = []
    ask = Ask(answers=["yes", "no"])
    cmd = "mv notes.txt docs"
    res, _, _ = _go(disk, cmd, ask=ask, seed=C10_SEED, script={cmd: _rename(disk, "notes.txt", "other/n.txt")},
                    on_event=events.append)
    asks = _asks(events)
    assert _places(events) == [("precheck", ["move"]), ("after_step", ["move"])]
    assert asks[0]["dests"] == ["docs/notes.txt"] and asks[1]["dests"] == ["other/n.txt"]
    assert res.undone == [True]
    assert _exists(disk, "notes.txt") and not _exists(disk, "other/n.txt")
    _assert_move_no_note(res.tool_results[0], "notes.txt")


def test_c10_a_yes_to_the_shown_place_does_not_cover_a_second_file_moved_by_the_same_step(disk):
    """The whole step is undone (C8), so the question after it names both moves, the allowed one included."""
    events = []
    ask = Ask(answers=["yes", "no"])
    cmd = "mv notes.txt docs"
    effect = _both(_rename(disk, "notes.txt", "docs/notes.txt"), _rename(disk, "keep.txt", "docs/keep.txt"))
    res, _, _ = _go(disk, cmd, ask=ask, seed=C10_SEED, script={cmd: effect}, on_event=events.append)
    assert _places(events) == [("precheck", ["move"]), ("after_step", ["move", "move"])]
    assert _asks(events)[1]["paths"] == ["keep.txt", "notes.txt"]
    assert res.undone == [True]
    assert _exists(disk, "keep.txt") and _exists(disk, "notes.txt") and not _exists(disk, "docs/keep.txt")


# --- 4. a link, an unpack or dd over an existing file: the overwrite ask ------------------------------


@pytest.mark.parametrize("command", [LN, DD, "ln -s -f other.txt notes.txt", "ln -fs other.txt docs/readme.md",
                                     "echo x | dd of=notes.txt", "ln -sf other.txt docs/"])
def test_c10_a_link_or_dd_over_an_existing_file_is_asked_as_a_replace_before_it_runs(disk, command):
    events = []
    seed = dict(C10_SEED, **{"docs/other.txt": "D\n"})
    res, _, shell = _go(disk, command, ask=Ask("no"), seed=seed, script={command: _boom}, on_event=events.append)
    target = "docs/readme.md" if "readme" in command else "docs/other.txt" if command.endswith("docs/") else "notes.txt"
    assert _places(events) == [("precheck", ["replace"])]
    assert _asks(events)[0]["paths"] == [target]
    assert shell.ran == []
    assert res.tool_results[0] == HELD
    assert _text(disk, target) in ("N\n", "R\n", "D\n") and not os.path.islink(disk.root / target)


@pytest.mark.parametrize("command,effect", [
    (TAR, lambda d: _write(d, "notes.txt", b"T\n")),
    (UNZIP, lambda d: _write(d, "notes.txt", b"T\n")),
    ("tar xzf a.tgz -C .", lambda d: _write(d, "notes.txt", b"T\n")),
    ("dd of=\"$f\" if=other.txt", lambda d: _write(d, "notes.txt", b"\0\0")),  # the text names no file
    ("python -c \"import os; os.remove('notes.txt'); os.symlink('other.txt', 'notes.txt')\"",
     lambda d: _symlink_swap(d, "notes.txt", "other.txt")),
])
def test_c10_an_unpack_dd_or_link_the_precheck_cannot_name_is_asked_as_a_replace_after_the_step(
        disk, command, effect):
    events = []
    res, _, shell = _go(disk, command, ask=Ask("no"), seed=C10_SEED, script={command: effect(disk)},
                        on_event=events.append)
    assert _places(events) == [("after_step", ["replace"])]
    assert _asks(events)[0]["paths"] == ["notes.txt"]
    assert shell.ran == [command] and res.undone == [True]
    assert res.tool_results[0] == HELD
    assert _text(disk, "notes.txt") == "N\n" and not os.path.islink(disk.root / "notes.txt")


def test_c10_tar_replace_answer_delete_lets_the_old_file_go_and_the_command_is_not_run_again(disk):
    res, _, shell = _go(disk, TAR, ask=Ask("delete"), seed=C10_SEED, script={TAR: _write(disk, "notes.txt", b"T\n")})
    assert shell.ran == [TAR] and res.undone == [True]
    assert not _exists(disk, "notes.txt")
    assert "deleted notes.txt" in res.tool_results[0] and "not run again" in res.tool_results[0]


def test_c10_tar_replace_answer_move_puts_the_old_file_aside(disk):
    res, _, _ = _go(disk, TAR, ask=Ask("move"), seed=C10_SEED, script={TAR: _write(disk, "notes.txt", b"T\n")})
    assert _text(disk, "_moved/notes.txt") == "N\n" and not _exists(disk, "notes.txt")
    assert "moved notes.txt to _moved/notes.txt" in res.tool_results[0]


def test_c10_an_ordinary_edit_in_the_step_after_an_unpack_is_not_asked(disk):
    """The unpack flag is for that one step: a sed -i in the next step is an edit, not a replace."""
    sed = "sed -i s/N/M/ notes.txt"
    ask = Ask("no")
    script = {TAR: _write(disk, "notes.txt", b"T\n"), sed: _write(disk, "notes.txt", b"M\n")}
    body = [reply(calls=[call("bash", command=TAR)]), reply(calls=[call("bash", command=sed)]), reply("done")]
    _go(disk, TAR, ask=ask, seed=C10_SEED, script=script, replies=body)
    assert ask.calls == [["notes.txt"]]
    assert _text(disk, "notes.txt") == "M\n"


def test_c10_an_unpack_over_a_file_the_bot_made_is_not_asked(disk):
    ask = Ask("no")
    script = {"touch new.txt": _write(disk, "new.txt", b"X\n"), TAR: _write(disk, "new.txt", b"T\n")}
    body = [reply(calls=[call("bash", command="touch new.txt")]), reply(calls=[call("bash", command=TAR)]),
            reply("done")]
    _go(disk, "touch new.txt", ask=ask, seed=C10_SEED, script=script, replies=body)
    assert ask.calls == []
    assert _text(disk, "new.txt") == "T\n"


@pytest.mark.parametrize("command", ["tar -tf a.tar", "ln -s other.txt new_link.txt", "ln -s other.txt notes.txt",
                                     "dd of=copy.bin if=notes.txt", "unzip -l a.zip"])
def test_c10_the_case_next_to_each_replace_rule_is_not_asked(disk, command):
    """A list, a link to a NEW name, a link without -f (fails on an existing name), dd READING a file."""
    ask = Ask("no")
    effect = _write(disk, "copy.bin", b"N\n") if "copy.bin" in command else (lambda: "")
    res, _, shell = _go(disk, command, ask=ask, seed=C10_SEED, script={command: effect})
    assert ask.calls == [] and shell.ran == [command] and res.undone == [False]


# --- 5. chmod: the permissions ask ----------------------------------------------------------------------


def test_c10_chmod_of_an_existing_file_asks_permissions_before_it_runs_answer_no(disk):
    events = []
    try:
        res, _, shell = _go(disk, CHMOD, ask=Ask("no"), seed=C10_SEED,
                            script={CHMOD: _read_only(disk.root / "run.sh")}, on_event=events.append)
        assert _places(events) == [("precheck", ["permissions"])]
        assert _asks(events)[0]["paths"] == ["run.sh"] and "dests" not in _asks(events)[0]
        assert shell.ran == [] and res.ran == [False]
        assert _mode(disk, "run.sh") & stat.S_IWRITE, "the mode changed although the person said no"
        note = res.tool_results[0]
        assert note.startswith("ERROR: harness note, not from the user"), note
        assert "said no" in note and "permissions" in note and "run.sh" in note, note
    finally:
        _writable(disk.root / "run.sh")  # a read-only file would stop the trial folder's cleanup


def test_c10_chmod_answer_yes_runs_it_once_and_tells_the_model(disk):
    res, _, shell = _go(disk, CHMOD, ask=Ask("yes"), seed=C10_SEED, script={CHMOD: _read_only(disk.root / "run.sh")})
    try:
        assert shell.ran == [CHMOD] and res.ran == [True] and res.undone == [False]
        assert not _mode(disk, "run.sh") & stat.S_IWRITE
        out = res.tool_results[0]
        assert out.startswith("[harness: the user said yes") and "permissions" in out and "run.sh" in out, out
    finally:
        _writable(disk.root / "run.sh")


def test_c10_chmod_of_a_file_the_bot_made_is_not_asked(disk):
    ask = Ask("no")
    script = {"touch new.sh": _write(disk, "new.sh", b"#!/bin/sh\n"), "chmod +x new.sh": (lambda: "")}
    body = [reply(calls=[call("bash", command="touch new.sh")]), reply(calls=[call("bash", command="chmod +x new.sh")]),
            reply("done")]
    _, _, shell = _go(disk, "touch new.sh", ask=ask, seed=C10_SEED, script=script, replies=body)
    assert ask.calls == [] and shell.ran == ["touch new.sh", "chmod +x new.sh"]


@pytest.mark.parametrize("command,paths", [
    ("chmod 755 run.sh", ["run.sh"]),
    ("chmod u+x run.sh", ["run.sh"]),
    ("chmod -x run.sh", ["run.sh"]),                       # a mode that looks like an option
    ("chmod -v +x run.sh", ["run.sh"]),
    ("chmod --reference=other.txt run.sh", ["run.sh"]),
    ("chmod +x run.sh other.txt", ["other.txt", "run.sh"]),
    ("chmod -R 755 docs", ["docs/readme.md"]),             # -R: the files under the folder
    ("chmod 644 ./run.sh", ["run.sh"]),
])
def test_c10_chmod_shapes_name_the_existing_files(disk, command, paths):
    events = []
    _, _, shell = _go(disk, command, ask=Ask("no"), seed=C10_SEED, script={command: _boom}, on_event=events.append)
    asks = _asks(events)
    assert len(asks) == 1 and sorted(asks[0]["paths"]) == paths
    assert asks[0]["why"] == ["permissions"] * len(paths) and shell.ran == []


def test_c10_chmod_on_a_folder_without_recursion_names_no_file(disk):
    ask = Ask("no")
    _, _, shell = _go(disk, "chmod 755 docs", ask=ask, seed=C10_SEED, script={"chmod 755 docs": (lambda: "")})
    assert ask.calls == [] and shell.ran == ["chmod 755 docs"]


def test_c10_a_move_and_a_chmod_in_one_step_ask_once_and_a_yes_covers_both(disk):
    events = []
    cmd = "mv notes.txt docs/ && chmod +x run.sh"
    effect = _both(_mv(disk, "notes.txt", "docs/"), _read_only(disk.root / "run.sh"))
    try:
        res, _, shell = _go(disk, cmd, ask=Ask("yes"), seed=C10_SEED, script={cmd: effect}, on_event=events.append)
        asks = _asks(events)
        assert len(asks) == 1 and asks[0]["paths"] == ["notes.txt", "run.sh"] and asks[0]["why"] == ["move", "permissions"]
        assert asks[0]["dests"] == ["docs/notes.txt", ""]
        assert shell.ran == [cmd] and res.undone == [False]
        assert _exists(disk, "docs/notes.txt") and not _mode(disk, "run.sh") & stat.S_IWRITE
        assert "notes.txt" in res.tool_results[0] and "run.sh" in res.tool_results[0]
    finally:
        _writable(disk.root / "run.sh")


def test_c10_a_move_and_a_chmod_in_one_step_answer_no_runs_nothing_and_names_both(disk):
    cmd = "mv notes.txt docs/ && chmod +x run.sh"
    res, _, shell = _go(disk, cmd, ask=Ask("no"), seed=C10_SEED, script={cmd: _boom})
    assert shell.ran == []
    assert _outcomes(res.tool_results[0]) == {"notes.txt": "not moved", "run.sh": "permissions unchanged"}


def test_c10_a_delete_and_a_chmod_in_one_step_answer_delete_does_not_let_the_chmod_through(disk):
    """A permission change cannot be checked after the step, so a delete answer must not run the command."""
    events = []
    cmd = "rm old_draft.txt && chmod +x run.sh"
    res, _, shell = _go(disk, cmd, ask=Ask("delete"), seed=C10_SEED, script={cmd: _boom}, on_event=events.append)
    assert _places(events) == [("precheck", ["delete", "permissions"])]
    assert shell.ran == [] and _exists(disk, "old_draft.txt")
    note = res.tool_results[0]
    assert "could not be done" in note and "permissions" in note, note
    assert _outcomes(note) == {"old_draft.txt": "kept (not deleted)", "run.sh": "permissions unchanged"}


def test_c10_a_delete_and_a_chmod_in_one_step_answer_move_puts_the_file_aside_and_runs_nothing(disk):
    cmd = "rm old_draft.txt && chmod +x run.sh"
    res, _, shell = _go(disk, cmd, ask=Ask("move"), seed=C10_SEED, script={cmd: _boom})
    assert shell.ran == []
    assert _outcomes(res.tool_results[0]) == {"old_draft.txt": "moved aside to _moved/old_draft.txt",
                                              "run.sh": "permissions unchanged"}


def test_c10_a_yes_to_a_chmod_is_for_that_message_only(disk):
    ask = Ask(answers=["yes", "no"])
    script = {CHMOD: (lambda: ""), "chmod -x run.sh": _boom}
    body = [reply(calls=[call("bash", command=CHMOD)]), reply("first done"),
            reply(calls=[call("bash", command="chmod -x run.sh")]), reply("second done")]
    ws = disk.ws(dict(C10_SEED))
    shell = FakeShell(script)
    run_task(C4, provider(ScriptedTransport(body), C4.provider), "fake:1", ["first", "second"], ws, shell=shell,
             undo=_store(disk), ask_delete=ask)
    assert ask.calls == [["run.sh"], ["run.sh"]]
    assert shell.ran == [CHMOD]


@pytest.mark.parametrize("command,effect_key", [(CHMOD, "chmod"), (TAR, "tar_x")])
def test_c10_last_step_answer_no_at_max_turns_1_asks_and_stops(disk, command, effect_key):
    events = []
    profile = _profile_max_turns(1)
    transport = ScriptedTransport([reply(calls=[call("bash", command=command)])])
    ws = disk.ws(dict(C10_SEED))
    shell = FakeShell({command: C10_CASES[effect_key][1](disk)})
    res = run_task(profile, provider(transport, profile.provider), "fake:1", "tidy up", ws, shell=shell,
                   undo=_store(disk), ask_delete=Ask("no"), on_event=events.append)
    assert [a["why"] for a in _asks(events)] == [["permissions"] if effect_key == "chmod" else ["replace"]]
    assert _text(disk, "notes.txt") == "N\n"
    assert len(transport.bodies) == 1 and res.reason == "max_steps"


# --- 6. the chat: the permissions question, the real path, the Driver -----------------------------------


def test_c10_chat_chmod_shows_the_permissions_question_and_n_keeps_the_mode(tmp_path):
    w, work = _chat(tmp_path, ["n"], CHMOD, lambda work: _read_only(work / "run.sh"))
    assert w.run() == 0, w.text
    assert "The bot wants to CHANGE PERMISSIONS of: run.sh." in w.text
    assert chat.PROMPT_ASK_PERM in w.keys.prompts and "Allow it? (y/n)" in chat.PROMPT_ASK_PERM
    assert [e["answer"] for e in w.events("ask_answer")] == ["no"]
    assert w.events("ask")[0]["why"] == ["permissions"]
    assert w.shells[0].ran == [] and os.access(work / "run.sh", os.W_OK)
    assert "MOVE" not in w.text and "[d] delete" not in "".join(w.keys.prompts)


def test_c10_chat_chmod_answer_y_runs_it(tmp_path):
    w, work = _chat(tmp_path, ["y"], CHMOD, lambda work: _read_only(work / "run.sh"))
    try:
        assert w.run() == 0, w.text
        assert [e["answer"] for e in w.events("ask_answer")] == ["yes"]
        assert w.shells[0].ran == [CHMOD] and not os.access(work / "run.sh", os.W_OK)
    finally:
        _writable(work / "run.sh")


@pytest.mark.parametrize("typed,answer", [("", "no"), ("N", "no"), ("Y", "yes"), ("d", "no")])
def test_c10_chat_permissions_prompt_takes_only_y_or_n(tmp_path, typed, answer):
    w, _ = _chat(tmp_path, [typed, "n", "n"], CHMOD, lambda work: (lambda: ""))
    assert w.run() == 0, w.text
    assert [e["answer"] for e in w.events("ask_answer")] == [answer]
    if typed == "d":
        assert "Please type y or n" in w.text


def test_c10_chat_move_question_shows_the_real_path_into_an_existing_folder(tmp_path):
    w, work = _chat(tmp_path, ["n"], "mv notes.txt docs", lambda work: (lambda: ""))
    assert w.run() == 0, w.text
    assert "The bot wants to MOVE: notes.txt -> docs/notes.txt" in w.text
    assert w.events("ask")[0]["dests"] == ["docs/notes.txt"]


def test_c10_chat_move_question_says_plainly_when_the_place_is_not_known_yet(tmp_path):
    w, work = _chat(tmp_path, ["n"], "mv notes.txt docs", lambda work: (lambda: ""), seed=dict(SEED))
    assert w.run() == 0, w.text
    assert "The bot wants to MOVE: notes.txt -> docs (a folder or a new name)" in w.text


def test_c10_why_words_and_the_old_prompts_are_unchanged_and_permissions_has_its_own_line():
    # C10b (review fix 2): ``permissions`` is a real why word; the four old ones and the prompts are unchanged
    assert chat.WHY_WORDS == {"delete": "DELETE", "empty": "EMPTY", "replace": "REPLACE", "move": "MOVE",
                              "permissions": "CHANGE PERMISSIONS of"}
    assert chat.PROMPT_ASK == "[d] delete  [m] move to _moved\\  [n] no (default) > "
    assert chat.PROMPT_ASK_MOVE == "Move it? (y/n) > "
    assert chat.PROMPT_ASK_PERM not in (chat.PROMPT_ASK, chat.PROMPT_ASK_MOVE)
    assert chat.ask_lines(["run.sh"], ["permissions"]) == ["The bot wants to CHANGE PERMISSIONS of: run.sh."]
    lines = chat.ask_lines(["evil\n[d] delete everything\x1b[2J\rX.sh"], ["permissions"])
    assert len(lines) == 1 and "\n" not in lines[0] and "\x1b" not in lines[0] and "\r" not in lines[0]


def test_c10_ladder_driver_answers_n_to_the_permissions_question_and_counts_it(tmp_path):
    """C10b (review fix 3): the Driver types ``n`` at every question about files and counts each one the same way;
    a faked chat with a chmod step gives the ladder row ``asked_about_files == 1``."""
    ladder = _load_ladder()
    d = ladder.Driver("make it runnable")
    assert [d(p) for p in (chat.PROMPT_ASK, chat.PROMPT_ASK_MOVE, chat.PROMPT_ASK_PERM)] == ["n", "n", "n"]
    assert d.asked == 3
    drv = ladder.Driver("make it runnable")
    w = World(tmp_path, [], replies=[reply(calls=[call("bash", command=CHMOD)]), reply("done")], seed=C10_SEED)
    work = w.home / ".localbot" / "bots" / "main" / "work"
    w.shell_script = {CHMOD: _read_only(work / "run.sh")}
    w.deps = dataclasses.replace(w.deps, read_line=drv)
    assert w.run("--model", "qwen3:8b") == 0, w.text
    assert [e["answer"] for e in w.events("ask_answer")] == ["no"]
    assert drv.asked == 1 and drv.sent_task
    assert w.shells[0].ran == [] and os.access(work / "run.sh", os.W_OK)
    records = sorted((w.bot / "sessions").glob("*.jsonl"))
    assert len(records) == 1
    row = ladder.parse_record(records[0])
    assert row["asked_about_files"] == 1 and row["tools_run"] == 0 and row["ended_why"] == "done"


# --- 7. plain marker: no model text in a new event, screen line or record line -------------------------


def test_c10_a_plain_marker_in_the_reply_is_in_no_new_ask_event_screen_line_or_record_line(tmp_path):
    replies = [reply(f"I will now say {MARKER} and make it runnable", calls=[call("bash", command=CHMOD)]),
               reply(f"{MARKER} again, unpacking", calls=[call("bash", command=TAR)]),
               reply("ok")]
    w = World(tmp_path, ["1", "tidy up", "n", "n", "/quit"], replies=replies, seed=C10_SEED)
    work = w.home / ".localbot" / "bots" / "main" / "work"
    w.shell_script = {CHMOD: _read_only(work / "run.sh"), TAR: _write_at(work / "notes.txt", b"T\n")}
    assert w.run() == 0, w.text
    asks = w.events("ask")
    assert [a["why"] for a in asks] == [["permissions"], ["replace"]], asks
    assert MARKER not in json.dumps(asks) and MARKER not in json.dumps(w.events("ask_answer"))
    lines = [ln for ln in w.text.splitlines()
             if "CHANGE PERMISSIONS" in ln or "REPLACE" in ln or "Allow it" in ln or "[d] delete" in ln]
    assert len(lines) >= 2, w.text
    assert all(MARKER not in ln for ln in lines), lines
    for block in w.record_text().splitlines():
        if '"event": "ask"' in block or '"event": "ask_answer"' in block:
            assert MARKER not in block, block
        if "CHANGE PERMISSIONS" in block or "Allow it?" in block or "REPLACE" in block:
            assert MARKER not in block, block
    assert any(e.get("event") == "say" and MARKER in (e.get("text") or "") for e in w.events())


def _write_at(path, data: bytes):
    def run():
        path.write_bytes(data)
        return ""
    return run
