"""H1-J J3: the guard's path key. The lab normalised paths with ``lstrip("./")``, which strips
*characters*, not the ``./`` prefix: ``.env`` -> ``env``, ``.lib.py`` -> ``lib.py``, ``../x`` -> ``x``.
core4 (container mode) now uses one ``removeprefix``-based normaliser (``product.core_rel``, like
``context._norm`` but case kept); parity (emulated) keeps the lab's ``lstrip`` byte for byte.

One probe per call site in ``guard/product.py`` (container mode), each a bypass or a false match the
old key caused; then the same inputs in emulated mode against the lab's own ``harness_guards.py``.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

from localbot.guard import product as new
from localbot.guard.product import CallerGuard, ProductGuard

REPO = Path(__file__).resolve().parents[1]
LIB = {"lib.py": "def helper():\n    return 1\n", "app.py": "from lib import helper\nhelper()\n"}


def _exec_lab(rel: str, modname: str) -> types.ModuleType:
    """Run a frozen lab module from source (nothing imported from or written into lab_ref)."""
    mod = types.ModuleType(modname)
    sys.modules[modname] = mod
    src = (REPO / "lab_ref" / rel).read_text(encoding="utf-8")
    exec(compile(src, f"lab_ref/{rel}", "exec"), mod.__dict__)
    return mod


lab = _exec_lab("lib/harness_guards.py", "lab_harness_guards_h1j")


def core4() -> ProductGuard:
    return ProductGuard(shell_mode="container")


# --- the normaliser ------------------------------------------------------------------------------


def test_core_rel_drops_only_dot_slash_prefixes():
    assert new.core_rel(".env") == ".env"
    assert new.core_rel("./.env") == ".env"
    assert new.core_rel("././a\\b.py") == "a/b.py"
    assert new.core_rel("../x") == "../x"
    assert new.core_rel(".../x") == ".../x"
    assert new.core_rel("") == "" and new.core_rel(None) == ""
    assert new.lab_rel(".env") == "env" and new.lab_rel("../x") == "x"  # the lab's, kept for parity


# --- one probe per call site (container mode) ------------------------------------------------------


def test_probe_245_parse_search_hits_keeps_the_dot():
    """``parse_search_hits``: grep output ``./.env:1:…`` named a file ``env`` (false match)."""
    g = core4()
    g.observe("bash", {"command": "grep -rn KEY ."}, "./.env:1:KEY=x\n./app.py:2:KEY\n", LIB)
    assert g.caller.hits == [".env", "app.py"]


def test_probe_285_note_read_does_not_match_a_different_file():
    """``CallerGuard.note_read``: reading ``.env`` counted as reading a hit named ``env``."""
    g = core4()
    g.caller.hits = ["lib.py", "env"]
    g.caller.note_read(".env")
    assert g.caller.read_later is False


def test_probe_293_veto_write_body_is_not_the_definer_without_its_dot():
    """``CallerGuard.veto_write`` body: an answer ``.lib.py`` was read as ``lib.py`` and refused as
    "lib.py defines the function" (false match: ``.lib.py`` is another file)."""
    g = core4()
    assert g.check("write", {"path": "answer.txt", "content": ".lib.py"}, dict(LIB)) is None


def test_probe_294_veto_write_dest_is_not_the_body_without_its_dot():
    """``CallerGuard.veto_write`` dest: writing ``lib.py`` into ``.lib.py`` matched "a file holding its
    own name" (``.lib.py`` -> ``lib.py``) and skipped the caller rule (bypass)."""
    g = core4()
    out = g.check("write", {"path": ".lib.py", "content": "lib.py"}, dict(LIB))
    assert out == "harness: lib.py defines the function; app.py calls it. Write the caller filename, not the definer."


def test_probe_536_mutate_after_harm_does_not_lock_a_new_dot_file():
    """``veto_mutate_after_harm``: after a refused destructive bash, a NEW ``.app.py`` was refused as
    "changing existing files" because ``app.py`` exists (false match)."""
    g = core4()
    g.refused_destroy = True
    assert g.check("write", {"path": ".app.py", "content": "x = 1\n"}, dict(LIB)) is None
    assert g.check("write", {"path": "app.py", "content": "x = 1\n"}, dict(LIB)) is not None  # still locked


def test_probe_544_a_dot_file_is_not_a_test_file():
    """``_is_test_path``: ``.test_notes.py`` became ``test_notes.py`` and was refused as a test edit."""
    g = core4()
    assert g.check("write", {"path": ".test_notes.py", "content": "x\n"}, dict(LIB)) is None
    assert g.check("write", {"path": "test_notes.py", "content": "x\n"}, dict(LIB)) is not None  # exam rule


def test_probe_583_emptying_a_new_dot_file_is_not_emptying_its_namesake():
    """``veto_empty_existing``: an empty write to a NEW ``.app.py`` was refused as "emptying an existing
    file" because ``app.py`` holds text (false match)."""
    g = core4()
    assert g.check("write", {"path": ".app.py", "content": ""}, dict(LIB)) is None
    assert g.check("write", {"path": "app.py", "content": ""}, dict(LIB)) == "harness: refused emptying an existing file."


def test_probe_760_repeat_read_is_parity_only():
    """``ProductGuard.repeat_read`` is the lab's (parity); core4 counts with ``read_is_repeated`` /
    ``note_read_ok``. In container mode it would key ``.env`` and ``env`` apart."""
    g = core4()
    g.repeat_read("env")
    g.repeat_read("env")
    assert g.repeat_read(".env") is False


def test_probe_769_774_read_counts_keep_dot_files_apart():
    """``read_is_repeated`` / ``note_read_ok``: two reads of ``env`` made ``.env`` count as read
    twice (false "already read … Contents unchanged" for a file never read)."""
    g = core4()
    g.note_read_ok("env")
    g.note_read_ok("env")
    assert g.read_is_repeated(".env") is False and g.read_is_repeated("env") is True
    g.note_read_ok("./.env")
    g.note_read_ok(".env")
    assert g.read_is_repeated(".env") is True  # './.env' and '.env' are the same file


def test_probe_781_reset_read_resets_only_that_file():
    """``reset_read``: a bash step that changed ``env`` reset ``.env``'s count (and kept ``env``'s)."""
    g = core4()
    g.note_read_ok(".env")
    g.note_read_ok(".env")
    g.reset_read("env")
    assert g.read_is_repeated(".env") is True


def test_probe_784_already_read_message_names_the_right_file():
    """``already_read_msg``: told the model it had read ``env`` when it read ``.env``."""
    assert core4().already_read_msg("./.env").startswith("already read .env this task.")


# --- parity: emulated mode == the lab, byte for byte, on the same inputs ---------------------------


def test_parity_parse_search_hits_is_the_labs():
    text = "./.env:1:KEY=x\n./app.py:2:KEY\n.lib.py:3:x\n"
    assert new.parse_search_hits(text) == lab.parse_search_hits(text) == ["env", "app.py", "lib.py"]


def test_parity_caller_guard_is_the_labs():
    for dest, content in [("answer.txt", ".lib.py"), (".lib.py", "lib.py"), ("x.txt", "./lib.py")]:
        assert CallerGuard().veto_write(dest, content, dict(LIB)) == lab.CallerGuard().veto_write(dest, content, dict(LIB))
    a, b = CallerGuard(hits=["lib.py", "env"]), lab.CallerGuard(hits=["lib.py", "env"])
    a.note_read(".env")
    b.note_read(".env")
    assert a.read_later == b.read_later is True


def test_parity_module_vetoes_are_the_labs():
    for args in [{"path": ".app.py", "content": "x"}, {"path": "./app.py", "content": ""}, {"path": ".app.py", "content": ""}]:
        assert new.veto_mutate_after_harm(True, "write", args, dict(LIB)) == lab.veto_mutate_after_harm(True, "write", args, dict(LIB))
        assert new.veto_empty_existing("write", args, dict(LIB)) == lab.veto_empty_existing("write", args, dict(LIB))
    for p in [".test_notes.py", "./test_a.py", "a_test.py", ".x.py"]:
        assert new._is_test_path(p) == lab._is_test_path(p)


def test_parity_read_counts_and_message_are_the_labs():
    g, old = ProductGuard(), lab.ProductGuard()
    for p in ["env", "env", ".env", "./.env"]:
        assert g.repeat_read(p) == old.repeat_read(p)
    assert g.already_read_msg("./.env") == old.already_read_msg("./.env")
