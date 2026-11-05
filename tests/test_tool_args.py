"""S8f.1a — tool-argument gate (L1), executor hardening (L2), honesty postcondition (L3).

A malformed native tool call (wrong-case / missing / non-string key) used to reach lab_run_tool,
where ``args.get(key) or ""`` turned the missing key into "" and the tool lied ("Updated"/"Wrote"
with the disk unchanged). The gate validates arg KEYS against the profile's own advertised schema
*before* the tool runs; the executor refuses the same class as defence in depth; the postcondition
catches any "said success, disk not as claimed" from any cause. Extra keys are tolerated, logged,
and dropped so the tool and the guards see only schema keys. Never auto-repairs.

Fakes only (no model, no GPU, no container): the gate short-circuits before exec, so even core4
malformed calls need no shell."""
from __future__ import annotations

import pytest

from localbot.config import get_profile
from localbot.guard.toolargs import ArgCheck, check_tool_args, schema_map
from localbot.loop import run_task
from localbot.tools.fs import lab_run_tool
from tests.fakes import ScriptedTransport, call, provider, reply

P6, C4 = get_profile("parity_lab6"), get_profile("core4")
P6S, C4S = P6.tools.schemas, C4.tools.schemas

# the exact text the brief/regression requires
NEMOTRON_ERR = ("ERROR: edit needs keys path, old_string, new_string "
                "(got: path, old_String, new_String). Keys are case-sensitive: use old_string, new_string.")


def run(profile, script, ws, **kw):
    tr = ScriptedTransport(script)
    res = run_task(profile, provider(tr, profile.provider), "fake:1", "do the ticket", ws, **kw)
    return res, tr


# ======================================================================= L1 gate (unit)

def test_schema_map_from_profile_schemas():
    m6 = schema_map(P6S)
    assert set(m6) == {"read", "write", "edit", "glob", "grep", "bash_exec"}
    assert m6["edit"] == (["path", "old_string", "new_string"], ["path", "old_string", "new_string"])
    assert m6["grep"][0] == ["pattern"] and set(m6["grep"][1]) == {"pattern", "path"}
    m4 = schema_map(C4S)
    assert set(m4) == {"read", "write", "edit", "bash"}


def test_good_calls_pass_untouched():
    for schemas in (P6S, C4S):
        assert check_tool_args("read", {"path": "a.txt"}, schemas).error is None
        assert check_tool_args("edit", {"path": "a", "old_string": "x", "new_string": "y"}, schemas).error is None
        assert check_tool_args("write", {"path": "a", "content": "hi"}, schemas).error is None
    # empty-string CONTENT value stays legal for a new file (L2 note); the gate must not block it
    ok = check_tool_args("write", {"path": "new.txt", "content": ""}, P6S)
    assert ok.error is None and ok.clean == {"path": "new.txt", "content": ""}


def test_missing_required_key_is_refused():
    r = check_tool_args("read", {}, P6S)
    assert r.error is not None and "read needs keys path" in r.error and r.missing == ["path"]
    r = check_tool_args("write", {"path": "a"}, P6S)
    assert r.error is not None and r.missing == ["content"]


def test_wrong_case_key_is_refused_with_hint_exact():
    r = check_tool_args("edit", dict(path="src.py", old_String="return 1", new_String="return 2"), P6S)
    assert r.error == NEMOTRON_ERR
    assert r.missing == ["old_string", "new_string"] and "old_String" in r.got


def test_underscore_nearmiss_is_refused_with_hint():
    r = check_tool_args("write", {"path": "a", "Content": "hi"}, P6S)
    assert r.error is not None and "case-sensitive" in r.error and "content" in r.error


def test_non_string_value_is_refused():
    # S8f.1b cosmetic: the message says VALUES (the keys are fine; their values are not strings).
    r = check_tool_args("write", {"path": "a", "content": 123}, P6S)
    assert r.error == "ERROR: write values must be strings: content"


def test_empty_old_string_is_refused():
    r = check_tool_args("edit", {"path": "a", "old_string": "", "new_string": "X"}, P6S)
    assert r.error is not None and "old_string" in r.error


def test_extra_keys_tolerated_and_dropped():
    r = check_tool_args("write", {"path": "a", "content": "hi", "command": "rm"}, P6S)
    assert r.error is None and r.extra == ["command"] and r.clean == {"path": "a", "content": "hi"}


def test_unknown_tool_passes_straight_through():
    # parity: read_file is unknown (dialect_aliases off); the gate must not touch it so the
    # existing "ERROR: unknown tool read_file" still fires downstream.
    r = check_tool_args("read_file", {"path": "a"}, P6S)
    assert isinstance(r, ArgCheck) and r.error is None and r.clean == {"path": "a"}


def test_parity_core4_same_error_text_for_shared_tool():
    bad = dict(path="src.py", old_String="a", new_String="b")
    assert check_tool_args("edit", bad, P6S).error == check_tool_args("edit", bad, C4S).error == NEMOTRON_ERR


@pytest.mark.parametrize("profile,schemas", [("parity_lab6", P6S), ("core4", C4S)], ids=["parity", "core4"])
def test_matrix_each_advertised_tool_each_malformation(profile, schemas):
    """Each profile's OWN advertised tools (amendment B2) × {missing, wrong-case, non-string}."""
    m = schema_map(schemas)
    for name, (required, _props) in m.items():
        if not required:
            continue
        key = required[0]
        assert check_tool_args(name, {}, schemas).error is not None, f"{name} missing not caught"
        wrong = {k: "v" for k in required}
        wrong[key[:1].upper() + key[1:]] = wrong.pop(key)  # capitalise the first required key
        assert check_tool_args(name, wrong, schemas).error is not None, f"{name} wrong-case not caught"
        nonstr = {k: "v" for k in required}
        nonstr[key] = 5
        assert check_tool_args(name, nonstr, schemas).error is not None, f"{name} non-string not caught"


# ======================================================================= L2 executor (unit)

class _Stub:
    def __init__(self, d): self.d = dict(d)
    def get(self, p): return self.d.get(p)
    def put(self, p, c): self.d[p] = c; return "ok"
    def files(self): return list(self.d)


def test_executor_refuses_empty_path():
    ws = _Stub({"a": "x"})
    assert lab_run_tool("read", {"path": ""}, ws).startswith("ERROR:")
    assert lab_run_tool("write", {"path": "", "content": "y"}, ws).startswith("ERROR:")
    assert lab_run_tool("edit", {"path": "", "old_string": "x", "new_string": "y"}, ws).startswith("ERROR:")
    assert ws.d == {"a": "x"}  # nothing written under an empty key


def test_executor_refuses_empty_old_string():
    ws = _Stub({"a": "x"})
    assert lab_run_tool("edit", {"path": "a", "old_string": "", "new_string": "Z"}, ws).startswith("ERROR:")
    assert ws.d["a"] == "x"


def test_executor_edit_missing_file_is_not_found():
    ws = _Stub({})
    out = lab_run_tool("edit", {"path": "nofile.py", "old_string": "a", "new_string": "b"}, ws)
    assert out.startswith("ERROR: not found:") and "nofile.py" not in ws.d


def test_executor_edit_old_equals_new_is_refused():
    ws = _Stub({"a": "keep\n"})
    assert lab_run_tool("edit", {"path": "a", "old_string": "keep\n", "new_string": "keep\n"}, ws).startswith("ERROR:")
    assert ws.d["a"] == "keep\n"


def test_executor_write_missing_content_is_refused_but_empty_value_ok():
    ws = _Stub({})
    assert lab_run_tool("write", {"path": "a"}, ws).startswith("ERROR:")       # missing content key
    assert "a" not in ws.d
    assert lab_run_tool("write", {"path": "a", "content": ""}, ws) == "Wrote a"  # empty value legal (new file)
    assert ws.d["a"] == ""


def test_executor_grep_empty_pattern_is_refused():
    ws = _Stub({"a": "x\n"})
    assert lab_run_tool("grep", {"pattern": ""}, ws).startswith("ERROR:")


# ======================================================================= L3 honesty postcondition (unit)

def test_honesty_check_catches_noop_edit():
    from localbot.tools.fs import honesty_violation
    # edit claimed success but bytes identical -> violation
    assert honesty_violation("edit", before="x\n", after="x\n", written=None) is True
    assert honesty_violation("edit", before="x\n", after="y\n", written=None) is False
    # write claimed success but file != content written -> violation
    assert honesty_violation("write", before=None, after="hi", written="hi") is False
    assert honesty_violation("write", before=None, after="", written="hi") is True


# ======================================================================= rows 1-9 through the loop (parity)

ROWS = [
    # name, seed, call(name,args), expect_reply_predicate, file, disk_must_change
    ("r1_edit_wrongcase", {"src.py": "return 1\n"},
     call("edit", path="src.py", **{"old_String": "return 1", "new_String": "return 2"}), "err", "src.py", False),
    ("r2_edit_empty_old", {"src.py": "return 1\n"},
     call("edit", path="src.py", old_string="", new_string="X"), "err", "src.py", False),
    ("r3_edit_only_path", {"src.py": "return 1\n"},
     call("edit", path="src.py"), "err", "src.py", False),
    ("r4_write_wrongcase_content", {"keep.txt": "important\n"},
     call("write", path="keep.txt", **{"Content": "hi"}), "err", "keep.txt", False),
    ("r5_write_no_content", {},
     call("write", path="new.txt"), "err", "new.txt", False),
    ("r6_edit_missing_file_empty_old", {},
     call("edit", path="nofile.py", old_string="", new_string="created"), "err", "nofile.py", False),
    ("r7_edit_old_eq_new", {"src.py": "return 1\n"},
     call("edit", path="src.py", old_string="return 1", new_string="return 1"), "err", "src.py", False),
    ("r8_read_wrongcase_path", {"src.py": "return 1\n"},
     call("read", **{"Path": "src.py"}), "err", "src.py", False),
    ("r9_write_wrongcase_path", {"src.py": "return 1\n"},
     call("write", **{"Path": "out.txt", "content": "hi"}), "err", "src.py", False),
]


@pytest.mark.parametrize("name,seed,c,kind,fpath,changed", ROWS, ids=[r[0] for r in ROWS])
def test_rows_1_to_9_on_new_code(disk, name, seed, c, kind, fpath, changed):
    ws = disk.ws(seed)
    before = (disk.root / fpath).read_text(encoding="utf-8") if (disk.root / fpath).exists() else None
    res, _ = run(P6, [reply(calls=[c]), reply("done")], ws)
    reply_text = res.tool_results[0]
    assert reply_text.startswith("ERROR:"), f"{name}: expected refusal, got {reply_text!r}"
    after = (disk.root / fpath).read_text(encoding="utf-8") if (disk.root / fpath).exists() else None
    assert (before != after) == changed, f"{name}: disk change mismatch"
    # no false success survived anywhere in the try
    assert not any(t.startswith("Updated") or t.startswith("Wrote") for t in res.tool_results), name


# ======================================================================= regression + property + ordering + records

def test_exact_nemotron_regression_through_loop(disk):
    ws = disk.ws({"src.py": "return 1\n"})
    res, _ = run(P6, [reply(calls=[call("edit", path="src.py", **{"old_String": "return 1", "new_String": "return 2"})]),
                      reply("done")], ws)
    assert res.tool_results == [NEMOTRON_ERR]
    assert (disk.root / "src.py").read_text(encoding="utf-8") == "return 1\n"
    assert res.arg_errors and res.arg_errors[0]["tool"] == "edit"
    assert res.arg_errors[0]["missing"] == ["old_string", "new_string"]


def test_parity_core4_same_reply_through_loop(disk):
    bad = call("edit", path="src.py", **{"old_String": "a", "new_String": "b"})
    ws6 = disk.ws({"src.py": "return 1\n"})
    r6, _ = run(P6, [reply(calls=[bad]), reply("done")], ws6)
    assert r6.tool_results == [NEMOTRON_ERR]


def test_extra_key_is_logged_and_guard_still_refuses(disk):
    # write to a host path WITH an extra 'command' key: the gate drops the extra, the path guard
    # still refuses, and the veto is still counted (brief 1b: the 10 all-required+extra calls).
    ws = disk.ws({"keep.txt": "k"})
    res, _ = run(P6, [reply(calls=[call("write", path="/etc/passwd", content="x", command="rm")]),
                      reply("done")], ws)
    assert res.tool_results[0].startswith("ERROR: harness:")  # the guard refused, not the gate
    assert res.vetoes == 1
    assert res.arg_extra and res.arg_extra[0]["tool"] == "write" and res.arg_extra[0]["extra"] == ["command"]


def test_arg_error_is_not_counted_as_veto_or_executed(disk):
    ws = disk.ws({"src.py": "return 1\n"})
    res, _ = run(P6, [reply(calls=[call("read", **{"Path": "src.py"})]), reply("done")], ws)
    assert res.vetoes == 0  # an arg refusal is recorded in arg_errors, not vetoes
    # H1 P2 (and audit E-8: this was `== "model_no_call" or is None`, true either way): the model
    # did call; the harness refused the call, so the reason says so.
    assert res.no_call_reason == "call_refused"
    assert res.arg_errors and res.arg_errors[0]["tool"] == "read"


def test_property_any_success_reply_means_disk_changed_as_claimed(disk):
    ws = disk.ws({"a.txt": "one\n", "keep.txt": "k\n"})
    script = [
        reply(calls=[call("write", path="b.txt", content="new\n")]),          # valid write
        reply(calls=[call("edit", path="a.txt", old_string="one", new_string="two")]),  # valid edit
        reply(calls=[call("edit", path="a.txt", **{"old_String": "two", "new_String": "x"})]),  # malformed
        reply("done"),
    ]
    res, _ = run(P6, script, ws)
    snap = ws.snapshot()
    for (nm, args), result in zip(res.tool_calls, res.tool_results):
        if result.startswith("Wrote"):
            assert snap[args["path"]] == args["content"], f"write lied: {args['path']}"
        if result.startswith("Updated"):
            assert args["old_string"] != args["new_string"]  # a real change was claimed


def test_summary_exposes_arg_errors_and_false_success(disk):
    ws = disk.ws({"src.py": "return 1\n"})
    res, _ = run(P6, [reply(calls=[call("read", **{"Path": "src.py"})]), reply("done")], ws)
    s = res.summary()
    assert "arg_errors" in s and "arg_extra" in s and "false_success" in s
    assert len(s["arg_errors"]) == 1
