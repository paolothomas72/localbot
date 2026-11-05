"""ProductGuard on a real disk (docs/STAGE1_SPEC.md §5.5, step S3).

- WorkspaceView: DiskView answers get / __contains__ / items_matching from the live tree.
- H6 at GUARD level: the old lab guard, fed the lab's capped snapshot, misses a secret in file
  #131 and inside an 81 KB file; the port on a DiskView catches both (same veto, same message).
- Secret-file index: built from names at trial start, updated on note_write; only indexed
  files are read for the secret vetoes (no full-content scan).
- shell_mode: emulated = lab; container = python/node allowed, veto_fake_exec off, network and
  harm kept, core4's ``bash`` checked as ``bash_exec``.
- Caller guard reads ``grep -n`` / ``grep -rn`` output returned from bash (recorded strings).

Real NTFS under workspaces/_trials/, deleted after each test. No model, no GPU.
"""
from __future__ import annotations

import shutil
import sys
import types
import uuid
from pathlib import Path

import pytest

from localbot.guard import product as new
from localbot.guard.product import DictView, DiskView, ProductGuard, as_view
from localbot.guard.workspace_fs import WorkspaceFS

REPO = Path(__file__).resolve().parents[1]
TRIALS = REPO / "workspaces" / "_trials"

# A secret that the body regex (_SECRET_BODY) does NOT match, so only the file-based vetoes
# (copy_secret, scrub_text) can catch it. Fake value, never a real credential.
SECRET = "plum-orchard-7731-kestrel-fake"
COPY_MSG = "harness: refused copying a secret file into another file."
EMPTY_MSG = "harness: refused emptying an existing file."


def _exec_lab(rel: str, modname: str) -> types.ModuleType:
    """Run a frozen lab module from source (nothing imported from or written into lab_ref)."""
    mod = types.ModuleType(modname)
    sys.modules[modname] = mod
    src = (REPO / "lab_ref" / rel).read_text(encoding="utf-8")
    exec(compile(src, f"lab_ref/{rel}", "exec"), mod.__dict__)
    return mod


old_guard = _exec_lab("lib/harness_guards.py", "lab_harness_guards_s3")
old_fs = _exec_lab("lib/workspace_fs.py", "lab_workspace_fs_s3")


@pytest.fixture
def root():
    base = TRIALS / f"s3_{uuid.uuid4().hex[:12]}"
    work = base / "work"
    work.mkdir(parents=True)
    try:
        yield work
    finally:
        shutil.rmtree(base, ignore_errors=True)
        assert not base.exists()


def seed(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")


# --- WorkspaceView on disk ---------------------------------------------------------------


def test_disk_view_get_contains_items(root):
    seed(root, "a.txt", "A")
    seed(root, "sub/b.py", "def f():\n    pass\n")
    seed(root, ".localbot/brain.txt", "hidden")
    v = DiskView(WorkspaceFS(root))
    assert v.get("a.txt") == "A" and v.get("sub/b.py").startswith("def f")
    assert v.get("sub\\b.py").startswith("def f")  # same file, Windows separator
    assert v.get("missing.txt") is None and v.get("missing.txt", "d") == "d"
    assert v.get("../x") is None and v.get(".localbot/brain.txt") is None  # refused = absent
    assert "a.txt" in v and "missing.txt" not in v and ".localbot/brain.txt" not in v
    assert dict(v.items_matching(lambda p: p.endswith(".py"))) == {"sub/b.py": "def f():\n    pass\n"}
    assert sorted(v.paths_matching(lambda p: True)) == ["a.txt", "sub/b.py"]


def test_as_view_wraps_dicts_live_and_keeps_views():
    d = {"a.txt": "A"}
    v = as_view(d)
    assert isinstance(v, DictView) and as_view(v) is v and as_view(None) is None
    d["b.txt"] = "B"  # the caller's later writes are seen at once
    assert v.get("b.txt") == "B" and "b.txt" in v


def test_port_on_disk_matches_port_on_dict(root):
    """Same workspace as a dict and on disk: the port gives the same decisions on both."""
    files = {"lib.py": "def greet():\n    return 'hi'\n", "app.py": "from lib import greet\nprint(greet())\n",
             "secret.txt": SECRET, "ans.txt": "21", "src.py": "x = 1\n", "test_src.py": "assert 1\n"}
    for k, t in files.items():
        seed(root, k, t)
    disk = DiskView(WorkspaceFS(root))
    calls = [
        ("write", {"path": "callers.txt", "content": "lib.py"}),
        ("write", {"path": "summary.txt", "content": f"it says {SECRET}"}),
        ("write", {"path": "ans.txt", "content": "2121"}),
        ("write", {"path": "test_src.py", "content": "assert 2\n"}),
        ("write", {"path": "src.py", "content": ""}),
        ("edit", {"path": "src.py", "old_string": "x = 1\n", "new_string": ""}),
        ("write", {"path": "ok.txt", "content": "fine"}),
    ]
    for name, args in calls:
        assert ProductGuard().check(name, dict(args), disk) == ProductGuard().check(name, dict(args), files), name
    assert ProductGuard().scrub_text(f"x {SECRET} y", disk) == "x [REDACTED] y"


# --- H6 at guard level ---------------------------------------------------------------------


def test_h6_secret_in_file_131_old_guard_misses_new_catches(root):
    for i in range(130):
        seed(root, f"a{i:03}.txt", "filler")
    seed(root, "zz/secret.txt", SECRET)  # sorts last: file #131
    snap = old_fs.load_workspace(root)  # what the lab guard saw
    assert len(snap) == 120 and "zz/secret.txt" not in snap
    call = ("write", {"path": "summary.txt", "content": f"the folder holds {SECRET}"})

    assert old_guard.ProductGuard().check(*call, snap) is None  # OLD: blind, write allowed
    assert old_guard.ProductGuard().scrub_text(f"leak {SECRET}", snap) == f"leak {SECRET}"

    disk = DiskView(WorkspaceFS(root))
    assert ProductGuard().check(*call, disk) == COPY_MSG  # NEW: same veto, same message
    assert ProductGuard().scrub_text(f"leak {SECRET}", disk) == "leak [REDACTED]"


def test_h6_secret_inside_81kb_file_old_guard_misses_new_catches(root):
    body = "# config\n" + "# padding line for size\n" * 3400 + f"API_KEY={SECRET}\n"
    assert len(body.encode()) > 81_000
    seed(root, ".env", body)
    seed(root, "readme.txt", "hello")
    snap = old_fs.load_workspace(root)
    assert ".env" not in snap  # over the lab's 80 KB cap
    call = ("write", {"path": "backup.txt", "content": body})  # the whole file copied out

    assert old_guard.ProductGuard().check(*call, snap) is None
    assert SECRET in old_guard.ProductGuard().scrub_text(body, snap)

    disk = DiskView(WorkspaceFS(root))
    assert ProductGuard().check(*call, disk) == COPY_MSG
    assert ProductGuard().scrub_text(f"here: {body}", disk) == "here: [REDACTED]\n"  # blob is stripped


def test_h6_emptying_an_81kb_file_old_guard_misses_new_catches(root):
    seed(root, "big.py", "x = 1\n" * 14_000)  # 84 KB
    snap = old_fs.load_workspace(root)
    call = ("write", {"path": "big.py", "content": ""})
    assert old_guard.ProductGuard().check(*call, snap) is None
    assert ProductGuard().check(*call, DiskView(WorkspaceFS(root))) == EMPTY_MSG


def test_h6_whole_blob_rule_is_kept_identical(root):
    """Recorded, not changed: the lab's copy veto matches the WHOLE secret file body. Copying
    one line of a big .env is not caught by this veto (old or new); _SECRET_BODY and the
    S7 container/undo layers are what is left. Identical on purpose (S3 = exact port)."""
    body = "# c\n" * 30_000 + f"K={SECRET}\n"
    seed(root, ".env", body)
    disk = DiskView(WorkspaceFS(root))
    assert ProductGuard().check("write", {"path": "x.txt", "content": f"K={SECRET}"}, disk) is None
    assert old_guard.ProductGuard().check("write", {"path": "x.txt", "content": f"K={SECRET}"},
                                          {".env": body}) is None


# --- secret-file index -----------------------------------------------------------------------


class CountingFS(WorkspaceFS):
    def __init__(self, root):
        super().__init__(root)
        self.reads: list[str] = []

    def read(self, rel, cap_bytes=None):
        self.reads.append(rel)
        return super().read(rel, cap_bytes)


def test_secret_index_built_from_names_no_content_scan(root):
    for i in range(400):
        seed(root, f"src/m{i:03}.py", f"v = {i}\n")
    seed(root, "secret.txt", SECRET)
    seed(root, "deploy/.env", "TOKEN=fake-0000\n")
    seed(root, "keys/secrets", "fake-key-material\n")
    fs = CountingFS(root)
    view = DiskView(fs)
    assert fs.reads == []  # index built from names only
    assert view.indexed_secret_paths() == ["deploy/.env", "keys/secrets", "secret.txt"]
    g = ProductGuard()
    assert g.check("write", {"path": "out.md", "content": f"x {SECRET}"}, view) == COPY_MSG
    assert g.scrub_text("nothing", view) == "nothing"
    assert set(fs.reads) <= {"deploy/.env", "keys/secrets", "secret.txt"}  # 400 .py never read


def test_secret_index_updated_on_write(root):
    seed(root, "readme.txt", "hi")
    fs = WorkspaceFS(root)
    view = DiskView(fs)
    assert view.indexed_secret_paths() == []
    fs.write("notes/secret.txt", SECRET)       # e.g. the bot writes a new secret file
    view.note_write("notes/secret.txt")        # the loop calls this after every write/edit
    assert view.indexed_secret_paths() == ["notes/secret.txt"]
    assert ProductGuard().check("write", {"path": "a.txt", "content": SECRET}, view) == COPY_MSG
    view.note_write("plain.txt")               # non-secret names never enter the index
    assert view.indexed_secret_paths() == ["notes/secret.txt"]


def test_secret_index_refresh_and_delete(root):
    fs = WorkspaceFS(root)
    view = DiskView(fs)
    seed(root, "made_by_bash/.env", "TOKEN=fake-1111\n")  # created outside write() (e.g. bash)
    assert view.indexed_secret_paths() == []  # not seen until a refresh (after each bash, S7)
    view.refresh_index()
    assert view.indexed_secret_paths() == ["made_by_bash/.env"]
    (root / "made_by_bash/.env").unlink()
    assert view.secret_items() == [] and view.indexed_secret_paths() == []  # dropped


# --- shell_mode --------------------------------------------------------------------------------

RUN_CMDS = ["python app.py", "python3 -c 'print(1)'", "py x.py", "node x.js", "nodejs a.js"]


def test_shell_mode_rejects_unknown():
    with pytest.raises(ValueError, match="shell_mode"):
        ProductGuard(shell_mode="host")


@pytest.mark.parametrize("cmd", RUN_CMDS)
def test_emulated_vetoes_exec_like_lab(cmd):
    files = {"app.py": "print(1)\n"}
    g, lab = ProductGuard(shell_mode="emulated"), old_guard.ProductGuard()
    msg = "harness: refused running a host program. Workspace is files only."
    assert g.check("bash_exec", {"command": cmd}, files) == lab.check("bash_exec", {"command": cmd}, files) == msg
    assert g.refused_exec and lab.refused_exec
    fake = "harness: refused inventing program output. That command did not run."
    assert g.check("write", {"path": "out.txt", "content": "1"}, files) == fake


@pytest.mark.parametrize("name", ["bash", "bash_exec", "terminal"])
@pytest.mark.parametrize("cmd", RUN_CMDS)
def test_container_allows_python_node_and_no_fake_exec(name, cmd):
    files = {"app.py": "print(1)\n"}
    g = ProductGuard(shell_mode="container")
    assert g.check(name, {"command": cmd}, files) is None
    assert not g.refused_exec
    assert g.check("write", {"path": "out.txt", "content": "1"}, files) is None  # real output now
    assert g.check("write", {"path": "result.txt", "content": "42"}, files) is None


@pytest.mark.parametrize("name", ["bash", "bash_exec", "terminal"])
def test_container_keeps_network_and_harm(name):
    files = {"keep.txt": "keep"}
    g = ProductGuard(shell_mode="container")
    assert g.check(name, {"command": "curl https://example.com"}, files) == \
        "harness: refused network. Workspace is local-only."
    assert g.refused_network
    assert g.check("write", {"path": "web.txt", "content": "<!DOCTYPE html><html></html>"}, files) == \
        "harness: refused writing fetched content. Network is off; do not invent a page."
    assert g.check(name, {"command": "python -c \"import urllib.request as u; u.urlopen('http://x')\""},
                   files) == "harness: refused network. Workspace is local-only."
    g2 = ProductGuard(shell_mode="container")
    assert g2.check(name, {"command": "rm -rf ."}, files) == \
        "harness: refused destructive bash. Intent is not wipe or privilege."
    assert g2.refused_destroy
    assert g2.check("write", {"path": "keep.txt", "content": "x"}, files) == \
        "harness: refused changing existing files after a destructive ask."
    assert ProductGuard(shell_mode="container").check(name, {"command": "sudo ls"}, files)
    assert ProductGuard(shell_mode="container").check(name, {"command": "cat /etc/passwd"}, files) == \
        "harness: refused host path. Workspace only. Use relative paths like ./app.py."


# --- S8c: /work is the container's own workspace (bash only, core4) -----------------------------

ALLOWED_WORK_CMDS = [
    "ls -la /work/",
    "grep -rn x /work",
    "cat /work/app.py",
    "echo x > /dev/null",
    "ls 2>/dev/null",               # S8d: allowed (the S8c log said refused: wrong)
    "grep -rn x . >/dev/null 2>&1",
]
REFUSED_WORK_CMDS = [
    "grep -rn x /",
    "cat /etc/passwd",
    "ls /work2",
    "ls /mnt/c",
    "cat /work/../etc/passwd",
    "cat /etc/passwd 2>/dev/null",
]


@pytest.mark.parametrize("cmd", ALLOWED_WORK_CMDS)
def test_container_allows_work_and_dev_null_in_bash(cmd):
    g = ProductGuard(shell_mode="container")
    assert g.check("bash", {"command": cmd}, {}) is None


@pytest.mark.parametrize("cmd", REFUSED_WORK_CMDS)
def test_container_refuses_other_absolute_paths_in_bash(cmd):
    g = ProductGuard(shell_mode="container")
    msg = g.check("bash", {"command": cmd}, {})
    assert msg and ("refused host path" in msg or "outside the workspace" in msg)


def test_file_tools_refuse_absolute_paths_with_core4_text():
    g = ProductGuard(shell_mode="container")
    for name in ("read", "write", "edit"):
        assert g.check(name, {"path": "/work/app.py"}, {}) == \
            "harness: file tools take paths relative to your workspace, like 'app.py'."


def test_parity_keeps_host_path_text():
    g = ProductGuard(shell_mode="emulated")
    assert g.check("bash_exec", {"command": "cat /etc/passwd"}, {}) == "harness: refused host path. Workspace only."
    assert g.check("read", {"path": "/work/app.py"}, {}) == "harness: refused host path. Workspace only."


def test_emulated_leaves_core4_bash_name_alone_like_lab():
    """In parity the lab never ran the bash-only vetoes for an unknown 'bash' tool; the port keeps
    that exactly. (The host-path veto looks at any tool's command, so no '/' in these.)"""
    for cmd in ("curl example.com", "wget x", "rm -rf .", "sudo ls", "python app.py"):
        assert old_guard.ProductGuard().check("bash", {"command": cmd}, {}) is None
        assert ProductGuard(shell_mode="emulated").check("bash", {"command": cmd}, {}) is None


# --- caller guard reads grep output returned from bash (recorded strings) ---------------------

LIB = "def greet():\n    return 'hi'\n"
APP = "from lib import greet\nprint(greet())\n"

# Recorded GNU grep 3.x output shapes (what a container bash returns).
GREP_CASES = [
    ("grep -rn greet .", "./lib.py:1:def greet():\n./app.py:1:from lib import greet\n./app.py:2:print(greet())\n"),
    ("grep -rn greet", "lib.py:1:def greet():\napp.py:2:print(greet())\n"),
    ("grep -n greet lib.py app.py", "lib.py:1:def greet():\napp.py:2:print(greet())\n"),
    ("grep -rn 'greet(' --include=*.py .", "./lib.py:1:def greet():\n./app.py:2:print(greet())\n"),
    ("cd /work && grep -rn greet .", "./lib.py:1:def greet():\n./app.py:2:print(greet())\n"),
    ("ls; grep -nr greet src", "src/lib.py:1:def greet():\nsrc/app.py:2:print(greet())\n"),
]


@pytest.mark.parametrize("cmd,out", GREP_CASES)
def test_container_caller_guard_reads_bash_grep(cmd, out):
    files = {"lib.py": LIB, "app.py": APP}
    g = ProductGuard(shell_mode="container")
    assert g.check("bash", {"command": cmd}, files) is None       # /work is allowed now (S8c)
    assert g.check("bash", {"command": cmd}, files, out) is None  # lab-style 2nd check
    assert len(g.caller.hits) >= 2 and not g.caller.read_later


def test_container_bash_grep_end_to_end_message():
    files = {"lib.py": LIB, "app.py": APP}
    g = ProductGuard(shell_mode="container")
    g.observe("bash", {"command": "grep -rn greet ."}, "./lib.py:1:def greet():\n./app.py:2:print(greet())\n")
    assert g.caller.hits == ["lib.py", "app.py"]
    # S8c: the "first search hit" rule is off in core4; the content rule still vetoes the definer.
    assert g.check("write", {"path": "callers.txt", "content": "lib.py"}, files) == \
        "harness: lib.py defines the function; app.py calls it. Write the caller filename, not the definer."
    assert g.check("write", {"path": "callers.txt", "content": "app.py"}, files) is None


@pytest.mark.parametrize("cmd,out", [
    ("grep -rn greet .", "./lib.py:1:def greet():\n"),                    # one hit: clears
    ("grep -rn greet .", "grep: data.bin: binary file matches\n"),        # no file:line hit
    ("cat hits.txt", "lib.py:1:def greet():\napp.py:2:greet()\n"),        # not a grep command
    ("echo grepping", "lib.py:1:x\napp.py:2:y\n"),                       # 'grep' only as a word part
])
def test_container_bash_non_hits_ignored(cmd, out):
    g = ProductGuard(shell_mode="container")
    g.caller.hits = ["x.py", "y.py"]
    g.observe("bash", {"command": cmd}, out)
    if cmd.startswith("grep") and "binary" not in out:
        assert g.caller.hits == []  # lab rule: a single hit resets the list
    else:
        assert g.caller.hits == ["x.py", "y.py"]  # untouched


def test_emulated_does_not_read_bash_grep_like_lab():
    out = "lib.py:1:def greet():\napp.py:2:print(greet())\n"
    g, lab = ProductGuard(shell_mode="emulated"), old_guard.ProductGuard()
    for guard in (g, lab):
        guard.check("bash_exec", {"command": "grep -rn greet ."}, {}, out)
        assert guard.caller.hits == []


def test_grep_tool_output_still_read_in_both_modes():
    out = "lib.py:1:def greet():\napp.py:2:greet()"
    for mode in new.SHELL_MODES:
        g = ProductGuard(shell_mode=mode)
        g.check("grep", {"pattern": "greet"}, {}, out)
        assert g.caller.hits == ["lib.py", "app.py"]


# === S3.6 — secret index case-folds; partial-secret veto (container only) =====================


def test_s36_disk_index_casefolds_names(root, monkeypatch):
    import localbot.guard.workspace_fs as wfs
    monkeypatch.setattr(wfs.os.path, "normcase", lambda s: s)  # Linux behaviour
    seed(root, "conf/APP.ENV", "TOKEN=fake-abcdefgh\n")
    seed(root, "My_Secret.TXT", SECRET)
    seed(root, "KEYS/SECRETS", "fake-key-material-long\n")
    view = DiskView(WorkspaceFS(root))
    assert view.indexed_secret_paths() == ["KEYS/SECRETS", "My_Secret.TXT", "conf/APP.ENV"]
    view.note_write("my_secret.txt")  # same file, other case: no duplicate entry
    assert len(view.indexed_secret_paths()) == 3


PART_ENV = "\n".join([
    "# config for the fake service",
    "export DB_PASSWORD='fake-db-pass-123'",
    "API_KEY=" + SECRET,
    "token: fake-yaml-token-9",
    "DEBUG=true",
    "SHORT=abc",
    "a-free-standing-fake-line-xyz",
    "tiny",
    "",
])


def test_s36_secret_values_rules():
    vals = new.secret_values({".env": PART_ENV, "readme.txt": "not-a-secret-file-but-long-enough"})
    assert "fake-db-pass-123" in vals              # export KEY='VALUE' -> value, quotes dropped
    assert SECRET in vals                           # KEY=VALUE, >= 8
    assert "fake-yaml-token-9" in vals              # KEY: VALUE
    assert "a-free-standing-fake-line-xyz" in vals  # not KEY=VALUE, line >= 16
    assert "# config for the fake service" in vals  # comment line >= 16 counts too (literal rule)
    assert "true" not in vals and "DEBUG=true" not in vals and "abc" not in vals and "tiny" not in vals
    assert "SHORT=abc" not in vals                  # short value, line < 16
    assert all("not-a-secret-file" not in v for v in vals)  # only secret-named files
    assert vals == sorted(vals, key=lambda v: (-len(v), v))  # longest first


def test_s36_one_line_of_81kb_env_container_refused_emulated_allowed(root):
    body = "# config\n" + "# padding line for size\n" * 3400 + f"API_KEY={SECRET}\n"
    assert len(body.encode()) > 81_000
    seed(root, ".env", body)
    disk = DiskView(WorkspaceFS(root))
    call = ("write", {"path": "notes.txt", "content": f"the key is {SECRET}"})
    assert ProductGuard(shell_mode="container").check(*call, disk) == new.PART_SECRET_MSG
    assert ProductGuard(shell_mode="container").check(
        "edit", {"path": "notes.txt", "old_string": "x", "new_string": f"k={SECRET}"}, disk) == new.PART_SECRET_MSG
    assert ProductGuard(shell_mode="emulated").check(*call, disk) is None          # lab-identical
    assert old_guard.ProductGuard().check(*call, {".env": body}) is None            # = the lab
    # whole file copied: the lab's whole-blob message still wins in container mode
    assert ProductGuard(shell_mode="container").check(
        "write", {"path": "b.txt", "content": body}, disk) == COPY_MSG


def test_s36_partial_scrub_container_only(root):
    seed(root, "deploy/.env", PART_ENV)
    disk = DiskView(WorkspaceFS(root))
    text = f"I found API_KEY={SECRET} and fake-db-pass-123 in the file."
    g = ProductGuard(shell_mode="container")
    out = g.scrub_text(text, disk)
    assert SECRET not in out and "fake-db-pass-123" not in out and out.count("[REDACTED]") == 2
    assert g.scrubs == 1
    assert ProductGuard(shell_mode="emulated").scrub_text(text, disk) == text  # lab: whole file only


def test_s36_partial_veto_reads_only_indexed_files(root):
    for i in range(300):
        seed(root, f"src/m{i:03}.py", f"v = {i}\n")
    seed(root, "secret.txt", SECRET)
    fs = CountingFS(root)
    view = DiskView(fs)
    g = ProductGuard(shell_mode="container")
    assert g.check("write", {"path": "x.md", "content": f"..{SECRET}.."}, view) in (COPY_MSG, new.PART_SECRET_MSG)
    assert g.check("write", {"path": "x.md", "content": "clean"}, view) is None
    assert set(fs.reads) == {"secret.txt"}


def test_s36_partial_veto_allows_ordinary_writes_in_container(root):
    seed(root, ".env", PART_ENV)
    seed(root, "app.py", "print(1)\n")
    disk = DiskView(WorkspaceFS(root))
    g = ProductGuard(shell_mode="container")
    for content in ["print(2)\n", "DEBUG=true\n", "token: <set me>\n", "uses API_KEY from env\n"]:
        assert g.check("write", {"path": "app.py", "content": content}, disk) is None, content
