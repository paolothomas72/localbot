"""H1 G-3: one secret-file name list (``localbot/guard/secret_names.py``), shared by the core4 guard's
secret index, the write tool's secrets-path rule and the same rule on bash-made files. Host side
only. Parity (emulated) keeps the lab's rule exactly: only ``secret.txt`` / ``.env`` / ``secrets``
suffixes.

The audit's proof: ``cp .env.local notes.txt`` stood (vetoes 0, notes.txt held the value), because
``.env.local``, ``secrets.json``, ``credentials``, ``id_rsa``, ``.npmrc`` were not indexed.
"""
from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from localbot.config import UndoConfig, get_profile
from localbot.guard.product import (
    PART_SECRET_MSG,
    SECRETS_PATH_MSG,
    DiskView,
    ProductGuard,
    path_rule,
    secret_values,
)
from localbot.guard.secret_names import SECRET_GLOBS, SecretNames
from localbot.guard.undo import UndoStore
from localbot.guard.workspace_fs import WorkspaceFS, remove_tree
from localbot.loop import BASH_MADE, run_task
from localbot.tools.fs import DiskWorkspace
from tests.fakes import FakeShell, ScriptedTransport, call, provider, reply

TRIALS = Path(__file__).resolve().parents[1] / "workspaces" / "_trials"
VALUE = "plum-orchard-7731-kestrel-fake"  # a fake value, like the lab canary

SECRET = [".env.local", ".env.production", "config/.env.development.local", "secrets.json",
          "secrets.yaml", "credentials", ".aws/credentials", "credentials.json", "id_rsa",
          "id_ed25519", "server.pem", "tls.key", "cert.p12", ".npmrc", ".pypirc", ".netrc",
          ".git-credentials", ".pgpass", ".htpasswd",
          ".env", "prod.env", "secret.txt", "topsecret.txt", "app_secrets", "config/secrets",  # lab ones
          # H1-K K6 D4 (owner decision, fail toward hiding): secrets.* / credentials.* with ANY
          # extension, so a module named credentials.py is a secret file too.
          "secrets.cfg", "secrets.conf", "conf/credentials.ini", "credentials.py"]
NOT_SECRET = [".env.example", ".env.sample", ".env.template", "config/.env.dist", "id_rsa.pub",
              "secrets.example", "credentials.sample",  # templates win, as for .env.example
              "docs/secrets_management.md", "secrets_test.py", "notes.txt",
              "app.py", "README.md", "keys.md", "environment.yml"]


@pytest.fixture
def work():
    base = TRIALS / f"h1g3_{uuid.uuid4().hex[:10]}"
    (base / "work").mkdir(parents=True)
    (base / "state").mkdir()
    try:
        yield base
    finally:
        remove_tree(base)


@pytest.mark.parametrize("rel", SECRET)
def test_secret_names_match(rel):
    assert SecretNames().matches(rel) and SecretNames().matches(rel.upper())


@pytest.mark.parametrize("rel", NOT_SECRET)
def test_secret_names_leave_templates_and_code_alone(rel):
    assert not SecretNames().matches(rel)


def test_secret_names_are_configurable():
    names = SecretNames(globs=(*SECRET_GLOBS, "*.vault"))
    assert names.matches("prod.vault") and not SecretNames().matches("prod.vault")


def test_container_guard_indexes_the_wide_list_parity_keeps_the_lab_rule(work):
    root = work / "work"
    for rel in [".env.local", "secrets.json", "id_rsa", ".env", ".env.example", "app.py"]:
        p = root / rel
        p.write_text(f"API_KEY={VALUE}\n", encoding="utf-8")
    core4 = DiskView(WorkspaceFS(root))
    ProductGuard(shell_mode="container").attach(core4)
    assert core4.indexed_secret_paths() == [".env", ".env.local", "id_rsa", "secrets.json"]
    parity = DiskView(WorkspaceFS(root))
    ProductGuard(shell_mode="emulated").attach(parity)
    assert parity.indexed_secret_paths() == [".env"]


@pytest.mark.parametrize("rel", [".env.local", "id_rsa", "secrets.json", ".npmrc"])
def test_write_tool_refuses_a_wide_list_secret_path_in_container_mode(work, rel):
    view = DiskView(WorkspaceFS(work / "work"))
    assert ProductGuard(shell_mode="container").check("write", {"path": rel, "content": "x"}, view) \
        == SECRETS_PATH_MSG
    assert path_rule(rel, "container") == (SECRETS_PATH_MSG, False)
    assert path_rule(rel, "emulated") is None  # parity: the lab rule


def test_write_tool_allows_env_example(work):
    view = DiskView(WorkspaceFS(work / "work"))
    assert ProductGuard(shell_mode="container").check(
        "write", {"path": ".env.example", "content": "API_KEY=\n"}, view) is None


def _run(work, files: dict[str, str], cmd: str, effect):
    root = work / "work"
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    ws = DiskWorkspace(WorkspaceFS(root), order=list(files))
    prof = get_profile("core4")
    store = UndoStore(UndoConfig(store_dir=work / "state"), "bot1", root)
    script = [reply(calls=[call("bash", command=cmd)]), reply("done")]
    return run_task(prof, provider(ScriptedTransport(script), prof.provider), "fake:1", "go", ws,
                    shell=FakeShell({cmd: effect}), undo=store)


def test_cp_env_local_into_notes_is_undone_and_vetoed(work):
    """The audit's re-run: ``cp .env.local notes.txt`` used to stand (vetoes 0)."""
    root = work / "work"

    def cp() -> str:
        (root / "notes.txt").write_bytes((root / ".env.local").read_bytes())
        return ""

    res = _run(work, {".env.local": f"API_KEY={VALUE}\n", "keep.txt": "keep\n"}, "cp .env.local notes.txt", cp)
    assert res.vetoes == 1 and res.veto_texts == [f"ERROR: {PART_SECRET_MSG}"]
    assert not (root / "notes.txt").exists()


def test_bash_made_id_rsa_is_undone_with_the_write_tool_text(work):
    root = work / "work"

    def make() -> str:
        (root / "id_rsa").write_text("-----BEGIN OPENSSH " + "PRIVATE KEY-----\n", encoding="utf-8")
        return ""

    res = _run(work, {"keep.txt": "keep\n"}, "ssh-keygen -f id_rsa", make)
    assert res.veto_texts == [f"ERROR: {SECRETS_PATH_MSG} {BASH_MADE}"]
    assert not (root / "id_rsa").exists()


def test_chat_scrub_covers_a_wide_list_secret_file(work):
    root = work / "work"
    (root / "secrets.json").write_text(f'{{"token": "{VALUE}"}}\n', encoding="utf-8")
    (root / "credentials").write_text(f"aws_secret_access_key = {VALUE}x\n", encoding="utf-8")
    view = DiskView(WorkspaceFS(root))
    g = ProductGuard(shell_mode="container")
    out = g.scrub_text(f"the key is {VALUE}x", view)
    assert VALUE not in out and "[REDACTED]" in out


def _indexed(root: Path, name: str, body: str):
    dest = root / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(body, encoding="utf-8")
    view = DiskView(WorkspaceFS(root))
    guard = ProductGuard(shell_mode="container")
    guard.attach(view)
    return view, guard


def test_source_credentials_py_indexes_the_key_value_not_the_import(work):
    """CURSOR 001: a source-code secret file yields KEY = VALUE values only, not code lines."""
    fake = "n" * 20
    imp = "from pathlib import Path"
    view, _guard = _indexed(work / "work", "credentials.py", f"{imp}\nAPI_KEY = \"{fake}\"\n")
    vals = secret_values(view)
    assert imp not in vals
    assert fake in vals


@pytest.mark.parametrize("name", ["credentials.env", "credentials"])
def test_non_source_credentials_still_index_whole_lines(work, name):
    """``.env`` and a no-extension ``credentials`` file keep the whole-line rule."""
    fake = "n" * 20
    imp = "from pathlib import Path"
    view, _guard = _indexed(work / "work", name, f"{imp}\nAPI_KEY = \"{fake}\"\n")
    vals = secret_values(view)
    assert imp in vals
    assert fake in vals


def _shape(kind: str, fake: str) -> str:
    """A secret-file body. ``fake`` is built by the caller, never a literal key.

    The PEM armor is concatenated at run time so the source file does not contain the
    export scan's private-key literal.
    """
    armor = "-----" + "BEGIN PRIVATE KEY" + "-----"
    end = "-----" + "END PRIVATE KEY" + "-----"
    shapes = {
        "kv": f'API_KEY = "{fake}"\n',
        "dict-one-line": f'TOKENS = {{"github": "{fake}"}}\n',
        "dict-multi": f'TOKENS = {{\n    "github": "{fake}",\n}}\n',
        "paren": f'API_KEY = (\n    "{fake}"\n)\n',
        "environ": f'os.environ["GITHUB_TOKEN"] = "{fake}"\n',
        "list": f'TOKENS = [\n    "{fake}",\n]\n',
        "pem-body": f'KEY = """{armor}\n{fake}\n{end}"""\n',
        "dict-call": f'CONFIG = dict(\n    token="{fake}",\n)\n',
        "json": f'{{\n"api_key": "{fake}"\n}}\n',
    }
    return shapes[kind]


_SHAPE_CASES = [
    ("credentials.py", "kv"),
    ("credentials.py", "dict-one-line"),
    ("credentials.py", "dict-multi"),
    ("credentials.py", "paren"),
    ("credentials.py", "environ"),
    ("credentials.py", "list"),
    ("credentials.py", "pem-body"),
    ("credentials.py", "dict-call"),
    ("secrets.json", "json"),
    ("credentials.json", "json"),
]


@pytest.mark.parametrize("filename,kind", _SHAPE_CASES)
def test_secret_shape_refuses_the_bare_value_and_its_line(work, filename, kind):
    """CURSOR 001b: a container write of the bare value, and of the line that holds it, is refused."""
    fake = "n" * 20
    body = _shape(kind, fake)
    if filename.endswith(".py"):
        body = "from pathlib import Path\n" + body
    view, guard = _indexed(work / "work", filename, body)
    line = next(part.strip() for part in _shape(kind, fake).splitlines() if fake in part)
    assert fake in secret_values(view)
    assert guard.check("write", {"path": "notes.txt", "content": fake}, view) == PART_SECRET_MSG
    assert guard.check("write", {"path": "notes.txt", "content": line}, view) is not None


def _quote_row(row: int, fake: str) -> str:
    """One credentials.py body. ``fake`` is built by the caller."""
    rows = {
        1: f"# owner's token: TOKEN = \"{fake}\"",
        2: f"# it's owner's: TOKEN = \"{fake}\"",
        3: f"# don't: TOKEN = '{fake}'",
        4: f"TOKEN = \"{fake}\"  # owner's",
        5: f"# say \"hi: TOKEN = \"{fake}\"",
        6: f"# `cmd: TOKEN = \"{fake}\"",
        7: "TOKEN = \"a\\\"b" + fake + "\"",
        8: f"TOKEN = \"\"\"{fake}\"\"\"",
        9: f"TOKEN = f\"{fake}\"",
        10: "TOKEN = \u201c" + fake + "\u201d",
        11: "TOKEN\t=\t\"" + fake + "\"\r",
        12: f"# owner's notes\nTOKEN = \"{fake}\"",
        13: f"url = \"https://x/it's\"; TOKEN = \"{fake}\"",
    }
    return rows[row]


def _indexed_in(vals, text: str) -> bool:
    return any(v and v in text for v in vals)


@pytest.mark.parametrize("row", range(1, 14))
def test_quote_shapes_never_hide_a_later_value(work, row):
    """CURSOR 001c: a quote earlier on the line must not hide a later value.

    The write is refused when an indexed secret value occurs in it (the partial-secret
    veto). Row 10 uses smart quotes; the baseline did not index the bare value either.
    Row 7's value is the raw quoted interior (escapes are not decoded).
    """
    fake = "n" * 20
    body = _quote_row(row, fake)
    root = work / "work"
    (root / "credentials.py").write_bytes(body.encode("utf-8"))
    view = DiskView(WorkspaceFS(root))
    guard = ProductGuard(shell_mode="container")
    guard.attach(view)
    vals = secret_values(view)
    line = next(part.strip() for part in body.splitlines() if fake in part)
    bare = ("a\\\"b" + fake) if row == 7 else fake
    assert _indexed_in(vals, line), row
    assert guard.check("write", {"path": "notes.txt", "content": line}, view) is not None, row
    if row == 10:
        return
    assert _indexed_in(vals, bare), row
    assert guard.check("write", {"path": "notes.txt", "content": bare}, view) == PART_SECRET_MSG, row


def test_source_secret_file_does_not_index_a_short_token_or_spaced_code(work):
    """Rule 4: no-whitespace under 16 is not a value, and a code line with spaces is not either."""
    short = "n" * 15
    spaced = "from pathlib import Path\ndef main():\nimport os\n"
    view, guard = _indexed(work / "work", "credentials.py", f"{spaced}{short}\n")
    vals = secret_values(view)
    assert short not in vals
    for line in ("from pathlib import Path", "def main():", "import os"):
        assert line not in vals
        assert guard.check("write", {"path": "notes.txt", "content": line}, view) is None
    assert guard.check("write", {"path": "notes.txt", "content": short}, view) is None


def test_container_write_of_the_import_is_allowed_and_the_key_value_is_refused(work):
    fake = "n" * 20
    imp = "from pathlib import Path"
    view, guard = _indexed(work / "work", "credentials.py", f"{imp}\nAPI_KEY = \"{fake}\"\n")
    assert guard.check("write", {"path": "notes.txt", "content": imp + "\n"}, view) is None
    assert guard.check("write", {"path": "notes.txt", "content": fake}, view) == PART_SECRET_MSG


def test_d4_consequence_a_credentials_py_module_makes_its_code_lines_secret(work):
    """Review R5, changed by CURSOR 001 (same test, new rule). Owner decision D4 still stands:
    ``credentials.*`` matches any extension, so credentials.py stays a secret file. It used to index
    every line of 16+ characters, which refused an ordinary import copied into another file and
    redacted that line in chat. A source-code secret file now indexes only a KEY = VALUE value of
    8+ characters, never the whole line."""
    root = work / "work"
    fake = "n" * 20
    (root / "credentials.py").write_text(
        f"from pathlib import Path\n\nAPI_KEY = \"{fake}\"\n\ndef load(path):\n    return Path(path).read_text()\n",
        encoding="utf-8")
    view = DiskView(WorkspaceFS(root))
    g = ProductGuard(shell_mode="container")
    g.attach(view)
    assert view.indexed_secret_paths() == ["credentials.py"]
    code = "from pathlib import Path\n\nprint(Path('.').resolve())\n"
    assert g.check("write", {"path": "app.py", "content": code}, view) is None
    assert g.scrub_text("add from pathlib import Path at the top", view) == "add from pathlib import Path at the top"
    assert g.check("write", {"path": "app.py", "content": fake}, view) == PART_SECRET_MSG
    redacted = g.scrub_text(f"the token is {fake}", view)
    assert fake not in redacted and "[REDACTED]" in redacted
    assert g.check("write", {"path": "app.py", "content": "import os\n"}, view) is None  # under 16: not a value
    parity = DiskView(WorkspaceFS(root))
    ProductGuard(shell_mode="emulated").attach(parity)
    assert parity.indexed_secret_paths() == []  # the lab rule: credentials.py is plain code there
    other = work / "other"  # the same module under another name: the name alone causes it
    other.mkdir()
    (other / "store.py").write_bytes((root / "credentials.py").read_bytes())
    view2 = DiskView(WorkspaceFS(other))
    g2 = ProductGuard(shell_mode="container")
    g2.attach(view2)
    assert view2.indexed_secret_paths() == [] and g2.check("write", {"path": "app.py", "content": code}, view2) is None


def test_secret_txt_stays_index_only_as_in_the_lab(work):
    """The lab indexed ``secret.txt`` but never refused writing it; the exam's privacy judges score
    what a model does to it. So it is a secret file (indexed) but not a refused write path."""
    assert SecretNames().matches("secret.txt") and not SecretNames().matches_path_rule("secret.txt")
    view = DiskView(WorkspaceFS(work / "work"))
    assert ProductGuard(shell_mode="container").check(
        "write", {"path": "cfg/secret.txt", "content": "x"}, view) is None


@pytest.mark.parametrize("rel", [".env ", ".env.", "id_rsa ", "secrets.json."])
def test_trailing_dot_or_space_alias_is_the_same_secret_name(rel):
    """Audit P2: Win32 drops a trailing '.' or space, so ``.env `` opens ``.env``. The write tool's
    path layer refuses such names; a bash step can still make one, so the name rule folds them."""
    assert SecretNames().matches(rel) and path_rule(rel, "container") == (SECRETS_PATH_MSG, False)


def test_bash_made_env_with_a_trailing_space_is_undone(work):
    root = work / "work"
    from localbot.guard.workspace_fs import extended_path

    def make() -> str:
        (extended_path(root) / ".env ").write_text("API_KEY=x\n", encoding="utf-8")
        return ""

    res = _run(work, {"keep.txt": "keep\n"}, "echo API_KEY=x > '.env '", make)
    assert res.veto_texts == [f"ERROR: {SECRETS_PATH_MSG} {BASH_MADE}"]
    assert not (extended_path(root) / ".env ").exists()
