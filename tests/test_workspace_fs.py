"""Path-layer attack pack + hole repro (docs/STAGE1_SPEC.md §5.4, §7, step S2).

Each hole H1-H7 is tested twice: ``test_h*_hole_in_old_code`` reproduces the bad behaviour
against the FROZEN lab code (lab_ref/lib/workspace_fs.py), and ``test_h*_closed_in_new_code``
shows it refused by localbot.guard.workspace_fs. The rest is the §7 path-layer attack pack.

Real NTFS: every test runs in its own trial folder under workspaces/_trials/, deleted after.
Junctions are made with ``mklink /J`` (no admin). Symlinks use os.symlink; if Windows refuses
(no admin / Developer Mode), that test is skipped with the reason. No model, no GPU.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import subprocess
import uuid
from pathlib import Path

import pytest

from localbot.guard.workspace_fs import WorkspaceError, WorkspaceFS

REPO = Path(__file__).resolve().parents[1]
TRIALS = REPO / "workspaces" / "_trials"

# H1-J J5: tests of Windows-only behaviour are skipped elsewhere (the rules they test hold on every OS)
WIN = os.name == "nt"
JUNCTION = pytest.mark.skipif(not WIN, reason="mklink /J (junctions) is Windows-only")
SHORT_NAMES = pytest.mark.skipif(not WIN, reason="8.3 short names are NTFS / Windows-only")
ADS = pytest.mark.skipif(not WIN, reason="alternate data streams are NTFS-only")
TRAILING_DOT = pytest.mark.skipif(not WIN, reason="Win32 drops a trailing '.': the old-code hole is Windows-only")

# --- load the frozen lab code without importing the package or touching sys.path -----------
# Compiled from source (S4): spec_from_file_location's loader wrote a .pyc into
# lab_ref/lib/__pycache__/, and nothing may be written into lab_ref/.

old = importlib.util.module_from_spec(importlib.machinery.ModuleSpec("lab_workspace_fs", None))
exec(compile((REPO / "lab_ref" / "lib" / "workspace_fs.py").read_text(encoding="utf-8"),
             "lab_ref/lib/workspace_fs.py", "exec"), old.__dict__)


# --- a disposable trial folder on real disk ------------------------------------------------


class Trial:
    def __init__(self) -> None:
        self.base = TRIALS / f"s2_{uuid.uuid4().hex[:12]}"
        self.root = self.base / "work"
        self.outside = self.base / "outside"
        self.root.mkdir(parents=True)
        self.outside.mkdir(parents=True)
        (self.outside / "secret.txt").write_text("CANARY_pw_outside\n", encoding="utf-8")
        self._links: list[Path] = []
        self.new = WorkspaceFS(self.root)

    def seed(self, rel: str, text: str) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def junction(self, name: str, target: Path) -> Path:
        link = self.root / name
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"mklink /J failed: {(r.stderr or r.stdout).strip()}")
        self._links.append(link)
        return link

    def symlink(self, name: str, target: Path) -> Path:
        link = self.root / name
        try:
            os.symlink(target, link)  # file symlink
        except OSError as exc:
            pytest.skip(f"os.symlink refused (no admin / Developer Mode): "
                        f"winerror={getattr(exc, 'winerror', None)} {exc}")
        self._links.append(link)
        return link

    def short_name(self, rel: str) -> str:
        """The real 8.3 short name of an existing file (from ``dir /x``), or skip."""
        p = self.root / rel
        out = subprocess.run(["cmd", "/c", "dir", "/x", str(p.parent)],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            if p.name.lower() in line.lower():
                for tok in line.split():
                    if "~" in tok:
                        return tok
        pytest.skip("no 8.3 short name created on this volume")

    def cleanup(self) -> None:
        # Remove reparse points as links (never follow them into the target) before rmtree.
        for link in self._links:
            try:
                if link.is_dir():
                    os.rmdir(link)
                else:
                    os.unlink(link)
            except FileNotFoundError:
                pass  # H1-J J5: the test already moved / replaced it (on Linux the teardown errored here)
            except OSError:
                if WIN:
                    subprocess.run(["cmd", "/c", "rmdir", str(link)], capture_output=True)
        import shutil
        shutil.rmtree(self.base, ignore_errors=True)


@pytest.fixture
def trial():
    t = Trial()
    try:
        yield t
    finally:
        t.cleanup()
        assert not t.base.exists(), f"trial folder not deleted: {t.base}"


# --- H1: reparse points (symlink / junction) ------------------------------------------------


@JUNCTION
def test_h1_junction_hole_in_old_code(trial):
    """OLD load_workspace follows a junction and reads a file from outside the root."""
    trial.junction("jlink", trial.outside)
    loaded = old.load_workspace(trial.root)
    assert loaded.get("jlink/secret.txt", "").strip() == "CANARY_pw_outside"  # leaked


@JUNCTION
def test_h1_junction_closed_in_new_code(trial):
    trial.junction("jlink", trial.outside)
    assert "jlink/secret.txt" not in trial.new.walk()
    assert all("jlink" not in rel for rel in trial.new.walk())
    with pytest.raises(WorkspaceError, match="reparse point"):
        trial.new.read("jlink/secret.txt")


def test_h1_symlink_file_closed_in_new_code(trial):
    trial.symlink("slink.txt", trial.outside / "secret.txt")  # skips if no privilege
    with pytest.raises(WorkspaceError, match="reparse point"):
        trial.new.read("slink.txt")
    assert "slink.txt" not in trial.new.walk()


# --- H2: trailing dot / space ---------------------------------------------------------------


@TRAILING_DOT
def test_h2_trailing_dot_hole_in_old_code(trial):
    """OLD write_file('.env.') lands on '.env' (Windows strips the trailing dot)."""
    old.write_file(trial.root, ".env.", "CANARY_pw_env\n")
    assert (trial.root / ".env").read_text(encoding="utf-8").strip() == "CANARY_pw_env"


def test_h2_trailing_space_hole_in_old_code(trial):
    old.write_file(trial.root, ".env ", "CANARY_pw_env\n")
    assert (trial.root / ".env").read_text(encoding="utf-8").strip() == "CANARY_pw_env"


@pytest.mark.parametrize("name", [".env.", ".env ", "a.txt.", "dir./b.txt"])
def test_h2_closed_in_new_code(trial, name):
    with pytest.raises(WorkspaceError, match="ending in"):
        trial.new.resolve(name)


# --- H3: 8.3 short names --------------------------------------------------------------------


@SHORT_NAMES
def test_h3_short_name_hole_in_old_code(trial):
    """OLD resolve accepts an 8.3 alias and reads the long-named file behind it."""
    trial.seed("secretlongname.txt", "CANARY_long\n")
    sn = trial.short_name("secretlongname.txt")  # e.g. SECRET~1.TXT
    p = old.resolve(trial.root, sn)
    assert p.read_text(encoding="utf-8").strip() == "CANARY_long"  # read via the alias


def test_h3_short_name_pattern_closed_in_new_code(trial):
    for name in ("SECRET~1.TXT", "secret~1.txt", "foo~12"):
        with pytest.raises(WorkspaceError, match="short-name"):
            trial.new.resolve(name)


@SHORT_NAMES
def test_h3_long_name_of_existing_alias_rechecked(trial):
    """GetLongPathNameW: an existing target's real name is run through the name rules too."""
    trial.seed("secretlongname.txt", "CANARY_long\n")
    sn = trial.short_name("secretlongname.txt")
    # Refused by the ~N rule regardless, but this asserts the long-name recheck path runs:
    with pytest.raises(WorkspaceError):
        trial.new.resolve(sn)


# --- H4: alternate data streams -------------------------------------------------------------


@ADS
def test_h4_ads_hole_in_old_code(trial):
    """OLD write_file('a.txt:hidden') writes a hidden NTFS stream on a.txt."""
    old.write_file(trial.root, "a.txt:hidden", "CANARY_pw_stream")
    raw = subprocess.run(["cmd", "/c", "more", "<", str(trial.root / "a.txt") + ":hidden"],
                         capture_output=True, text=True, shell=True).stdout
    assert "CANARY_pw_stream" in raw


@pytest.mark.parametrize("name", ["a.txt:hidden", "a.txt:$DATA", "x:stream"])
def test_h4_ads_closed_in_new_code(trial, name):
    with pytest.raises(WorkspaceError, match="alternate data stream"):
        trial.new.resolve(name)


# --- H5: reserved device names --------------------------------------------------------------


@pytest.mark.parametrize("name", ["CON", "con", "nul.txt", "AUX", "PRN", "COM1", "lpt9.log"])
def test_h5_reserved_accepted_by_old_code(trial, name):
    """OLD resolve does not reject reserved device names (no name check at all)."""
    assert old.resolve(trial.root, name).name.lower() == name.lower()  # no raise = hole


@pytest.mark.parametrize("name", ["CON", "con", "nul.txt", "AUX", "PRN", "COM1", "lpt9.log", "COM9.a.b"])
def test_h5_reserved_closed_in_new_code(trial, name):
    with pytest.raises(WorkspaceError, match="reserved device"):
        trial.new.resolve(name)


# --- H6: guard snapshot caps (fix completes in S3 WorkspaceView) ----------------------------


def test_h6_byte_cap_hole_in_old_code(trial):
    """OLD load_workspace drops an 81 KB file, so the guard never sees its contents."""
    trial.seed("big.txt", "x" * 81_000)
    assert "big.txt" not in old.load_workspace(trial.root)  # blind spot


def test_h6_file_cap_hole_in_old_code(trial):
    many = trial.base / "many"
    many.mkdir()
    for i in range(130):
        (many / f"f{i:03}.txt").write_text("y", encoding="utf-8")
    assert len(old.load_workspace(many)) == 120  # capped at 120 of 130


def test_h6_no_cap_in_new_code(trial):
    trial.seed("big.txt", "x" * 81_000)
    for i in range(130):
        trial.seed(f"many/f{i:03}.txt", "y")
    loaded = trial.new.load_workspace()
    assert "big.txt" in loaded and len(loaded["big.txt"]) == 81_000
    assert sum(1 for k in loaded if k.startswith("many/")) == 130


# --- H7: case-insensitive name rules --------------------------------------------------------


def test_h7_case_variants_accepted_by_old_code(trial):
    for name in ("CON", "con", "NUL.TXT", "nul.txt"):
        assert old.resolve(trial.root, name).name == name  # all pass, any case


@pytest.mark.parametrize("name", ["CON", "con", "Con", "NUL.TXT", "nul.txt", "Com1"])
def test_h7_case_variants_closed_in_new_code(trial, name):
    with pytest.raises(WorkspaceError):
        trial.new.resolve(name)


# --- §7 path-layer attack pack (scripted, must all be refused) ------------------------------

REFUSED = [
    r"..\x", "../x", r"C:\x", "C:/x", r"\\?\C:\x", r"\\server\share\x", r"~\x",
    "/etc/passwd", "%USERPROFILE%", ".env.", ".env ", "SECRET~1.TXT", "a.txt:hidden",
    "CON", "nul.txt", "a/" + "b" * 300, "foo\x00bar", ".localbot/x", ".git/config",
    "sub/../../x", "work/.LOCALBOT/y",
]


@pytest.mark.parametrize("bad", REFUSED)
def test_attack_pack_refused(trial, bad):
    with pytest.raises(WorkspaceError):
        trial.new.resolve(bad)


def test_write_into_localbot_refused(trial):
    with pytest.raises(WorkspaceError, match="hidden"):
        trial.new.write(".localbot/stash", "x")


@pytest.mark.parametrize("ok", ["a.txt", "sub/dir/b.py", "Secret.TXT", "notes.md", "a.b.c.txt"])
def test_valid_relative_names_allowed(trial, ok):
    """Ordinary relative names pass. (Catching 'Secret.TXT' as a secret is ProductGuard's
    job in S3, not the path layer's.)"""
    dest = trial.new.resolve(ok)
    assert str(dest).startswith(str(trial.root))


def test_read_cap_marks_truncation(trial):
    trial.seed("big.txt", "A" * 9000)
    out = trial.new.read("big.txt", cap_bytes=4000)
    assert out.startswith("A" * 4000) and "[truncated 5000 bytes]" in out


def test_write_is_atomic_and_leaves_no_temp(trial):
    trial.new.write("out.txt", "hello")
    assert (trial.root / "out.txt").read_text(encoding="utf-8") == "hello"
    assert not any(p.name.endswith(".localbot-tmp") for p in trial.root.iterdir())


# === S2.5 — review fixes ======================================================================
# The "hole" side of each S2.5 test runs against the S2 version of this module, read straight
# from git (commit 7aba138), so the repro is against the exact code the reviewer attacked.

import hashlib
import types

import localbot.guard.workspace_fs as wfs

S2_COMMIT = "7aba138"


def _load_s2_module() -> types.ModuleType:
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{S2_COMMIT}:localbot/guard/workspace_fs.py"],
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        pytest.skip(f"git show {S2_COMMIT} unavailable: {r.stderr.strip()}")
    mod = types.ModuleType("s2_workspace_fs")
    exec(compile(r.stdout, f"{S2_COMMIT}:workspace_fs.py", "exec"), mod.__dict__)
    return mod


def _tree_digest(folder: Path) -> dict[str, str]:
    """Hash of every entry under ``folder`` (the bait). Links are recorded, never followed."""
    out: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(folder, followlinks=False):
        for n in filenames:
            p = Path(dirpath) / n
            out[p.relative_to(folder).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
        for n in dirnames:
            out[(Path(dirpath) / n).relative_to(folder).as_posix() + "/"] = "dir"
    return out


def _plant(trial: Trial, kind: str, name: str) -> Path:
    """Put something at ``name`` in the root: a file symlink / dangling symlink to outside, a
    junction to outside, or a plain file."""
    if kind == "symlink":
        return trial.symlink(name, trial.outside / "secret.txt")
    if kind == "dangling":
        return trial.symlink(name, trial.outside / "planted_new.txt")
    if kind == "junction":
        return trial.junction(name, trial.outside)
    return trial.seed(name, "squatter")


# --- 1. temp-file escape -----------------------------------------------------------------------


def test_tmp_symlink_hole_in_s2_code(trial):
    """S2 write() wrote through the fixed name '<name>.localbot-tmp' without checking it."""
    s2 = _load_s2_module()
    trial.symlink("notes.txt.localbot-tmp", trial.outside / "secret.txt")
    s2.WorkspaceFS(trial.root).write("notes.txt", "PWNED\n")
    assert (trial.outside / "secret.txt").read_text(encoding="utf-8") == "PWNED\n"  # escaped
    assert (trial.root / "notes.txt").is_symlink()  # and the workspace file is now a link out
    trial._links.append(trial.root / "notes.txt")


def test_tmp_symlink_at_old_fixed_name_is_harmless_now(trial):
    trial.symlink("notes.txt.localbot-tmp", trial.outside / "secret.txt")
    before = _tree_digest(trial.outside)
    trial.new.write("notes.txt", "hello\n")
    assert _tree_digest(trial.outside) == before
    assert not (trial.root / "notes.txt").is_symlink()
    assert (trial.root / "notes.txt").read_bytes() == b"hello\n"


def test_tmp_name_is_random_and_in_target_folder(trial, monkeypatch):
    seen: list[Path] = []
    real_replace = os.replace

    def spy(src, dst):
        seen.append(Path(src))
        real_replace(src, dst)

    monkeypatch.setattr(wfs.os, "replace", spy)
    trial.new.write("sub/a.txt", "1")
    trial.new.write("sub/a.txt", "2")
    assert len(seen) == 2 and seen[0] != seen[1]
    assert all(p.parent == trial.root / "sub" and p.name.endswith(".localbot-tmp") for p in seen)


@pytest.mark.parametrize("kind", ["symlink", "dangling", pytest.param("junction", marks=JUNCTION), "file"])
def test_tmp_name_squatted_is_refused(trial, monkeypatch, kind):
    """Anything already at the temp name -> refused, outside unchanged, nothing written."""
    monkeypatch.setattr(wfs, "_tmp_token", lambda: "FIXED")
    planted = _plant(trial, kind, "notes.txt.FIXED.localbot-tmp")
    before = _tree_digest(trial.outside)
    with pytest.raises(WorkspaceError, match="already exists at the temp name"):
        trial.new.write("notes.txt", "PWNED\n")
    assert _tree_digest(trial.outside) == before
    assert not os.path.lexists(trial.root / "notes.txt")
    assert os.path.lexists(planted)  # not ours: left alone, not followed


@pytest.mark.parametrize("kind", ["symlink", "dangling", pytest.param("junction", marks=JUNCTION)])
def test_tmp_link_appearing_after_check_is_refused(trial, monkeypatch, kind):
    """Race path: the pre-check saw nothing, a link is there at create time. The create never
    follows it (on Windows O_EXCL alone *does* follow a dangling symlink and creates its target)."""
    monkeypatch.setattr(wfs, "_tmp_token", lambda: "FIXED")
    real_lexists = os.path.lexists
    monkeypatch.setattr(wfs.os.path, "lexists",
                        lambda p: False if str(p).endswith(".localbot-tmp") else real_lexists(p))
    _plant(trial, kind, "notes.txt.FIXED.localbot-tmp")
    before = _tree_digest(trial.outside)
    with pytest.raises(WorkspaceError, match="could not create a fresh temp file"):
        trial.new.write("notes.txt", "PWNED\n")
    assert _tree_digest(trial.outside) == before  # incl. no outside/planted_new.txt
    assert not real_lexists(trial.root / "notes.txt")


def test_dest_turned_reparse_point_after_replace_is_refused(trial, monkeypatch):
    """After os.replace the destination is re-checked; a link there is removed and refused."""
    real_replace = os.replace

    def replace_then_swap(src, dst):
        real_replace(src, dst)
        os.unlink(dst)
        os.symlink(trial.outside / "secret.txt", dst)
        trial._links.append(Path(dst))

    trial.symlink("_probe_privilege", trial.outside / "secret.txt")  # skip early if no symlinks
    monkeypatch.setattr(wfs.os, "replace", replace_then_swap)
    before = _tree_digest(trial.outside)
    with pytest.raises(WorkspaceError, match="became a reparse point"):
        trial.new.write("notes.txt", "x")
    assert not os.path.lexists(trial.root / "notes.txt")
    assert _tree_digest(trial.outside) == before


def test_write_bytes_unchanged_by_new_write_path(trial):
    """Same bytes as S2's write_text(encoding='utf-8', newline='\\n'): no CRLF added, UTF-8."""
    trial.new.write("u.txt", "a\nb\r\nc é\n")
    assert (trial.root / "u.txt").read_bytes() == "a\nb\r\nc é\n".encode("utf-8")
    trial.new.write("u.txt", "")
    assert (trial.root / "u.txt").read_bytes() == b""


# --- 2. reserved names (S2.5) ------------------------------------------------------------------

S25_RESERVED = [
    "COM¹", "COM²", "COM³", "LPT¹", "LPT²", "LPT³",
    "com¹.txt", "CONIN$", "CONOUT$", "CLOCK$", "conin$.log", "ConOut$", "clock$.txt",
    "CON .txt", "nul .txt", "con  .txt", "COM1 .log", "COM0", "LPT0",
]


@pytest.mark.parametrize("name", S25_RESERVED)
def test_s25_reserved_accepted_by_s2_code(trial, name):
    s2 = _load_s2_module()
    assert s2.WorkspaceFS(trial.root).resolve(name).name == name  # no raise = hole


@pytest.mark.parametrize("name", S25_RESERVED + ["sub/CONIN$", "sub/COM².c"])
def test_s25_reserved_closed_in_new_code(trial, name):
    with pytest.raises(WorkspaceError, match="reserved device"):
        trial.new.resolve(name)


@pytest.mark.parametrize("ok", ["CONIN", "COM10", "LPT", "CONX.txt", " CON", "console.log", "clock.txt"])
def test_near_miss_device_names_allowed(trial, ok):
    """Names that look like devices but are not (see the probe below) stay usable."""
    assert trial.new.resolve(ok).name == ok


PROBE_NAMES = [
    "CON", "con", "CON.txt", "CON .txt", "CON ", " CON", "CON..txt", "nul", "nul.txt", "nul .txt",
    "AUX", "PRN", "COM1", "COM9", "COM0", "LPT1", "LPT0", "COM¹", "COM²", "COM³",
    "LPT¹", "LPT²", "LPT³", "com¹.txt", "COM1 .log", "CONIN$", "CONOUT$",
    "CLOCK$", "conin$.txt", "CONOUT$ .txt", "CLOCK$.txt",
]


# Probe without writing: the names os.path.abspath (GetFullPathNameW) turns into \\.\<dev> on
# this machine. On Windows 11 26300 that is 17 of the 31 (table in BUILD_LOG S2.5).
DEVICE_NAMES = [n for n in PROBE_NAMES if os.name == "nt" and os.path.abspath(n).startswith("\\\\.\\")]


@pytest.mark.skipif(os.name != "nt", reason="GetFullPathNameW probe is Windows-only")
def test_probe_found_devices():
    assert {"CON", "nul", "COM\u00b9", "CONIN$", "CONOUT$"} <= set(DEVICE_NAMES)


@pytest.mark.parametrize("name", DEVICE_NAMES)
def test_every_name_windows_maps_to_a_device_is_refused(trial, name):
    """Every name the OS calls a device must be refused. Most names Windows 11 treats as files
    are refused too (the rule is stricter than the OS, see S25_RESERVED); ' CON' is a plain file
    and stays allowed."""
    with pytest.raises(WorkspaceError):
        trial.new.resolve(name)


# --- 3. H1 write through a junction / H3 long-name branch --------------------------------------


@JUNCTION
@pytest.mark.parametrize("rel", ["jlink/new.txt", "jlink/secret.txt", "jlink/deep/x.txt"])
def test_h1_write_through_junction_refused_outside_unchanged(trial, rel):
    trial.junction("jlink", trial.outside)
    before = _tree_digest(trial.outside)
    with pytest.raises(WorkspaceError, match="reparse point"):
        trial.new.write(rel, "PWNED\n")
    assert _tree_digest(trial.outside) == before
    assert (trial.outside / "secret.txt").read_text(encoding="utf-8") == "CANARY_pw_outside\n"


@pytest.mark.parametrize("long_leaf,reason", [
    ("nul.txt", "reserved device"), (".env.", "ending in"), ("CONIN$", "reserved device"),
    ("a.txt:hidden", "alternate data stream"), (".git", "hidden"),
])
def test_h3_long_name_branch_alone(trial, monkeypatch, long_leaf, reason):
    """Portable: _get_long_name is faked, so only the long-name recheck can refuse. The short
    name 'innocent.txt' passes every component rule on its own."""
    trial.seed("innocent.txt", "x")
    trial.new.resolve("innocent.txt")  # passes without the fake
    monkeypatch.setattr(wfs, "_get_long_name", lambda p: str(Path(p).parent / long_leaf))
    with pytest.raises(WorkspaceError, match=reason):
        trial.new.resolve("innocent.txt")


def test_h3_long_name_branch_quiet_when_long_equals_short(trial, monkeypatch):
    trial.seed("innocent.txt", "x")
    monkeypatch.setattr(wfs, "_get_long_name", lambda p: str(p))
    assert trial.new.resolve("innocent.txt").name == "innocent.txt"


# === S3.6 — name rules case-fold on every OS =================================================
# On Linux/macOS os.path.normcase is the identity, so rules written in lower case missed
# 'COM¹', 'CONIN$', '.GIT' etc. These tests force normcase to the identity (Linux behaviour)
# on Windows: the rules must still refuse every case variant.

CASE_VARIANTS = [
    "COM¹", "Com²", "LPT³", "CONIN$", "ConOut$", "CLOCK$", "CON .txt", "NUL .txt",
    "sub/NUL.tar.gz", "AUX", "Prn.log", "COM1", "LPT9.x", "CON", ".GIT/x", ".Git/config",
    ".LOCALBOT/y", ".LocalBot/brain.txt", "SECRET~1.TXT", "A.TXT:HIDDEN", ".ENV.", "FOO~12",
]


@pytest.fixture
def linux_normcase(monkeypatch):
    monkeypatch.setattr(wfs.os.path, "normcase", lambda s: s)
    assert os.path.normcase("ABC") == "ABC"


def test_s36_case_hole_reproduced_on_s3_code(trial, linux_normcase):
    """The S3 code (commit 99e8c15) with normcase = identity lets upper-case variants through."""
    r = subprocess.run(["git", "-C", str(REPO), "show", "99e8c15:localbot/guard/workspace_fs.py"],
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        pytest.skip("git show 99e8c15 unavailable")
    s3 = types.ModuleType("s3_workspace_fs")
    exec(compile(r.stdout, "99e8c15:workspace_fs.py", "exec"), s3.__dict__)
    fs = s3.WorkspaceFS(trial.root)
    allowed = [n for n in ["COM¹", "CONIN$", "CLOCK$", "CON .txt", "sub/NUL.tar.gz", ".GIT/x"]
               if fs.resolve(n) is not None]
    assert allowed == ["COM¹", "CONIN$", "CLOCK$", "CON .txt", "sub/NUL.tar.gz", ".GIT/x"]


@pytest.mark.parametrize("name", CASE_VARIANTS)
def test_s36_rules_casefold_with_linux_normcase(trial, linux_normcase, name):
    with pytest.raises(WorkspaceError):
        trial.new.resolve(name)


def test_s36_walk_hides_hidden_dirs_any_case_with_linux_normcase(trial, linux_normcase):
    trial.seed(".GIT/config", "x")
    trial.seed("sub/.LocalBot/brain.txt", "x")
    trial.seed("ok.txt", "x")
    assert trial.new.walk() == ["ok.txt"]


def test_s36_fold_name_is_casefold():
    assert wfs.fold_name("CONIN$") == "conin$" and wfs.fold_name("COM¹") == "com¹"
    assert wfs.fold_name(".GIT") == ".git" and wfs.fold_name("STRASSE") == wfs.fold_name("straße")


# === S6 — control / reserved characters (nemotron wrote ' \nq.txt' in the parity run) =====

BAD_CHAR_NAMES = [" \nq.txt", " \n2.txt", "a\tb.txt", "a\rb", "sub/\x01y", "x\x7f", "a<b", "a>b",
                  'a"b', "a|b", "a?b", "a*b", "dir\n/x.txt"]


@pytest.mark.parametrize("name", BAD_CHAR_NAMES)
def test_s6_control_and_reserved_chars_refused(trial, name):
    with pytest.raises(WorkspaceError, match="control or reserved character"):
        trial.new.resolve(name)


def test_s6_bad_char_write_refused_before_touching_disk(trial):
    with pytest.raises(WorkspaceError, match="control or reserved character"):
        trial.new.write(" \nq.txt", "Q")
    assert list(trial.root.iterdir()) == []


@pytest.mark.parametrize("ok", ["q.txt", "a b.txt", "x-y_z.txt", "naïve.txt", "a+b=c.txt", "[x].txt"])
def test_s6_ordinary_names_still_allowed(trial, ok):
    assert trial.new.resolve(ok).name == ok
