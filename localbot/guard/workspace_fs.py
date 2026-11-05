r"""Hardened one-folder workspace on disk (Windows rules first).

Ported from lab_ref/lib/workspace_fs.py and hardened per docs/STAGE1_SPEC.md §5.4. The job is
the same as the lab's: map a model-supplied relative name onto a real file *inside the
workspace root* and never anywhere else. ProductGuard still decides what a call may do; this
layer only decides whether a path is allowed to exist at all.

Every hole the spec predicted (H1-H7) is closed here and reproduced-then-closed in
tests/test_workspace_fs.py against the frozen lab code. Rules enforced:

- relative paths only; reject absolute paths, drive letters, UNC, ``\\?\``, ``~``, ``..``,
  NUL bytes, ``%ENV%`` references, and paths over 240 characters;
- reject control characters (newline, tab, DEL, ...) and ``<>"|?*`` in any component (S6),
  and the U+F000-U+F0FF private-use characters WSL stores them as on NTFS (S7a);
- reject any path component ending in ``.`` or space (H2), containing ``:`` (ADS, H4),
  matching a short-name ``~<digit>`` pattern (H3), or naming a reserved device
  (CON/NUL/AUX/PRN/COM0-9/LPT0-9/COM¹²³/LPT¹²³/CONIN$/CONOUT$/CLOCK$, any extension,
  trailing spaces on the stem ignored, so ``CON .txt`` counts) (H5, S2.5); all name rules
  run on the ``str.casefold()`` form on every OS, so they are case-insensitive (H7) even
  where ``os.path.normcase`` is the identity (Linux, macOS; S3.6);
- ``.localbot`` and ``.git`` are invisible to the model (reject on read and write);
- never follow a reparse point (symlink/junction): every existing component is checked with
  ``follow_symlinks=False`` and refused if it is a reparse point, on read and write (H1);
- for an existing target, its Windows long name (``GetLongPathNameW``) is run through the
  same name rules, so an 8.3 alias cannot smuggle a forbidden name past them (H3);
- reads are capped and say ``[truncated N bytes]`` rather than silently cutting; writes go
  through a fresh, randomly named temp file in the target's folder (refused if anything is
  already at that name, created ``O_CREAT|O_EXCL``, checked to be the file the descriptor
  points at) + ``os.replace``, and the result is refused if it ends up a reparse point (S2.5);
- ``walk`` enumerates the live tree with no file/byte cap (the guard's live view, H6 — the
  dict-capped snapshot that blinded the lab guard is gone; the ProductGuard side lands in S3).
"""
from __future__ import annotations

import ctypes
import os
import re
import secrets
import shutil
import stat
import sys
from pathlib import Path

FILE_ATTRIBUTE_REPARSE_POINT = 0x400

MAX_PATH_LEN = 240
READ_CAP_BYTES = 4000

# Components the model may never see or touch.
HIDDEN_DIRS = {".localbot", ".git"}

# Reserved DOS device names (any extension), matched on the stem, case-folded. The digit set
# includes 0 and the superscripts ¹²³ (U+00B9/B2/B3): Windows treats COM¹ as a device.
_DEVICE_DIGITS = "0123456789¹²³"
_RESERVED = (
    {"con", "prn", "aux", "nul", "conin$", "conout$", "clock$"}
    | {f"com{d}" for d in _DEVICE_DIGITS}
    | {f"lpt{d}" for d in _DEVICE_DIGITS}
)

# Temp files for write(): created exclusively, binary, never inheritable. O_NOFOLLOW where it
# exists (not on Windows, so write() also checks the name before and after the open).
_TMP_FLAGS = (
    os.O_CREAT | os.O_EXCL | os.O_WRONLY
    | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0) | getattr(os, "O_NOFOLLOW", 0)
)
_TMP_SUFFIX = ".localbot-tmp"

_SHORTNAME = re.compile(r"~\d")  # 8.3 alias marker, e.g. SECRE~1
# Control characters (incl. newline, tab, DEL) and the characters Windows forbids in names.
_BAD_CHARS = re.compile(r'[\x00-\x1f\x7f<>"|?*]')
# WSL (the podman/docker bind mount on Windows) stores a Linux name's control characters and
# "*:<>?\| as private-use U+F000 + code: bash's 'a:b.txt' is 'ab.txt' on NTFS, a newline
# is U+F00A (S7a probe). Same characters in disguise, so refused like them.
_WSL_MAPPED = re.compile("[-]")
_ENVVAR = re.compile(r"%[^%]+%")  # unexpanded env reference, e.g. %USERPROFILE%
_DRIVE = re.compile(r"^[A-Za-z]:(?:/|$)")  # drive letter: C:, C:/x (not x:stream, an ADS)


class WorkspaceError(ValueError):
    """A path was refused. ``str(exc)`` is a short reason, safe to surface."""


def fold_name(name: str) -> str:
    """Case-fold a name for every name rule, on every OS. Not ``os.path.normcase``: that is the
    identity on Linux/macOS, so 'COM¹' or '.GIT' would slip past rules written in lower
    case (and macOS is case-insensitive, so '.GIT/x' would reach the real .git)."""
    return name.casefold()


def _get_long_name(path: Path) -> str | None:
    """Windows long name of an existing path (resolves 8.3 aliases). None off Windows/on error."""
    if os.name != "nt":
        return None
    buf = ctypes.create_unicode_buffer(1024)
    n = ctypes.windll.kernel32.GetLongPathNameW(str(path), buf, 1024)  # type: ignore[attr-defined]
    if n == 0 or n >= 1024:
        return None
    return buf.value


def get_short_name(path: Path) -> str | None:
    """Windows 8.3 short name of an existing path (``GetShortPathNameW``). None off Windows, on
    error, or when 8.3 names are off for the volume (then it returns the long name itself)."""
    if os.name != "nt":
        return None
    buf = ctypes.create_unicode_buffer(1024)
    n = ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, 1024)  # type: ignore[attr-defined]
    if n == 0 or n >= 1024:
        return None
    return buf.value


# --- H1-K K6 D1 (owner decision): a case-sensitive work folder per bot / trial ----------------
# A container mask sits on one spelling of a path; on a case-insensitive folder another spelling
# (CONFIG/.env.local, .eNv.local) reaches the value. Windows can make one folder case-sensitive
# (what `fsutil file setCaseSensitiveInfo <dir> enable` does), only while it is empty; folders
# made inside it inherit the flag. Linux is case-sensitive unless the file system folds case
# (checked: H1-K R9). No admin rights are needed on this Windows build (10.0.26300); Microsoft
# documents that the WSL feature must be installed.

class CaseSensitivityError(WorkspaceError):
    """A work folder is not, or could not be made, case-sensitive."""


_FILE_CASE_SENSITIVE_INFORMATION = 71  # FILE_INFORMATION_CLASS
_CASE_SENSITIVE_DIR = 0x1  # FILE_CS_FLAG_CASE_SENSITIVE_DIR
_NTSTATUS_TEXT = {0xC0000101: "the folder is not empty (the flag only goes on an empty folder)",
                  0xC00000BB: "not supported here (it needs an NTFS folder and Windows' WSL feature)",
                  0xC0000022: "access denied"}


def _case_info(folder: Path, turn_on: bool) -> int:
    """Windows: a folder's case-sensitivity flags (NtQueryInformationFile), or turn the flag on
    (NtSetInformationFile). The API, not fsutil: fsutil prints a localized error and exits 0."""
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # private handles: no global argtypes
    nt = ctypes.WinDLL("ntdll")
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]

    class _IoStatus(ctypes.Structure):
        _fields_ = [("status", ctypes.c_void_p), ("information", ctypes.c_size_t)]

    read_attributes, write_attributes = 0x80, 0x100
    share_all, open_existing, backup_semantics = 0x7, 3, 0x02000000  # a folder opens only with this
    h = k32.CreateFileW(str(folder), read_attributes | (write_attributes if turn_on else 0), share_all, None,
                        open_existing, backup_semantics, None)
    if h is None or h == ctypes.c_void_p(-1).value:
        raise CaseSensitivityError(f"{folder}: cannot open the folder ({ctypes.WinError(ctypes.get_last_error()).strerror})")
    try:
        fn = nt.NtSetInformationFile if turn_on else nt.NtQueryInformationFile
        fn.restype = ctypes.c_long
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(_IoStatus), ctypes.c_void_p, wintypes.ULONG, ctypes.c_int]
        iosb, flags = _IoStatus(), wintypes.ULONG(_CASE_SENSITIVE_DIR if turn_on else 0)
        status = fn(h, ctypes.byref(iosb), ctypes.byref(flags), ctypes.sizeof(flags),
                    _FILE_CASE_SENSITIVE_INFORMATION) & 0xFFFFFFFF
    finally:
        k32.CloseHandle(wintypes.HANDLE(h))
    if status != 0:
        raise CaseSensitivityError(f"{folder}: {_NTSTATUS_TEXT.get(status, f'NTSTATUS 0x{status:08X}')}")
    return flags.value


def _folds_case(folder: Path) -> bool | None:
    """Read-only (H1-K R9): does ``folder``'s file system fold case? Looks the folder's own name up
    with its case swapped (or the nearest ancestor's, if the name has no letter): the same directory
    found = names fold. A new folder inherits its parent's behaviour (ext4 casefold, WSL drvfs,
    vfat / ntfs3 / exfat mounts), so the lookup in the parent speaks for it. None = can't tell."""
    p = Path(os.path.abspath(folder))
    for q in (p, *p.parents):
        swapped = q.name.swapcase()
        if swapped == q.name:
            continue
        try:
            return os.path.samefile(q, q.parent / swapped)
        except FileNotFoundError:
            return False  # no such name: lookups keep case
        except OSError:
            return None
    return None


def is_case_sensitive(folder: Path) -> bool:
    """True when names in ``folder`` that differ only in case are different files. Linux: the
    read-only probe (``_folds_case``; can't tell = False). Windows: the folder's flag. Any other OS
    (macOS is case-insensitive by default): False."""
    if sys.platform.startswith("linux"):
        return _folds_case(Path(folder)) is False
    if sys.platform != "win32":
        return False
    try:
        return bool(_case_info(Path(folder), turn_on=False) & _CASE_SENSITIVE_DIR)
    except CaseSensitivityError:
        return False


def set_case_sensitive(folder: Path) -> None:
    """Make ``folder`` case-sensitive (it must be empty on Windows); CaseSensitivityError if it
    can't be done or doesn't stick. Linux: nothing to turn on; a folder that folds case (or can't
    be shown not to) is refused. Any other OS: refused."""
    if sys.platform.startswith("linux"):
        if not is_case_sensitive(folder):
            raise CaseSensitivityError(f"{folder}: names here fold case, or that can't be ruled out (a "
                                       f"case-insensitive file system; Linux can't change it per folder)")
        return
    if sys.platform != "win32":
        raise CaseSensitivityError(f"{folder}: a case-sensitive work folder needs Linux or Windows")
    _case_info(Path(folder), turn_on=True)
    if not is_case_sensitive(folder):
        raise CaseSensitivityError(f"{folder}: the case-sensitive flag did not stick")


def new_work_folder(path: Path) -> Path:
    """Create ``path`` (and its parents) and make it case-sensitive while it is still empty, before
    its first file. FileExistsError if it is there already: an existing folder may hold files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()
    set_case_sensitive(path)
    return path


def _tmp_token() -> str:
    """Random part of a write() temp name. A function so tests can force a known name."""
    return secrets.token_hex(8)


def _create_new_file(path: Path) -> int:
    """Create ``path`` as a brand-new file and return a write descriptor. Never follows a link.

    On Windows ``os.open(O_CREAT|O_EXCL)`` follows a *dangling* symlink and creates its
    target, and there is no O_NOFOLLOW, so the file is made with ``CreateFileW(CREATE_NEW,
    FILE_FLAG_OPEN_REPARSE_POINT)``: any existing entry at the name, link or not, fails it.
    """
    if os.name != "nt":
        return os.open(path, _TMP_FLAGS, 0o600)
    import msvcrt
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # private handle: no global argtypes
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    generic_write, create_new = 0x40000000, 1
    file_attribute_normal, file_flag_open_reparse_point = 0x80, 0x00200000
    h = k32.CreateFileW(str(path), generic_write, 0, None, create_new,
                        file_attribute_normal | file_flag_open_reparse_point, None)
    if h is None or h == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return msvcrt.open_osfhandle(h, os.O_WRONLY | os.O_BINARY)
    except OSError:
        k32.CloseHandle(wintypes.HANDLE(h))
        raise


def extended_path(path: str | os.PathLike[str]) -> Path:
    """``\\\\?\\`` + the absolute path on Windows: no device-name (``nul``) or trailing-dot/space
    parsing, so a container-made ``nul`` is a file there, not the NUL device. Unchanged off
    Windows or if already extended."""
    s = str(path)
    if os.name != "nt" or s.startswith("\\\\?\\"):
        return Path(path)
    return Path("\\\\?\\" + os.path.abspath(s))


def _remove_entry(path: Path) -> None:
    """Delete one directory entry without following it (a link is removed, not its target)."""
    try:
        if path.is_dir() and not _is_reparse_point(path):
            return  # a real directory is never ours to remove
        try:
            os.unlink(path)
        except (IsADirectoryError, PermissionError):
            os.rmdir(path)  # a junction on Windows: rmdir removes the link only
    except OSError:
        pass


def _make_writable(path: str | os.PathLike[str]) -> bool:
    """Clear the read-only flag, but only on an entry ``lstat`` shows is a plain regular file or
    directory: ``os.chmod`` follows links, so a link or reparse point is never chmod-ed.
    Returns True if it chmod-ed."""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    if getattr(st, "st_file_attributes", 0) & FILE_ATTRIBUTE_REPARSE_POINT or stat.S_ISLNK(st.st_mode):
        return False
    if not (stat.S_ISREG(st.st_mode) or stat.S_ISDIR(st.st_mode)):
        return False
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    return True


def _writable_then_retry(func, path, exc) -> None:
    """``shutil.rmtree`` onexc: a read-only file (chmod 000 from a container, or git's object
    files) can't be unlinked on Windows; clear the flag (plain entries only) and try once more."""
    if isinstance(exc, PermissionError) and func in (os.unlink, os.remove, os.rmdir) and _make_writable(path):
        func(path)
        return
    raise exc


def remove_tree(path: str | os.PathLike[str]) -> None:
    """Delete ``path`` (a folder tree or one entry) without ever following a link or junction:
    links are removed, never their targets. Works on the ``\\\\?\\`` form (``nul``, ``CON``,
    trailing dots are plain names there) and clears read-only flags."""
    path = extended_path(path)
    if not os.path.lexists(path):
        return
    if _is_reparse_point(path) or not stat.S_ISDIR(os.lstat(path).st_mode):
        _unlink_entry(path)
        return
    for dirpath, dirnames, filenames in os.walk(path, topdown=True, followlinks=False):
        for d in list(dirnames):
            full = Path(dirpath) / d
            if full.is_symlink() or _is_reparse_point(full):
                _remove_entry(full)
                dirnames.remove(d)
        for f in filenames:
            full = Path(dirpath) / f
            if full.is_symlink() or _is_reparse_point(full):
                _remove_entry(full)
    shutil.rmtree(path, onexc=_writable_then_retry)


def _unlink_entry(path: Path) -> None:
    """Remove one non-directory entry (file, symlink, junction) without following it."""
    if _is_reparse_point(path):
        _remove_entry(path)
        if not os.path.lexists(path):
            return
    try:
        os.unlink(path)
    except PermissionError:
        if not _make_writable(path):
            raise
        os.unlink(path)


def _hard_linked(path: Path) -> bool:
    """True if ``path`` is an existing regular file with more than one hard link. Directories
    normally have st_nlink > 1 (subdirs), so this applies to regular files only."""
    try:
        st = path.lstat()
    except (OSError, ValueError):
        return False
    return stat.S_ISREG(st.st_mode) and getattr(st, "st_nlink", 1) > 1


def _is_reparse_point(path: Path) -> bool:
    """True if ``path`` exists and is a reparse point (symlink or junction). Never follows it."""
    try:
        st = path.lstat()
    except (OSError, ValueError):
        return False
    attrs = getattr(st, "st_file_attributes", 0)
    if attrs & FILE_ATTRIBUTE_REPARSE_POINT:
        return True
    # Non-Windows (and belt-and-braces on Windows): a symlink is a reparse point too.
    return path.is_symlink()


def _check_name(name: str, *, original: str) -> None:
    """Apply every component-name rule to one name (already case-folded by the caller)."""
    if not name:
        raise WorkspaceError(f"empty path component in {original!r}")
    if any(0xD800 <= ord(ch) <= 0xDFFF for ch in name):  # H1 P2: a non-UTF-8 name from Linux
        raise WorkspaceError(f"refused name that is not valid Unicode (lone surrogate): {original!r}")
    if _BAD_CHARS.search(name):  # S6: nemotron wrote ' \nq.txt'; Windows refuses, Linux would not
        raise WorkspaceError(f"refused control or reserved character in path: {original!r}")
    if _WSL_MAPPED.search(name):
        raise WorkspaceError(f"refused WSL-mapped control or reserved character (U+F0xx) in path: {original!r}")
    if ":" in name:
        raise WorkspaceError(f"refused ':' in path component (alternate data stream): {original!r}")
    if name.endswith(".") or name.endswith(" "):
        raise WorkspaceError(f"refused path component ending in '.' or space: {original!r}")
    if _SHORTNAME.search(name):
        raise WorkspaceError(f"refused 8.3 short-name component (~N): {original!r}")
    if _ENVVAR.search(name):
        raise WorkspaceError(f"refused environment-variable reference in path: {original!r}")
    # Windows ignores trailing spaces on the stem: "CON .txt" names the same thing as "CON.txt".
    stem = name.split(".", 1)[0].rstrip(" ")
    if stem in _RESERVED:
        raise WorkspaceError(f"refused reserved device name ({stem}): {original!r}")
    if name in HIDDEN_DIRS:
        raise WorkspaceError(f"refused hidden path component ({name}): {original!r}")


class WorkspaceFS:
    """All disk access for one workspace root. Construct once per trial."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve()
        self._case_sensitive: bool | None = None

    @property
    def case_sensitive(self) -> bool:
        """True when ``A.txt`` and ``a.txt`` are two files here (a core4 work folder, Linux), so
        names must never be folded together (H1-K D1). Asked once per workspace."""
        if self._case_sensitive is None:
            self._case_sensitive = is_case_sensitive(self.root)
        return self._case_sensitive

    # -- path validation ---------------------------------------------------------------

    def resolve(self, rel: str) -> Path:
        """Validate a model-supplied relative name and return the real path inside the root.

        Raises WorkspaceError (never silently remaps) if the name breaks any rule, points
        outside the root, or crosses a reparse point. Does not create anything.
        """
        raw = rel or ""
        if "\x00" in raw:
            raise WorkspaceError("refused NUL byte in path")
        # No whole-path strip: a trailing space is exactly the ``.env ``/H2 attack, and
        # stripping it here is what let the lab code open ``.env``.
        blob = raw.replace("\\", "/")
        if not blob.strip():
            raise WorkspaceError("empty path")
        if len(blob) > MAX_PATH_LEN:
            raise WorkspaceError(f"refused path over {MAX_PATH_LEN} characters")
        if blob.startswith("//"):
            raise WorkspaceError("refused UNC or \\\\?\\ path. Workspace only.")
        if blob.startswith("/"):
            raise WorkspaceError("refused absolute path. Workspace only.")
        if blob.startswith("~"):
            raise WorkspaceError("refused home (~) path. Workspace only.")
        if _DRIVE.match(blob):
            raise WorkspaceError("refused drive-letter path. Workspace only.")

        parts = blob.split("/")
        for part in parts:
            if part in ("", "."):
                raise WorkspaceError(f"refused empty or '.' path component: {rel!r}")
            if part == "..":
                raise WorkspaceError("refused '..' path component. Workspace only.")
            _check_name(fold_name(part), original=rel)

        # Walk component by component from the root, refusing any reparse point we cross and
        # re-checking the real (long) name of each existing component. resolve() is never
        # used to build the path (it would follow links); containment is structural.
        cur = self.root
        for part in parts:
            cur = cur / part
            if _is_reparse_point(cur):
                raise WorkspaceError(f"refused reparse point (symlink/junction) in path: {rel!r}")
            if cur.exists():
                long_name = _get_long_name(cur)
                # GetLongPathNameW returns the full path; compare and re-check the leaf only.
                long_leaf = os.path.basename(long_name) if long_name else ""
                if long_leaf and fold_name(long_leaf) != fold_name(cur.name):
                    _check_name(fold_name(long_leaf), original=rel)
                # A hard link is a second name for one inode; a container could hard-link a
                # work/ name to a file elsewhere on the volume (S7). Refuse an existing regular
                # file with more than one link, on read and write.
                if _hard_linked(cur):
                    raise WorkspaceError(f"refused hard-linked file (st_nlink > 1): {rel!r}")

        # Defense in depth: the lexical real path must stay under the root.
        real = os.path.normcase(os.path.normpath(str(cur)))
        root_n = os.path.normcase(os.path.normpath(str(self.root)))
        if real != root_n and not real.startswith(root_n + os.sep):
            raise WorkspaceError(f"refused path outside the workspace: {rel!r}")
        return cur

    # -- read / write ------------------------------------------------------------------

    def read(self, rel: str, cap_bytes: int | None = READ_CAP_BYTES) -> str:
        """Read a file. Caps output and marks the cut; never silently truncates."""
        dest = self.resolve(rel)
        data = dest.read_bytes()
        if cap_bytes is not None and len(data) > cap_bytes:
            shown = data[:cap_bytes].decode("utf-8", errors="replace")
            return f"{shown}\n[truncated {len(data) - cap_bytes} bytes]"
        return data.decode("utf-8", errors="replace")

    def write(self, rel: str, content: str) -> Path:
        """Write a file atomically (temp + os.replace), inside the root.

        The temp file gets a random name in the target's folder and must be brand new: if
        anything (file, symlink, junction, dangling link) is already at that name the write
        is refused, never written through. The name is checked before the create, the create
        itself never follows a link (``_create_new_file``), and the descriptor is checked to be
        the entry at that name before any byte is written.
        """
        dest = self.resolve(rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8")  # same bytes as write_text(encoding="utf-8", newline="\n")
        tmp = dest.parent / f"{dest.name}.{_tmp_token()}{_TMP_SUFFIX}"
        if os.path.lexists(tmp):
            raise WorkspaceError(f"refused write: something already exists at the temp name for {rel!r}")
        try:
            fd = _create_new_file(tmp)
        except OSError as exc:
            raise WorkspaceError(f"refused write: could not create a fresh temp file for {rel!r}") from exc
        ok = False
        try:
            try:
                same = os.path.samestat(os.fstat(fd), os.lstat(tmp))
            except OSError:
                same = False
            if not same or _is_reparse_point(tmp):
                raise WorkspaceError(f"refused write: temp file for {rel!r} is not a plain new file")
            with os.fdopen(fd, "wb") as fh:
                fd = -1
                fh.write(data)
            os.replace(tmp, dest)
            ok = True
        finally:
            if fd != -1:
                os.close(fd)
            if not ok:
                _remove_entry(tmp)
        if _is_reparse_point(dest):
            _remove_entry(dest)
            raise WorkspaceError(f"refused write: {rel!r} became a reparse point (symlink/junction)")
        return dest

    def exists(self, rel: str) -> bool:
        try:
            return self.resolve(rel).is_file()
        except WorkspaceError:
            return False

    # -- enumeration (the guard's live view, H6) ---------------------------------------

    def walk(self) -> list[str]:
        """Every readable file under the root as a posix relative path. No file/byte cap.

        Never descends into a reparse point or a hidden dir, so it cannot leave the root or
        expose ``.localbot`` / ``.git``. Skips any entry whose own name the name rules refuse.
        """
        out: list[str] = []
        self._walk_into(self.root, "", out)
        return sorted(out)

    def _walk_into(self, folder: Path, prefix: str, out: list[str]) -> None:
        try:
            entries = list(os.scandir(folder))
        except OSError:
            return
        for entry in entries:
            name = entry.name
            if fold_name(name) in HIDDEN_DIRS:
                continue
            try:
                # A name the path layer refuses (bash-made `nul`, `CON`, U+F0xx, ...) is not
                # listed: it isn't readable through this layer (S7a.1). String rules only, no I/O.
                _check_name(fold_name(name), original=name)
            except WorkspaceError:
                continue
            try:
                if entry.is_symlink():
                    continue
                attrs = getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0)
                if attrs & FILE_ATTRIBUTE_REPARSE_POINT:
                    continue
                rel = f"{prefix}{name}"
                if entry.is_dir(follow_symlinks=False):
                    self._walk_into(Path(entry.path), rel + "/", out)
                elif entry.is_file(follow_symlinks=False):
                    out.append(rel)
            except OSError:
                continue

    def load_workspace(self) -> dict[str, str]:
        """Read the whole tree into a {relpath: text} dict (uncapped). The disk-backed guard
        view in S3 replaces this; kept so the exam runner can snapshot files_end."""
        out: dict[str, str] = {}
        for rel in self.walk():
            try:
                out[rel] = self.read(rel, cap_bytes=None)
            except (OSError, WorkspaceError):
                continue
        return out


def scan_tree(root: str | os.PathLike[str]) -> dict[str, list[str]]:
    """Walk ``root`` without following any link and report what a container exec may have left
    that a host-side tool must not touch (S7). Buckets (relative posix paths):

    - ``reparse``: symlinks and junctions (``resolve`` already refuses to cross these);
    - ``hard_links``: regular files with st_nlink > 1 (``resolve`` refuses these too);
    - ``refused_names``: entries whose own name ``resolve`` refuses — device / 8.3 / stream /
      control-character names and the hidden ``.localbot`` / ``.git`` (created by bash, say).

    A reparse-point directory is recorded and not descended into.
    """
    root = Path(root).resolve()
    out: dict[str, list[str]] = {"reparse": [], "hard_links": [], "refused_names": []}
    fs = WorkspaceFS(root)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        kept = []
        for d in dirnames:
            full = Path(dirpath) / d
            rel = full.relative_to(root).as_posix()
            if _is_reparse_point(full):
                out["reparse"].append(rel)
            else:
                kept.append(d)
                if _refused_name(fs, root, full):
                    out["refused_names"].append(rel)
        dirnames[:] = kept
        for n in filenames:
            full = Path(dirpath) / n
            rel = full.relative_to(root).as_posix()
            if _is_reparse_point(full):
                out["reparse"].append(rel)
                continue
            if _hard_linked(full):
                out["hard_links"].append(rel)
            if _refused_name(fs, root, full):
                out["refused_names"].append(rel)
    for v in out.values():
        v.sort()
    return out


def is_git_name(rel: str) -> bool:
    """The last component of ``rel`` is ``.git`` in any case (S8e)."""
    return fold_name(rel.rsplit("/", 1)[-1]) == ".git"


def git_entries(root: str | os.PathLike[str]) -> list[tuple[str, bool]]:
    """Every ``.git`` (any case, any depth) under ``root`` as ``(rel, is_dir)`` — a repository
    folder or a gitfile (worktree / submodule). Walked on the ``\\\\?\\`` form like the undo
    scan: no link is followed or listed, and no refused-name folder (``.git`` itself included)
    is descended into (S8e: the container masks exactly these)."""
    out: list[tuple[str, bool]] = []

    def walk(folder: Path, prefix: str) -> None:
        with os.scandir(folder) as it:
            entries = list(it)
        for e in entries:
            rel = prefix + e.name
            st = os.lstat(e.path)
            if (getattr(st, "st_file_attributes", 0) & FILE_ATTRIBUTE_REPARSE_POINT) or stat.S_ISLNK(st.st_mode):
                continue
            if is_git_name(rel):
                if stat.S_ISDIR(st.st_mode) or stat.S_ISREG(st.st_mode):
                    out.append((rel, stat.S_ISDIR(st.st_mode)))
                continue
            if stat.S_ISDIR(st.st_mode):
                try:
                    _check_name(fold_name(e.name), original=rel)
                except WorkspaceError:
                    continue
                walk(Path(e.path), rel + "/")

    walk(extended_path(root), "")
    return sorted(out)


def _case_spellings(name: str) -> set[str]:
    """Every upper/lower-case spelling of ``name``'s letters (``.git`` → 8, ``GIT~1`` → 8)."""
    letters = [i for i, ch in enumerate(name) if ch.isalpha()]
    out = set()
    for bits in range(1 << len(letters)):
        chars = list(name)
        for j, i in enumerate(letters):
            chars[i] = chars[i].upper() if (bits >> j) & 1 else chars[i].lower()
        out.add("".join(chars))
    return out


def git_mask_spellings(work: str | os.PathLike[str], rel: str) -> list[tuple[str, bool]]:
    """S8e.1: container paths that reach the same host ``.git`` as ``rel`` through a different
    spelling of its last component — the 8 case variants and the real 8.3 short name (in its
    case variants). Each is kept only when it exists and resolves to the *same* host entry, so a
    mask over it can never create a new folder or file in work/. ``rel`` itself is excluded.
    Returns ``(dest_rel, is_dir)`` (the type of the real entry). Only the entry's own name is
    varied; a nested ``.git`` reached through a parent folder's spelling is not maskable (the
    check+restore layer covers that)."""
    root = extended_path(work)
    real = root.joinpath(*rel.split("/"))
    try:
        if not os.path.lexists(real):
            return []
    except OSError:
        return []
    parent = rel.rsplit("/", 1)[0] + "/" if "/" in rel else ""
    leaf = rel.rsplit("/", 1)[-1]
    is_dir = os.path.isdir(real) and not _is_reparse_point(real)
    names = _case_spellings(leaf)
    short = get_short_name(real)
    if short:
        short_leaf = os.path.basename(short.rstrip("\\/"))
        if short_leaf and short_leaf.casefold() != leaf.casefold():
            names |= _case_spellings(short_leaf)
    out: list[tuple[str, bool]] = []
    for n in sorted(names):
        if n == leaf:
            continue
        cand = root.joinpath(*(parent + n).split("/"))
        try:
            if os.path.lexists(cand) and os.path.samefile(cand, real):
                out.append((parent + n, is_dir))
        except OSError:
            continue
    return out


def _refused_name(fs: "WorkspaceFS", root: Path, full: Path) -> bool:
    """True if ``resolve`` would refuse the relative path of ``full`` on a name rule (not for
    being a reparse point or hard link, which the scan reports separately)."""
    rel = full.relative_to(root).as_posix()
    try:
        fs.resolve(rel)
        return False
    except WorkspaceError as exc:
        s = str(exc)
        return not ("reparse point" in s or "hard-linked" in s)


RAW_SCAN_CAP = 1 << 20  # bytes; a refused file larger than this is not scanned -> "unscanned"


def read_refused_raw(work: Path, rels: list[str], cap: int = RAW_SCAN_CAP) -> dict[str, bytes | None]:
    """Raw bytes of the files the path layer refuses (refused names, hard links, ...; S7a.1),
    for secret scans only, so a secret written into ``nul`` or a newline name is still found.
    Read on the ``\\\\?\\`` form (``nul`` is a file there, not the device), never through a
    link: the entry is lstat'ed and must be a plain regular file. Directories are skipped
    (their files are listed on their own). ``None`` = could not be scanned (a link, too big,
    unreadable)."""
    out: dict[str, bytes | None] = {}
    base = extended_path(work)
    for rel in rels:
        p = base.joinpath(*rel.split("/"))
        try:
            st = os.lstat(p)
            if stat.S_ISDIR(st.st_mode):
                continue
            if _is_reparse_point(p) or not stat.S_ISREG(st.st_mode) or st.st_size > cap:
                out[rel] = None
                continue
            with open(p, "rb") as fh:
                out[rel] = fh.read(cap + 1)[:cap]
        except OSError:
            out[rel] = None
    return out
