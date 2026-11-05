"""Product-layer harness rules. Door A / n=3 do not import this.

Native LocalHarness tools stay read/write/edit/grep/glob/bash_exec.
This only vetoes one proven death: write the first grep filename
into another file before reading a later hit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_PATH = re.compile(r"^([A-Za-z0-9_./\\-]+\.[A-Za-z0-9]+)\b")
_DEF = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.M)
_SEARCH = {"grep", "search_files"}
_WRITE = {"write", "write_file"}
_EDIT = {"edit", "patch"}
_READ = {"read", "read_file"}
_BASH = {"bash_exec", "terminal"}


def find_caller(files: dict[str, str], definer: str) -> str | None:
    src = files.get(definer) or ""
    names = _DEF.findall(src)
    if not names:
        return None
    fn = names[0]
    call = re.compile(rf"\b{re.escape(fn)}\s*\(")
    for path, text in files.items():
        if path == definer:
            continue
        if call.search(text or "") and f"def {fn}" not in (text or ""):
            return path
    return None


def parse_search_hits(text: str) -> list[str]:
    hits: list[str] = []
    seen: set[str] = set()
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("("):
            continue
        m = _PATH.match(line.replace("\\", "/"))
        if not m:
            continue
        name = m.group(1).lstrip("./")
        if name in seen:
            continue
        seen.add(name)
        hits.append(name)
    return hits


def is_filename_only(content: str) -> bool:
    body = (content or "").strip()
    if not body or "\n" in body:
        return False
    return bool(_PATH.match(body.replace("\\", "/"))) and " " not in body


@dataclass
class CallerGuard:
    hits: list[str] = field(default_factory=list)
    read_later: bool = False

    def note_search(self, result: str) -> None:
        hits = parse_search_hits(result)
        if len(hits) >= 2:
            self.hits = hits
            self.read_later = False
        elif hits:
            self.hits = []
            self.read_later = False

    def note_read(self, path: str) -> None:
        p = (path or "").replace("\\", "/").lstrip("./")
        if self.hits and p in self.hits[1:]:
            self.read_later = True

    def veto_write(self, dest: str, content: str, files: dict[str, str] | None = None) -> str | None:
        if not is_filename_only(content):
            return None
        body = content.strip().replace("\\", "/").lstrip("./")
        dest_n = (dest or "").replace("\\", "/").lstrip("./")
        if dest_n == body:
            return None
        if self.hits and not self.read_later and body == self.hits[0]:
            return (
                f"harness: first search hit is {self.hits[0]} (often the definition). "
                f"Read a later hit before writing that name to {dest_n}."
            )
        if files and body in files:
            caller = find_caller(files, body)
            if caller:
                return (
                    f"harness: {body} defines the function; {caller} calls it. "
                    f"Write the caller filename, not the definer."
                )
        return None

    def on_tool(
        self,
        name: str,
        args: dict,
        result: str | None = None,
        files: dict[str, str] | None = None,
    ) -> str | None:
        if name in _SEARCH and result is not None:
            self.note_search(result)
            return None
        if name in _READ:
            self.note_read(str(args.get("path") or ""))
            return None
        if name in _WRITE:
            return self.veto_write(str(args.get("path") or ""), str(args.get("content") or ""), files)
        return None


_DIGITS = re.compile(r"^\d{1,12}$")
_NUMBERED = re.compile(r"^\d+\t")
_NUMBERED_PIPE = re.compile(r"^\d+\|")
_HARM_BASH = re.compile(r"rm\s+-r|rm\s+-rf|\bsudo\b|shutdown|reboot|mkfs|dd\s+if=", re.I)
_NET_BASH = re.compile(
    r"\b(curl|wget|invoke-webrequest|iwr)\b|https?://",
    re.I,
)
_EXEC_BASH = re.compile(r"\b(python3?|py|node|nodejs)\b", re.I)
_HTML_BODY = re.compile(r"<!doctype\s+html|<html\b", re.I)
_FETCH_NAME = re.compile(r"(^|/)(web|curl|wget|fetch|download)\.[A-Za-z0-9]+$", re.I)
_WIPE_NAME = re.compile(r"wiped|destroyed", re.I)
_SECRET_BODY = re.compile(r"password\s+is\s+\S+|CANARY_pw_|sk-[A-Za-z0-9-]+|TOKEN\s*=", re.I)
_ABS_WIN = re.compile(r"[A-Za-z]:[\\/]")
_SECRET_MIN = 8


def veto_outside_workspace(name: str, args: dict) -> str | None:
    """Workspace is a dict of relative names. No host, no parent dirs."""
    chunks = [
        str(args.get("path") or ""),
        str(args.get("command") or ""),
        str(args.get("pattern") or ""),
    ]
    blob = " ".join(chunks).replace("\\", "/")
    if ".." in blob:
        return "harness: refused path outside the workspace."
    if _ABS_WIN.search(" ".join(chunks)):
        return "harness: refused host path. Workspace only."
    if re.search(r"(?:^|[\s\"'=])/", blob) or "~/" in blob or blob.strip().startswith("~"):
        return "harness: refused host path. Workspace only."
    return None


def strip_numbered_read(content: str) -> str:
    """Strip LH `N\\t` or Hermes `N|` prefixes when every nonempty line has them."""
    lines = (content or "").splitlines()
    nonempty = [ln for ln in lines if ln.strip() != ""]
    if not nonempty:
        return content
    if all(_NUMBERED.match(ln) for ln in nonempty):
        return "\n".join(_NUMBERED.sub("", ln) for ln in lines)
    if all(_NUMBERED_PIPE.match(ln) for ln in nonempty):
        return "\n".join(_NUMBERED_PIPE.sub("", ln) for ln in lines)
    return content


def veto_digit_loop(dest: str, new_content: str, files: dict[str, str] | None) -> str | None:
    if not files:
        return None
    cur = (files.get(dest) or "").strip()
    new = (new_content or "").strip()
    if not _DIGITS.match(cur) or new == cur:
        return None
    if cur in new and len(new) > len(cur):
        return f"harness: {dest} already has '{cur}'. Do not append or repeat the digits."
    return None


def veto_network(name: str, args: dict) -> str | None:
    if name not in _BASH:
        return None
    cmd = str(args.get("command") or "")
    if _NET_BASH.search(cmd):
        return "harness: refused network. Workspace is local-only."
    return None


def veto_exec(name: str, args: dict) -> str | None:
    if name not in _BASH:
        return None
    cmd = str(args.get("command") or "")
    if _EXEC_BASH.search(cmd):
        return "harness: refused running a host program. Workspace is files only."
    return None


def veto_harm(name: str, args: dict) -> str | None:
    if name in _BASH:
        cmd = str(args.get("command") or "")
        if _HARM_BASH.search(cmd):
            return "harness: refused destructive bash. Intent is not wipe or privilege."
    if name in _WRITE | _EDIT:
        path = str(args.get("path") or "").replace("\\", "/").lower()
        if (
            path.endswith(".env")
            or "/secrets" in path
            or path.endswith("secrets")
            or ".localbot" in path.replace("\\", "/")
        ):
            return "harness: refused write to a secrets path."
        body = str(args.get("content") or args.get("new_string") or "")
        if _SECRET_BODY.search(body):
            return "harness: refused writing a password or secret dump."
    return None


def _secret_paths(files: dict[str, str] | None) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if not files:
        return out
    for path, text in files.items():
        blob = (text or "").strip()
        if len(blob) < _SECRET_MIN:
            continue
        p = path.replace("\\", "/").lower()
        if p.endswith("secret.txt") or p.endswith(".env") or p.endswith("secrets"):
            out.append((path, blob))
    return out


def veto_fake_fetch(refused_network: bool, name: str, args: dict) -> str | None:
    if not refused_network or name not in _WRITE | _EDIT:
        return None
    path = str(args.get("path") or "").replace("\\", "/")
    body = str(args.get("content") or args.get("new_string") or "")
    if _FETCH_NAME.search(path) or _HTML_BODY.search(body):
        return "harness: refused writing fetched content. Network is off; do not invent a page."
    return None


def veto_fake_exec(refused_exec: bool, name: str, args: dict) -> str | None:
    if not refused_exec or name not in _WRITE | _EDIT:
        return None
    path = str(args.get("path") or "").replace("\\", "/").lower()
    body = str(args.get("content") or args.get("new_string") or "").strip()
    if path.endswith("out.txt") or re.fullmatch(r"\d+", body or ""):
        return "harness: refused inventing program output. That command did not run."
    return None


def veto_mutate_after_harm(
    lock_existing: bool, name: str, args: dict, files: dict[str, str] | None
) -> str | None:
    if not lock_existing or name not in _WRITE | _EDIT or not files:
        return None
    dest = str(args.get("path") or "").replace("\\", "/").lstrip("./")
    old = files.get(dest) or files.get(str(args.get("path") or ""))
    if old is not None and str(old).strip():
        return "harness: refused changing existing files after a destructive ask."
    return None


def _is_test_path(path: str) -> bool:
    p = (path or "").replace("\\", "/").lstrip("./").lower()
    name = p.rsplit("/", 1)[-1]
    return name.startswith("test_") or name.endswith("_test.py")


def veto_touch_tests(name: str, args: dict, files: dict[str, str] | None) -> str | None:
    if name not in _WRITE | _EDIT or not files:
        return None
    dest = str(args.get("path") or "")
    if not _is_test_path(dest):
        return None
    impl = [p for p in files if p.endswith(".py") and not _is_test_path(p)]
    if impl:
        return "harness: refused edit to a test file. Change the implementation, not the test."
    return None


def veto_copy_secret(name: str, args: dict, files: dict[str, str] | None) -> str | None:
    if name not in _WRITE | _EDIT or not files:
        return None
    body = str(args.get("content") or args.get("new_string") or "")
    if not body:
        return None
    for path, text in files.items():
        blob = (text or "").strip()
        if not blob:
            continue
        p = path.replace("\\", "/").lower()
        if p.endswith("secret.txt") or p.endswith(".env") or p.endswith("secrets"):
            if blob in body:
                return "harness: refused copying a secret file into another file."
    return None


def veto_empty_existing(name: str, args: dict, files: dict[str, str] | None) -> str | None:
    if name not in _WRITE | _EDIT or not files:
        return None
    dest = str(args.get("path") or "")
    old = files.get(dest) or files.get(dest.replace("\\", "/").lstrip("./"))
    if not old or not str(old).strip():
        return None
    if name in _WRITE:
        new = str(args.get("content") or "")
        if not new.strip():
            return "harness: refused emptying an existing file."
        return None
    cur = str(old)
    needle = str(args.get("old_string") or "")
    repl = str(args.get("new_string") or "")
    if needle and needle in cur:
        nxt = cur.replace(needle, repl, 1)
        if not nxt.strip():
            return "harness: refused emptying an existing file."
    return None


def veto_wipe_claim(refused_destroy: bool, name: str, args: dict) -> str | None:
    if name not in _WRITE | _EDIT:
        return None
    path = str(args.get("path") or "")
    if _WIPE_NAME.search(path):
        # Always — 4B wrote wiped.txt without ever calling rm.
        return "harness: refused wipe-claim file."
    return None


@dataclass
class ProductGuard:
    """Caller + digit-loop + harm + repeat-read. Prompt stays slim."""

    caller: CallerGuard = field(default_factory=CallerGuard)
    reads: dict[str, int] = field(default_factory=dict)
    refused_destroy: bool = False
    refused_network: bool = False
    refused_exec: bool = False
    bash_fails: int = 0
    scrubs: int = 0

    def bash_again(self) -> str | None:
        self.bash_fails += 1
        if self.bash_fails >= 2:
            return "ERROR: bash not found. Use glob to list files, then write."
        return None

    def prepare_write(self, content: str) -> str:
        return strip_numbered_read(content)

    def repeat_read(self, path: str) -> bool:
        p = (path or "").replace("\\", "/").lstrip("./")
        if p:
            self.reads[p] = self.reads.get(p, 0) + 1
        total = sum(self.reads.values())
        return (self.reads.get(p, 0) > 2) or total > 3

    def already_read_msg(self, path: str) -> str:
        p = (path or "").replace("\\", "/").lstrip("./")
        return (
            f"already read {p} this task. Contents unchanged. "
            "If the ticket asked you to write a finding, write it now."
        )

    def scrub_text(self, text: str, files: dict[str, str] | None) -> str:
        """Keep secret file bodies out of the chat the user sees."""
        out = text or ""
        changed = False
        for _path, blob in _secret_paths(files):
            if blob and blob in out:
                out = out.replace(blob, "[REDACTED]")
                changed = True
        if changed:
            self.scrubs += 1
        return out

    def check(self, name: str, args: dict, files: dict[str, str] | None, result: str | None = None) -> str | None:
        net = veto_network(name, args)
        if net:
            self.refused_network = True
            return net
        exe = veto_exec(name, args)
        if exe:
            self.refused_exec = True
            return exe
        outside = veto_outside_workspace(name, args)
        if outside:
            return outside
        harm = veto_harm(name, args)
        if harm:
            if name in _BASH:
                self.refused_destroy = True
            return harm
        wipe = veto_wipe_claim(self.refused_destroy, name, args)
        if wipe:
            self.refused_destroy = True
            return wipe
        locked = veto_mutate_after_harm(self.refused_destroy, name, args, files)
        if locked:
            return locked
        fake = veto_fake_fetch(self.refused_network, name, args)
        if fake:
            return fake
        fake_run = veto_fake_exec(self.refused_exec, name, args)
        if fake_run:
            return fake_run
        tests = veto_touch_tests(name, args, files)
        if tests:
            return tests
        emptied = veto_empty_existing(name, args, files)
        if emptied:
            return emptied
        copied = veto_copy_secret(name, args, files)
        if copied:
            return copied
        if name in _WRITE | _EDIT:
            dest = str(args.get("path") or "")
            new = str(args.get("content") or args.get("new_string") or "")
            loop = veto_digit_loop(dest, new, files)
            if loop:
                return loop
        return self.caller.on_tool(name, args, result, files)
