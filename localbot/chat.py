"""C3: ``localbot chat``: a terminal chat with a bot that works in its own folder.

What it is: a thin layer over ``run_task`` (C1 hooks ``next_user`` / ``on_event`` and C2 hook ``ask_delete``).
It never decides anything the loop decides. It lists models, makes the bot's folder, keeps one record file per
session, and prints one short line per step. The rules are always ``core4_product``; the shell is always a real
container; an undo store is always attached. If any of those is missing the chat refuses to start.

Bot folder: ``<home>/.localbot/bots/<NAME>/`` with ``work/`` (the bot's computer), ``undo/`` (snapshots, outside
``work/``) and ``sessions/`` (one ``.jsonl`` per session).

Everything outside the loop that touches the PC is in ``Deps`` so a test can swap it for a fake.

Terminal safety: every text that came from a model, a tool, a file name or a command goes through ``clean``
before it is printed, and every line of it is printed behind a prefix (``bot> ``, ``  > ``, ``   | ``). A model or a
file therefore cannot redraw the screen, and cannot make a line that looks like the [d]/[m]/[n] question.
Secrets: event text is already masked by the loop; the chat masks what it adds (what you typed, the final
summary, error text) the same way, and masks the events once more before they reach the record file.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TextIO

from .config import ContainerConfig, ProviderConfig, UndoConfig, get_profile, with_run_settings
from .guard.product import ProductGuard
from .guard.secret_names import SecretNames
from .guard.undo import UndoError, UndoStore
from .guard.workspace_fs import WorkspaceFS, new_work_folder
from .loop import run_task
from .provider.ollama import ProviderError, ProviderTimeout, ps_row
from .telemetry import read_vram  # C7: the existing nvidia-smi reader (None when nvidia-smi is missing or fails)
from .tools.fs import DiskWorkspace
from .tools.shell_container import ContainerError

if os.name == "nt":
    import msvcrt
else:
    import fcntl

PROFILE = "core4_product"
TRIAL_S = 30 * 24 * 3600.0  # 30 days: no limit in practice, and small enough for a thread wait (< threading.TIMEOUT_MAX)
# C7 part A: the chat's own defaults (the owner's choice, from the two real ladder runs). Only the chat uses them:
# the profiles in config.py (exam, gate, LONG) keep 512 and 8. Context (8192) and thinking (the model's own) stay.
CHAT_REPLY_LIMIT = 1024
CHAT_STEPS = 12
MAX_STEPS = 25  # the most steps to try, for now (the owner's choice)
REPEAT_RETRIES = 2  # C7 part B: a looping reply gets two second chances per message
NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,40}")  # used with fullmatch: $ would also accept a final newline
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
WHY_WORDS = {"delete": "DELETE", "empty": "EMPTY", "replace": "REPLACE", "move": "MOVE",
             "permissions": "CHANGE PERMISSIONS of"}  # C10b: a real why word (ladders and reports split asks by it)
PROMPT_ASK = "[d] delete  [m] move to _moved\\  [n] no (default) > "
PROMPT_ASK_MOVE = "Move it? (y/n) > "
PROMPT_ASK_PERM = "Allow it? (y/n) > "  # C10: a permissions question (or one that also moves); y/n like a move
YES_WHYS = ("move", "permissions")  # C10: the questions answered y/n
PERM_LINE = "The bot wants to {word}: {name}."  # C10: its own line shape (no "it was here before this chat")
HELP = (
    "/files   list the bot's folder\n"
    "/stats   numbers for this session\n"
    "/new     start a new conversation (same folder)\n"
    "/model   pick the model again (starts a new conversation)\n"
    "/undo N  put the folder back N steps (default 1)\n"
    "/help    this list\n"
    "/quit    leave (Ctrl+C works too)"
)

# C7: what the person is told when a message ended because the reply kept repeating itself (only real commands)
REPEAT_LINES = (
    "The model kept repeating the same lines over and over, even after it was asked twice for a short answer, so "
    "this message was stopped. Nothing in those replies was run.",
    "What you can do: send your message again (maybe shorter or clearer), type /model to pick another model, or "
    "move on to something else. /undo puts the folder back if you need it; /help lists the commands.",
)

# --- terminal safety ---------------------------------------------------------------------------------------

_ESC_SEQ = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"                     # CSI: colours, cursor moves, clear screen
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?"        # OSC: window title, hyperlinks
    r"|\x1b[PX^_][^\x1b]*(?:\x1b\\)?"              # DCS, SOS, PM, APC
    r"|\x1b[@-Z\\-_]|\x1b")                         # two-character escapes and a lone ESC
_CONTROL = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f\u061c\u200e\u200f\u2028\u2029\u202a-\u202e\u2066-\u2069]")


def clean(text: str) -> str:
    """Text safe to print: no escape sequences, no control characters (``\\n`` and ``\\t`` stay), no ``\\r``, no
    text-direction tricks. A CRLF becomes ``\\n``; a lone ``\\r`` is dropped."""
    text = (text or "").replace("\r\n", "\n")
    return _CONTROL.sub("", _ESC_SEQ.sub("", text))


def clean_name(name: str) -> str:
    """A file name for a one-line message: cleaned, and a line break or tab inside it shown as ``?``."""
    return clean(name).replace("\n", "?").replace("\t", "?")


def valid_bot_name(name: str) -> bool:
    return bool(NAME_RE.fullmatch(name or "")) and name.lower() not in _RESERVED


def ask_lines(paths: list[str], why: list[str], dests: list[str] | None = None) -> list[str]:
    """What the person is told about each path the bot wants to change (one line per path).

    C8: a move shows ``MOVE: source -> destination``; delete/empty/replace wording is unchanged.
    C10: a permission change shows ``CHANGE PERMISSIONS of: name.`` (``PERM_LINE``).
    """
    out = []
    for i, p in enumerate(paths):
        w = why[i] if i < len(why) else ""
        word = WHY_WORDS.get(w, "CHANGE")
        if w == "move":
            dest = (dests[i] if dests is not None and i < len(dests) else "") or "?"
            out.append(f"The bot wants to MOVE: {clean_name(p)} -> {clean_name(dest)}")
        elif w == "permissions":
            out.append(PERM_LINE.format(word=word, name=clean_name(p)))
        else:
            out.append(f"The bot wants to {word} {clean_name(p)} (it was here before this chat).")
    return out


# --- the PC, as one object a test can replace --------------------------------------------------------------


@dataclass
class Deps:
    home: Path
    runtime: Callable[[], str | None]                       # "podman" | "docker" | None (none running)
    provider: Callable[[ProviderConfig], Any]               # -> an OllamaProvider
    shell: Callable[[ContainerConfig, Path], Any]           # -> a ContainerShell (start / close / run)
    sweep: Callable[[ContainerConfig], Any]                 # remove containers nobody owns any more
    new_work: Callable[[Path], Path]                        # make the bot's folder (case-sensitive)
    make_store: Callable[[UndoConfig, str, Path], Any]      # -> an UndoStore, or None
    quiet: Callable[[], bool]                               # True when no Local Bot container is running
    commit: Callable[[], str]                               # the Local Bot commit, for the record header
    read_line: Callable[[str], "str | None"]                # one typed line; None at end of input
    out: TextIO
    now: Callable[[], datetime]
    clock: Callable[[], float] = time.monotonic
    vram: "Callable[[], Any] | None" = None                 # C7: the card's memory (nvidia-smi); None = not read


def _running_runtime() -> str | None:
    """Podman or Docker, whichever answers ``info``. Podman first (the product's default)."""
    import subprocess
    for exe in ("podman", "docker"):
        try:
            r = subprocess.run([exe, "info"], capture_output=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            continue
        if r.returncode == 0:
            return exe
    return None


def default_make_store(cfg: UndoConfig, bot: str, work: Path):
    return UndoStore(cfg, bot, work)


def default_new_work(path: Path) -> Path:
    new_work_folder(path)
    return path


def _default_commit() -> str:
    try:
        from .exam.runner import git_record
        rec = git_record()
        return (rec.get("commit") or "unknown")[:12] + (" (uncommitted changes)" if rec.get("dirty") else "")
    except Exception:
        return "unknown"


def _stdin_line(prompt: str) -> "str | None":
    sys.stdout.write(prompt)
    sys.stdout.flush()
    line = sys.stdin.readline()
    return None if line == "" else line.rstrip("\r\n")


def default_deps() -> Deps:
    from .provider.ollama import OllamaProvider
    from .tools.shell_container import ContainerShell
    from . import __main__ as cli
    try:
        sys.stdout.reconfigure(errors="replace")  # a stray character must never crash the printing
    except (AttributeError, ValueError):
        pass
    return Deps(
        home=Path.home(),
        runtime=_running_runtime,
        provider=lambda cfg: OllamaProvider(cfg),
        shell=lambda cfg, work: ContainerShell(cfg, work),
        sweep=lambda cfg: ContainerShell.sweep(cfg),
        new_work=default_new_work,
        make_store=default_make_store,
        quiet=lambda: cli.running_localbot_containers() == [],
        commit=_default_commit,
        read_line=_stdin_line,
        out=sys.stdout,
        now=lambda: datetime.now(timezone.utc),
        vram=read_vram,
    )


# --- one conversation's numbers and the speed meter ---------------------------------------------------------


class _Metered:
    """The provider, with a running total of what the model generated (for /stats). Everything else passes through."""

    def __init__(self, inner: Any, on_first: Callable[[], Any] | None = None, think: bool | None = None) -> None:
        self.inner, self.eval_count, self.eval_s = inner, 0, 0.0
        self.on_first, self.think, self.think_error = on_first, think, None  # C6: the gpu_fit note; the --think refusal

    def chat(self, *a: Any, num_predict: int | None = None, **k: Any):
        try:
            if num_predict is None:
                res = self.inner.chat(*a, **k)
            else:  # C7: this one call with a smaller reply limit (the loop asks for it after a 2nd looping reply)
                cfg = self.inner.cfg
                self.inner.cfg = dataclasses.replace(cfg, num_predict=num_predict, num_predict_by_model=())
                try:
                    res = self.inner.chat(*a, **k)
                finally:
                    self.inner.cfg = cfg
        except Exception as exc:
            # C6: with --think set, an error Ollama answers (not a timeout) is most likely "this model can't switch
            # thinking". Keep its words so the chat can say it plainly and stop. The loop still ends the message as before.
            if self.think is not None and isinstance(exc, ProviderError) and not isinstance(exc, ProviderTimeout):
                self.think_error = str(exc)
            raise
        if self.on_first is not None:
            first, self.on_first = self.on_first, None
            try:
                first()
            except Exception:
                pass  # a note about the card must never end a chat
        n, s = getattr(res, "eval_count", None), getattr(res, "eval_s", None)
        if n and s:
            self.eval_count += int(n)
            self.eval_s += float(s)
        return res

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


@dataclass
class _Stats:
    messages: int = 0
    steps: int = 0
    refused: int = 0
    asks: int = 0
    answers: dict | None = None
    cut: int = 0  # C5 F2: for the message in progress (reset when it ends)
    unread: int = 0
    ran: int = 0
    loops: int = 0  # C7: looping replies in the message in progress (reset when it ends)

    def __post_init__(self) -> None:
        self.answers = {"delete": 0, "move": 0, "no": 0, "yes": 0}


def _duration(s: float) -> str:
    s = int(s)
    return f"{s // 3600} h {s % 3600 // 60} min {s % 60} s" if s >= 3600 else f"{s // 60} min {s % 60} s"


class _Ended(Exception):
    """Raised inside the chat to leave with a plain message and an exit code."""

    def __init__(self, message: str, code: int = 2) -> None:
        super().__init__(message)
        self.message, self.code = message, code


# --- one chat per bot --------------------------------------------------------------------------------------


class BotBusy(Exception):
    """Another process already has this bot open."""


class BotLock:
    """A held lock on one bot. ``release()`` lets it go; so does the death of the process, however it dies."""

    def __init__(self, fd: int) -> None:
        self._fd: int | None = fd

    def release(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            if os.name == "nt":
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            os.close(fd)


def acquire_bot_lock(path: Path) -> BotLock:
    """Take the byte-range lock on byte 0 of ``path`` (made if needed) or raise ``BotBusy`` at once; never waits.
    The same kind of lock as scripts/runlock.py: the OS owns it, so there is never a stale lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if os.name == "nt":
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(fd)
        raise BotBusy(str(path)) from exc
    return BotLock(fd)


# --- the chat ----------------------------------------------------------------------------------------------


_SETTING_WORDS = (("num_predict", "reply limit"), ("num_ctx", "context"), ("max_turns", "steps"))


def setting_message(exc: Exception) -> str:
    """C6: with_run_settings' error in the words of the chat's flags."""
    text = str(exc)
    for old, new in _SETTING_WORDS:
        text = text.replace(old, new)
    return f"Those settings can't be used: {text}. Nothing was started."


class Chat:
    def __init__(self, bot: str, model: str | None, deps: Deps, *, reply_limit: int | None = None,
                 context: int | None = None, steps: int | None = None, think: bool | None = None) -> None:
        self.bot, self.want_model, self.deps = bot, model, deps
        self.settings = {"reply_limit": reply_limit, "context": context, "steps": steps}  # None = today's value
        self.think = think
        self.gpu_done, self.gpu = False, None
        self.sizes: dict = {}
        profile = get_profile(PROFILE)
        # A chat has no whole-conversation clock: the person's thinking time and a long chat must not end it.
        # One model call keeps its own time limit; one message keeps ``max_turns``.
        self.profile = dataclasses.replace(
            profile,
            watchdog=dataclasses.replace(profile.watchdog, task_wall_clock_s=None),
            loop=dataclasses.replace(profile.loop, trial_deadline_s=TRIAL_S))
        self.out = deps.out
        self.model: str | None = None
        self.prov: Any = None
        self.store: Any = None
        self.bot_dir = deps.home / ".localbot" / "bots" / bot
        self.work = self.bot_dir / "work"
        self.record: TextIO | None = None
        self.record_path: Path | None = None
        self.end_reason = "quit"
        self.mask: Callable[[str], str] = lambda t: t
        self.pending: str | None = None          # what ended the last run_task: "new", "model", "undo:N", "quit", "eof"
        self.pending_ask: dict | None = None
        self.stats = _Stats()
        self.conv = 0
        self.t_start = deps.clock()
        self.metered: _Metered | None = None
        self.swept = False
        self.exit_code = 0
        self.lock: BotLock | None = None

    # ---- printing
    def w(self, text: str = "") -> None:
        self.out.write(text + "\n")
        try:
            self.out.flush()
        except Exception:
            pass

    def plain(self, text: str) -> None:
        """Our own words (not from a model, a tool or a file)."""
        self.w(text)

    def block(self, first: str, text: str) -> None:
        lines = clean(text).split("\n")
        self.w(first + lines[0])
        for ln in lines[1:]:
            self.w("   | " + ln)

    # ---- masking
    def mask_obj(self, v: Any) -> Any:
        if isinstance(v, str):
            return self.mask(v)
        if isinstance(v, dict):
            return {k: self.mask_obj(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [self.mask_obj(x) for x in v]
        return v

    # ---- the record
    def open_record(self) -> None:
        sessions = self.bot_dir / "sessions"
        sessions.mkdir(parents=True, exist_ok=True)
        now = self.deps.now()
        base = f"{now.strftime('%Y%m%dT%H%M%SZ')}_{re.sub(r'[^A-Za-z0-9._-]', '_', self.model or 'model')}"
        n, path = 1, sessions / f"{base}.jsonl"
        while path.exists():
            n += 1
            path = sessions / f"{base}_{n}.jsonl"
        self.record_path = path
        self.record = open(path, "w", encoding="utf-8", newline="\n")
        self.gpu_done, self.gpu = False, None
        self.rec({"kind": "header", "bot": self.bot, "model": self.model, "profile": PROFILE,
                  "commit": self.deps.commit(), "started": now.isoformat(),
                  "reply_limit": self.profile.provider.num_predict_for(self.model or ""),
                  "context": self.profile.provider.num_ctx, "steps": self.profile.loop.max_turns,
                  "think": "default" if self.think is None else "on" if self.think else "off"})

    def gpu_note(self) -> None:
        """C6: after the model's first reply (so it is loaded), one record line: its size, how much of it is on the
        card, and whether it all fits. ``doubtful`` when /api/ps reports far less than the model's size on disk
        (it has under-reported before), so a reader knows not to trust a "yes"."""
        if self.gpu_done:
            return
        self.gpu_done = True
        row = ps_row(self.prov.ps() or [], self.model or "")
        disk = self.sizes.get(self.model)
        size, vram = (int(row.get("size") or 0), int(row.get("size_vram") or 0)) if row else (0, 0)
        fits = "unknown" if not size else "yes" if vram >= size else "no"
        doubtful = bool(size and isinstance(disk, (int, float)) and disk and size < 0.5 * disk)
        used_mib = total_mib = None
        if self.deps.vram is not None:  # C7: what the card itself says (nvidia-smi), next to what /api/ps says
            try:
                card = self.deps.vram()
            except Exception:
                card = None  # no reading must ever end a chat
            used_mib, total_mib = getattr(card, "used_mib", None), getattr(card, "total_mib", None)
        self.gpu = {"model_bytes": size or None, "on_card_bytes": vram if size else None, "fits": fits,
                    "doubtful": doubtful, "disk_bytes": int(disk) if isinstance(disk, (int, float)) else None,
                    "card_used_mib": used_mib, "card_total_mib": total_mib}
        self.rec({"kind": "gpu_fit", **self.gpu})

    def rec(self, obj: dict) -> None:
        if self.record is None:
            return
        self.record.write(json.dumps(self.mask_obj(obj), ensure_ascii=False, default=str) + "\n")
        self.record.flush()

    def close_record(self, reason: str) -> None:
        if self.record is None:
            return
        try:
            self.rec({"kind": "end", "reason": reason, "at": self.deps.now().isoformat()})
        finally:
            self.record.close()
            self.record = None
            self.w(f"Session record: {self.record_path}")

    # ---- start
    def preflight(self) -> None:
        d = self.deps
        if not valid_bot_name(self.bot):
            raise _Ended(f"'{clean_name(self.bot)}' is not a usable bot name. Use letters, digits, - and _ only "
                         "(up to 40 characters).")
        runtime = d.runtime()
        if not runtime:
            raise _Ended("No container runtime is running, and a chat needs one: the bot's commands only ever run "
                         "inside a container. Start Podman or Docker, then try again. Nothing was started.")
        cont = dataclasses.replace(self.profile.container, runtime=runtime)
        self.profile = dataclasses.replace(self.profile, container=cont)
        # one chat per bot: two windows would change one work folder and number undo snapshots over each other
        try:
            self.lock = acquire_bot_lock(self.bot_dir / "chat.lock")
        except BotBusy as exc:
            raise _Ended(f"Bot '{self.bot}' is already open in another window. Close that chat first, or use --bot "
                         "with another name. Nothing was started.") from exc
        except OSError as exc:
            raise _Ended(f"I could not take the lock for bot '{self.bot}' ({clean(str(exc))}). "
                         "Nothing was started.") from exc
        # the bot's folder
        try:
            self.bot_dir.mkdir(parents=True, exist_ok=True)
            if not self.work.exists():
                d.new_work(self.work)
        except Exception as exc:
            raise _Ended(f"I could not make the bot's work folder ({self.work}): {clean(str(exc))} "
                         "Nothing was started.") from exc
        # the undo store: a chat never starts without one
        cfg = dataclasses.replace(self.profile.undo, store_dir=d.home / ".localbot" / "bots")
        try:
            self.store = d.make_store(cfg, self.bot, self.work)
        except Exception as exc:
            raise _Ended(f"I refuse to start: the undo store could not be made ({clean(str(exc))}). A chat "
                         "always keeps undo snapshots, so nothing was started.") from exc
        if self.store is None:
            raise _Ended("I refuse to start: there is no undo store. A chat always keeps undo snapshots, so "
                         "nothing was started.")
        self.profile = dataclasses.replace(self.profile, undo=cfg)

    def models(self) -> list[str]:
        try:
            self.prov = self.deps.provider(self.profile.provider)
            if self.think is not None:
                self.prov.think = self.think  # C6: only a chat given --think changes what is sent
            tags = self.prov.tags()
        except Exception as exc:
            tags = []
            self.plain(f"(Ollama said: {clean(str(exc))})")
        names = []
        for t in tags or []:
            n = t.get("name") or t.get("model") if isinstance(t, dict) else None
            if n:
                names.append(str(n))
        if not names:
            raise _Ended("I can't see any models. Is Ollama running? Start it (and install a model in it if you "
                         "have none), then try again.")
        self.sizes = {(t.get("name") or t.get("model")): t.get("size") for t in tags if isinstance(t, dict)}
        return names

    def pick_model(self, names: list[str], first: bool) -> bool:
        if first and self.want_model:
            if self.want_model in names:
                self.model = self.want_model
                return True
            self.plain(f"The model '{clean_name(self.want_model)}' is not installed in Ollama.")
        self.plain("Models installed in Ollama:")
        for i, n in enumerate(names, 1):
            size = self.sizes.get(n)
            self.plain(f"  {i}. {clean_name(n)}" + (f"  ({size / 1e9:.1f} GB)" if isinstance(size, (int, float)) else ""))
        for _ in range(3):
            raw = self.deps.read_line("Pick a model (number or name)> ")
            if raw is None:
                break
            s = raw.strip()
            if s.isdigit() and 1 <= int(s) <= len(names):
                self.model = names[int(s) - 1]
                return True
            if s in names:
                self.model = s
                return True
            self.plain(f"Type a number from 1 to {len(names)}.")
        self.plain("No model chosen, so nothing was started.")
        return False

    # ---- the session
    def run(self) -> int:
        code = 0
        try:
            try:
                s = self.settings
                n = s["steps"]
                if isinstance(n, int) and not isinstance(n, bool) and n > MAX_STEPS:
                    self.plain(f"Those settings can't be used: steps can be at most {MAX_STEPS} for now (you asked "
                               f"for {n}). Nothing was started.")
                    return 2
                # C7: the chat's own defaults where no flag was given (the profile itself keeps its values)
                self.profile = with_run_settings(
                    self.profile, num_ctx=s["context"], max_turns=CHAT_STEPS if n is None else n,
                    num_predict=CHAT_REPLY_LIMIT if s["reply_limit"] is None else s["reply_limit"])
            except ValueError as exc:
                self.plain(setting_message(exc))
                return 2
            self.preflight()
            names = self.models()
            if not self.pick_model(names, True):
                return 2
            while True:
                if not self.swept:
                    self.deps.sweep(self.profile.container)
                    self.swept = True
                self.open_record()
                self.plain(f"Bot '{self.bot}', model {clean_name(self.model or '')}. Folder: {self.work}. "
                           "Type /help for the commands.")
                action = self.conversations()
                self.close_record(self.end_reason)
                if self.exit_code:
                    code = self.exit_code
                    break
                if action == "quit":
                    break
                if not self.pick_model(names, False):
                    code = 2
                    break
        except _Ended as e:
            self.plain(e.message)
            code = e.code
        except KeyboardInterrupt:
            self.plain("")
            self.plain("Interrupted (Ctrl+C).")
            self.end_reason, code = "interrupted", 130
        except Exception as exc:
            self.plain(f"Something went wrong inside the chat: {type(exc).__name__}: {clean(self.mask(str(exc)))}")
            self.end_reason, code = "crash", 1
        finally:
            try:
                self.close_record(self.end_reason)
            finally:
                if self.lock is not None:  # last: the record and the container are closed before the bot is free
                    self.lock.release()
                    self.lock = None
        return code

    def conversations(self) -> str:
        """Conversations one after another until the person quits or asks for another model."""
        self.end_reason = "quit"
        while True:
            first: str | None = None
            if self.pending is not None:
                act, self.pending = self.pending, None
            else:
                kind, val = self.read_input()
                if kind == "msg":
                    first, act = val, ""
                else:
                    act = val
            if first is not None:
                if not self.converse(first):
                    return "quit"
                continue
            if act in ("quit", "eof"):
                self.end_reason = act
                return "quit"
            if act == "model":
                self.end_reason = "model-change"
                return "model"
            if act == "new":
                self.plain("New conversation. The bot's folder stays as it is.")
            elif act.startswith("undo:"):
                self.undo(act[5:])

    def read_input(self) -> tuple[str, str]:
        """One typed line: ("msg", text) or ("end", what-it-ends). The commands that only print are handled here."""
        while True:
            raw = self.deps.read_line("you> ")
            if raw is None:
                return "end", "eof"
            line = raw.strip()
            if not line:
                continue
            word = line.split(None, 1)[0]
            if not re.fullmatch(r"/[A-Za-z]+", word):
                return "msg", line
            arg = line[len(word):].strip()
            cmd = word.lower()
            if cmd == "/quit":
                return "end", "quit"
            if cmd == "/new":
                return "end", "new"
            if cmd == "/model":
                return "end", "model"
            if cmd == "/undo":
                return "end", "undo:" + arg
            if cmd == "/files":
                self.files()
            elif cmd == "/stats":
                self.show_stats()
            elif cmd == "/help":
                self.plain(HELP)
            else:
                self.plain(f"unknown command {clean_name(word)}. Type /help for the list.")

    def next_user(self, _last: str) -> "str | None":
        kind, val = self.read_input()
        if kind == "msg":
            self.stats.messages += 1
            self.rec({"kind": "user", "conversation": self.conv, "text": val})
            return val
        self.pending = val
        return None

    # ---- one conversation
    def converse(self, first: str) -> bool:
        """Run one conversation. False when the session must end (the container would not start)."""
        d = self.deps
        self.conv += 1
        ws = DiskWorkspace(WorkspaceFS(self.work))
        names = SecretNames(globs=self.profile.container.secret_globs, templates=self.profile.container.secret_templates)
        guard = ProductGuard(shell_mode="container", rules="product", secret_names=names)
        guard.attach(ws.view)
        self.mask = lambda t: guard.scrub_text(t, ws.view)
        self.metered = _Metered(self.prov, on_first=self.gpu_note, think=self.think)
        self.stats.messages += 1
        self.rec({"kind": "user", "conversation": self.conv, "text": first})
        shell = d.shell(self.profile.container, self.work)
        try:
            try:
                shell.start()
            except ContainerError as exc:
                self.plain(f"The container would not start: {clean(self.mask(str(exc)))}")
                self.end_reason = "container-error"
                self.exit_code = 1
                return False
            res = run_task(self.profile, self.metered, self.model, first, ws, shell=shell, undo=self.store,
                           next_user=self.next_user, on_event=self.on_event, ask_delete=self.ask,
                           repeat_retries=REPEAT_RETRIES)
            self.rec({"kind": "summary", "conversation": self.conv, "summary": res.summary()})
            if self.metered.think_error:
                word = "on" if self.think else "off"
                self.plain(f"Ollama would not run {clean_name(self.model or '')} with --think {word}: "
                           f"{clean(self.mask(self.metered.think_error))[:400]}")
                self.plain("Nothing was retried and the chat has stopped. Start it again without --think, "
                           "or with the other value.")
                self.end_reason = "think-error"
                self.exit_code = 1
                return False
            if self.pending is None:
                self.plain("(That conversation has ended. Your next message starts a new one.)")
            return True
        finally:
            try:
                shell.close()
            except Exception as exc:
                self.plain(f"(I could not close the container cleanly: {clean(str(exc))})")


    # ---- live lines
    def on_event(self, ev: dict) -> None:
        ev = self.mask_obj(ev)
        self.rec({"kind": "event", "event": ev})
        kind, st = ev.get("event"), self.stats
        if kind in ("say", "answer"):
            if (ev.get("text") or "").strip():
                self.block("bot> ", ev["text"])
        elif kind == "tool":
            status = ev.get("status")
            if status == "ran":
                st.steps += 1
                st.ran += 1
                self.block("  > ", f"{ev.get('tool')}: {self.tool_what(ev)}")
            elif status == "undone":
                st.refused += 1
                self.block("  > ", f"{ev.get('tool')}: {self.tool_what(ev)} (this step was undone)")
        elif kind == "refusal":
            st.refused += 1
            self.block("  ! ", f"refused: {ev.get('tool')} - {ev.get('reason')}")
        elif kind == "cut":
            st.cut += 1
            self.plain(self.cut_line(ev))
        elif kind == "loop":
            st.loops += 1
            self.plain(self.loop_line(ev))
        elif kind == "parse_fail":
            st.unread += 1
            self.plain("  ! reply not understood (no tool call or answer found); asked again")
        elif kind == "ask":
            self.pending_ask = ev
            st.asks += 1
        elif kind == "ask_answer":
            ans = ev.get("answer")
            if ans in st.answers:
                st.answers[ans] += 1
            self.plain(f"  you chose: {ans}")
        elif kind == "message_end":
            if ev.get("end_reason") not in (None, "done", "answered"):
                why = clean_name(str(ev.get("end_reason")))
                if why == "max_steps":
                    why += f"; {st.cut} replies cut, {st.unread} not understood, {st.ran} tools run"
                elif why == "repetition" and st.loops:
                    why += f"; the reply repeated itself {st.loops} times"
                self.plain(f"  (this message ended: {why})")
                if ev.get("end_reason") == "repetition":
                    for ln in REPEAT_LINES:
                        self.plain(ln)
            st.cut = st.unread = st.ran = st.loops = 0

    @staticmethod
    def loop_line(ev: dict) -> str:
        """C7: one line per looping reply. Numbers only: the reply's own words are never shown."""
        count, retries, nxt = ev.get("count"), ev.get("retries"), ev.get("next_limit")
        if not ev.get("retry"):
            return "  ! reply repeated itself again; no retries left, so this message stops"
        if isinstance(nxt, int) and not isinstance(nxt, bool):
            return (f"  ! reply repeated itself again; asked for a very short answer, next reply at most {nxt} tokens "
                    f"({count} of {retries} retries)")
        return f"  ! reply repeated itself; asked for a short answer ({count} of {retries} retries)"

    @staticmethod
    def cut_line(ev: dict) -> str:
        limit = ev.get("limit")
        at = f"{limit} tokens" if isinstance(limit, int) and not isinstance(limit, bool) else "the output limit"
        if ev.get("visible"):
            what = ""
        elif ev.get("thinking_tokens"):
            what = " (all of it was thinking, nothing visible)"
        else:
            what = " (nothing visible)"
        return f"  ! reply cut at {at}{what}; asked for a shorter one"

    @staticmethod
    def tool_what(ev: dict) -> str:
        raw = ev.get("args") or ""
        try:
            args = json.loads(raw)
        except (TypeError, ValueError):
            args = None
        if isinstance(args, dict):
            for key in ("command", "path", "file", "pattern", "query"):
                if isinstance(args.get(key), str):
                    return args[key] if len(args[key]) <= 160 else args[key][:160] + "..."
        return raw if len(raw) <= 160 else raw[:160] + "..."

    # ---- the question about task-start files
    def ask(self, _paths: list[str]) -> str:
        ev, self.pending_ask = self.pending_ask, None
        if ev is None:
            return "no"  # no event to show means nothing safe to show: refuse
        whys = list(ev.get("why") or [])
        dests = ev.get("dests")
        for ln in ask_lines(ev.get("paths") or [], whys, dests=dests):
            self.plain(ln)
        move_only = bool(whys) and all(w == "move" for w in whys)
        yes_only = bool(whys) and all(w in YES_WHYS for w in whys)  # C10: a permissions question is y/n too
        prompt = PROMPT_ASK_MOVE if move_only else PROMPT_ASK_PERM if yes_only else PROMPT_ASK
        for _ in range(3):
            try:
                raw = self.deps.read_line(prompt)
            except KeyboardInterrupt:
                self.plain("")
                return "no"
            if raw is None:
                self.plain("")
                return "no"
            a = raw.strip().lower()
            if yes_only:
                if a in ("", "n"):
                    return "no"
                if a == "y":
                    return "yes"
                self.plain("Please type y or n (Enter means no).")
            else:
                if a in ("", "n"):
                    return "no"
                if a == "d":
                    return "delete"
                if a == "m":
                    return "move"
                self.plain("Please type d, m or n (Enter means no).")
        return "no"

    # ---- slash commands
    def files(self) -> None:
        rows = []
        for root, dirs, names in os.walk(self.work, followlinks=False):
            dirs.sort()
            for n in sorted(names):
                p = Path(root) / n
                try:
                    size = p.lstat().st_size
                except OSError:
                    size = 0
                rows.append(f"  {clean_name(p.relative_to(self.work).as_posix())}  ({size} bytes)")
        if not rows:
            self.plain("The bot's folder is empty.")
        for r in rows[:100]:
            self.plain(r)
        if len(rows) > 100:
            self.plain(f"  ... and {len(rows) - 100} more")

    def show_stats(self) -> None:
        st, m = self.stats, self.metered
        rate = (f"{m.eval_count / m.eval_s:.1f} tokens/s" if m and m.eval_s > 0 else "not measured yet")
        a = st.answers
        self.plain(f"  messages you sent: {st.messages}\n"
                   f"  tool steps: {st.steps}\n"
                   f"  refused or undone: {st.refused}\n"
                   f"  asked about files: {st.asks} (delete {a['delete']}, move {a['move']}, no {a['no']})\n"
                   f"  model speed: {rate}\n"
                   f"  session time: {_duration(self.deps.clock() - self.t_start)}\n"
                   f"  reply limit: {self.profile.provider.num_predict_for(self.model or '')} tokens\n"
                   f"  context: {self.profile.provider.num_ctx} tokens\n"
                   f"  steps: {self.profile.loop.max_turns}\n"
                   f"  thinking: {'the model\'s own setting' if self.think is None else 'on' if self.think else 'off'}\n"
                   f"  {self.gpu_text()}")

    def gpu_text(self) -> str:
        g = self.gpu
        if not g:
            return "fits on the card: not measured yet (it is read after the model's first reply)"
        gb = lambda n: "?" if n is None else f"{n / 1e9:.1f} GB"
        extra = " (doubtful: Ollama reports much less than the model's size on disk)" if g["doubtful"] else ""
        card, used, total = "", g.get("card_used_mib"), g.get("card_total_mib")
        if isinstance(used, int) and isinstance(total, int) and total:  # C7: nvidia-smi's own numbers
            card = f"; card memory used {used * 1048576 / 1e9:.1f} of {total * 1048576 / 1e9:.1f} GB (nvidia-smi)"
        return (f"fits on the card: {g['fits']} (model {gb(g['model_bytes'])}, on the card {gb(g['on_card_bytes'])})"
                f"{extra}{card}")

    def undo(self, arg: str) -> None:
        arg = arg.strip()
        try:
            n = int(arg) if arg else 1
        except ValueError:
            self.plain("Nothing was restored: /undo takes a whole number, like /undo 2.")
            return
        try:
            rep = self.store.restore(n, quiet=self.deps.quiet)
        except UndoError as exc:
            self.plain(f"Nothing was restored: {clean(str(exc))}")
            return
        except Exception as exc:
            self.plain(f"Nothing was restored: {type(exc).__name__}: {clean(str(exc))}")
            return
        self.plain(f"Put the folder back {n} step(s): restored snapshot {rep['restored_to']}, verified; "
                   f"removed {len(rep['removed'])}, folders made {len(rep['dirs_made'])}, "
                   f"files written {len(rep['written'])}.")
        for rel in rep.get("not_restored_large", []):
            self.plain(f"  not restored (too big to have been stored): {clean_name(str(rel))}")


def chat_main(bot: str = "main", model: str | None = None, deps: Deps | None = None, *,
              reply_limit: int | None = None, context: int | None = None, steps: int | None = None,
              think: bool | None = None) -> int:
    return Chat(bot, model, deps or default_deps(), reply_limit=reply_limit, context=context, steps=steps,
                think=think).run()