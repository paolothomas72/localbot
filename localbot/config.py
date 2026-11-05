"""All Local Bot knobs in one place, grouped into named profiles.

Two profiles exist (docs/STAGE1_SPEC.md §2, §3, §5):
- ``parity_lab6``: copies the lab loop exactly (lab_ref/scripts/run_lh_product.py --slim):
  the lab's six tools, emulated shell, no output caps, no compaction, no text-call parsing.
  Its only change from the lab is that it runs on a real disk.
- ``core4``: the new default. Four tools (read/write/edit/bash), container shell, undo on,
  context caps and compaction on, text-call parsing on.

Values that the spec does not fix yet are marked ``# TBD Sx``. The step that builds the
component sets them and logs them in docs/BUILD_LOG.md.

Paths come from the environment where a machine-specific value is needed, so nothing
machine-specific is hard-coded here:
- ``LOCALBOT_STATE_DIR``: undo snapshots, bot state and the bots' workspaces (default: the user's
  own state folder, ``user_state_dir()``; H1: no longer ``<source tree>/state``, since an
  installed package has no source tree).
- ``LOCALBOT_OLLAMA_URL``: Ollama base URL (default: ``http://127.0.0.1:11434``). Must be localhost.

``REPO_ROOT`` is the source checkout. Only the exam uses it (``lab_ref/``, ``results/``, the
trial folders): those exist in a checkout only.
"""
from __future__ import annotations

import dataclasses
import os
import posixpath
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping
from urllib.parse import urlparse

from localbot.guard.secret_names import SECRET_GLOBS, TEMPLATE_GLOBS

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- System prompt and tool schemas -------------------------------------------------------

# Copied verbatim from lab_ref/scripts/run_lh_product.py SYSTEM_SLIM.
# tests/test_config.py checks that it still matches the lab source.
SYSTEM_SLIM = (
    "You are a local coding agent. Use tools. Be honest. "
    "Strip line-number prefixes from read() before edit or write. "
    "Do the smallest change that matches intent. "
    "If asked for JSON only, reply with JSON only."
)

# core4 only (S8c): appended to SYSTEM_SLIM. SYSTEM_SLIM itself is unchanged (tests/test_config.py
# checks it against the lab).
CORE4_SEARCH_HINT = "Your workspace is the current folder. To search files, use bash, for example: grep -rn 'name' ."

# Copied verbatim from lab_ref/scripts/run_all_fit.py TOOLS (tests/test_config.py checks it).
TOOLS_LAB6: list[dict[str, Any]] = [
    {"type": "function", "function": {"name": "read", "description": "Read a file. Numbered lines (N\\tcontent).", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write", "description": "Create or overwrite a file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit", "description": "Replace old_string with new_string.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}, "required": ["path", "old_string", "new_string"]}}},
    {"type": "function", "function": {"name": "glob", "description": "List files.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search file contents.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "bash_exec", "description": "Run bash. Often fails here.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]

# The four tools every model knows. read/write/edit keep the lab's schemas so the only
# tool difference from parity is glob/grep/bash_exec -> bash.
TOOLS_CORE4: list[dict[str, Any]] = [
    TOOLS_LAB6[0],
    TOOLS_LAB6[1],
    TOOLS_LAB6[2],
    {"type": "function", "function": {"name": "bash", "description": "Run a bash command in the workspace (Linux, no network). Use it for ls, grep, find, tests.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]


# --- Knob groups -------------------------------------------------------------------------


def _ollama_url() -> str:
    url = os.environ.get("LOCALBOT_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    host = urlparse(url).hostname
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(f"LOCALBOT_OLLAMA_URL must be localhost (rule: no network but local Ollama), got {host!r}")
    return url


@dataclass(frozen=True)
class ProviderConfig:
    """Ollama native /api/chat. Lab values: lab_ref/lib/ollama_chat.py + fit_roster.DEFAULT_CTX."""

    base_url: str = field(default_factory=_ollama_url)
    stream: bool = False
    temperature: float = 0.2
    num_predict: int = 512
    num_ctx: int = 8192
    keep_alive: str = "2m"
    retries_on_http_500: int = 1  # lab: 2 attempts, the second only when the error mentions 500
    # H1 loop P2: a refused / dropped connection (Ollama restarting) is retried after a pause that
    # doubles each time. The lab did not retry these (parity: 0).
    retries_on_connection_error: int = 2
    retry_backoff_s: float = 1.0
    # H1 L-6: per-model output budget, ((model tag, num_predict), ...). The capability card sets it
    # (Stage 2); a model not listed gets ``num_predict``.
    num_predict_by_model: tuple[tuple[str, int], ...] = ()

    def num_predict_for(self, model: str) -> int:
        return dict(self.num_predict_by_model).get(model, self.num_predict)


@dataclass(frozen=True)
class LoopConfig:
    system_prompt: str = SYSTEM_SLIM
    max_turns: int = 8
    task_timeout_s: int = 180
    trial_deadline_s: int = 900  # whole 12-task trial
    # Run parse.fn_call when a reply has no native tool_calls, and only while the model has made
    # no native call in this task (H1 L-1): a model with native calls that writes a call as text
    # is quoting it (e.g. a file it read), so that call is recorded, not run.
    text_call_parse: bool = True
    # The model's card says it writes calls as text (a text dialect): text calls run on every
    # turn, native call or not. No card sets it yet (Stage 2); off in both profiles.
    text_dialect: bool = False
    parse_fail_nudge: bool = True  # a nudge when a reply looks like a call but nothing parsed
    parse_fail_end_after: int = 3  # H1 P2: ... the task ends parse_fail at this many in a row
    # Observation only: when a reply has no native tool_calls, record whether its text held a
    # call (parsed calls, has_tool_call_attempt, unparsed dialect). Never executed, never nudged.
    record_text_call_attempts: bool = True
    # Map read_file -> read, patch -> edit, ... (parse/dialects.ALIASES). The lab did not: it
    # answered "ERROR: unknown tool read_file". Off in parity for the same reason as parsing.
    dialect_aliases: bool = True
    # H1 L-6: a reply that stopped at the output limit (done_reason "length") is never run and
    # never ends the task; the model is told it was cut and to write in parts. The lab ran it.
    refuse_cut_replies: bool = True
    # H1 L-4: at most this many calls of one reply run (the rest are named back to the model).
    # None = no cap (the lab had none).
    max_calls_per_turn: int | None = 8
    # H1 P2: tool output holding a call block, a forged "harness:" line or "ignore previous
    # instructions" gets a header saying it is data, not instructions. The lab showed it raw.
    mark_untrusted_output: bool = True


@dataclass(frozen=True)
class ToolsConfig:
    schemas: tuple[dict[str, Any], ...]
    shell_mode: Literal["emulated", "container"]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(t["function"]["name"] for t in self.schemas)


@dataclass(frozen=True)
class ContextConfig:
    """None means off (the lab loop had no caps, no eviction, no compaction)."""

    tool_output_cap_chars: int | None = 4000
    evict_tool_results_over_chars: int | None = 2500
    keep_first_messages: int = 2
    keep_last_messages: int = 6
    compaction: bool = True
    compaction_trigger: float = 0.80
    compaction_target: float = 0.60


@dataclass(frozen=True)
class WatchdogConfig:
    """Deadlines come from LoopConfig. None means that check is off."""

    stuck_nudge_after: int | None = 3  # same call signature N times -> nudge
    stuck_end_after: int | None = 5  # ... -> end with reason "stuck"
    repetition_min_unique_line_ratio: float | None = 0.3  # TBD S8: tune on recorded outputs
    hard_overrun_s: int = 10  # no trial may run past its deadline + this
    # Whole-task wall clock (S4). The lab had none: its 180 s was the timeout of each provider
    # call (LoopConfig.task_timeout_s), so parity turns this off. The trial deadline is
    # enforced mid-task in every profile (G7).
    task_wall_clock_s: int | None = 180


@dataclass(frozen=True)
class ContainerConfig:
    runtime: Literal["podman", "docker"] = "podman"
    # S7b.0: the official python:3.12-slim as pulled by the owner into BOTH podman and docker,
    # pinned by its multi-platform index digest. bash 5.2 + GNU coreutils 9.7 / grep 3.11 /
    # findutils 4.10 + python 3.12.15, Debian 13 (spec §5.6); ~124 MB. No `ps` (the shell reads
    # /proc), no curl/wget/git. (S7a used nikolaik/python-nodejs@sha256:8f958…, 1.56 GB.)
    image: str | None = "docker.io/library/python@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704"
    label_key: str = "localbot-container"  # leftover containers with this label are swept at run start
    # H1 S-2: a container expires this long after it starts (the sweep removes it once past);
    # one made before H1 (no owner label) is swept only once it is older than legacy_stale_s.
    max_lifetime_s: int = 7200
    legacy_stale_s: int = 4 * 3600
    # H1 P2: writable folders outside /work that outlive an exec; emptied after every step
    # (all but what the fresh container had), so nothing can be staged there between steps.
    scratch_dirs: tuple[str, ...] = ("/tmp", "/var/tmp", "/run", "/dev/shm", "/dev/mqueue")
    # H1 decision 3 / audit G-4: every secret-named entry in work/ at container start is masked
    # inside the container (a file: read-only bind of a notice; a folder: empty read-only tmpfs),
    # so bash can never read it. The host-side read tool (with its own rules) is the only route.
    secret_masks: bool = True
    # H1-I item 5: Track G's shared list (localbot.guard.secret_names), the same one the core4
    # guard indexes with (the loop builds its SecretNames from these two fields). Templates
    # (.env.example, *.pub ...) are not secrets, so not masked.
    secret_globs: tuple[str, ...] = SECRET_GLOBS
    secret_templates: tuple[str, ...] = TEMPLATE_GLOBS
    max_masks: int = 256  # .git + secret masks; more secret entries than fit = refuse to start
    # H1 S-4: work/ may grow by at most this much since the shell's first start (bytes of
    # regular files, entries of any kind). Past it the running command is stopped (polled every
    # quota_poll_s) and the task ends before the next undo snapshot copies the flood.
    work_growth_max_bytes: int = 2 * 1024**3
    work_growth_max_entries: int = 50_000
    quota_poll_s: float = 2.0
    run_flags: tuple[str, ...] = (
        "--network", "none",
        "--read-only",
        "--tmpfs", "/tmp",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", "256",
        "--memory", "2g",
        "--memory-swap", "2g",  # S7b.0 (owner decision): memory + swap = 2 GiB, i.e. no swap
        "--cpus", "2",
        "--user", "1000:1000",
    )
    workdir: str = "/work"
    exec_timeout_s: int = 30
    output_cap_chars: int = 4000


def user_state_dir(environ: Mapping[str, str] | None = None, platform: str | None = None) -> Path:
    """Where Local Bot keeps its own state (H1): ``LOCALBOT_STATE_DIR`` if set, else the user's
    state folder — Windows ``%LOCALAPPDATA%\\localbot``, macOS ``~/Library/Application
    Support/localbot``, else ``$XDG_STATE_HOME/localbot`` (absolute only, per the XDG spec) or
    ``~/.local/state/localbot``. Never the source tree."""
    env = os.environ if environ is None else environ
    plat = sys.platform if platform is None else platform
    if env.get("LOCALBOT_STATE_DIR"):
        return Path(env["LOCALBOT_STATE_DIR"])
    if plat == "win32":
        return Path(env.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "localbot"
    if plat == "darwin":
        return Path.home() / "Library" / "Application Support" / "localbot"
    xdg = env.get("XDG_STATE_HOME") or ""
    # POSIX rules (posixpath), not the host OS's: this branch is the Linux/macOS answer.
    return (Path(xdg) if posixpath.isabs(xdg) else Path.home() / ".local" / "state") / "localbot"


@dataclass(frozen=True)
class UndoConfig:
    """Which tools snapshot is loop._MUTATING (bash, write, edit; the parity bash_exec too)."""

    enabled: bool = True
    store_dir: Path = field(default_factory=user_state_dir)  # outside the workspace; <store_dir>/<bot>/undo/
    skip_files_over_mb: float = 50  # S7b.1: a bigger file is not stored (warned, named on restore)
    store_cap_mb: float = 2048  # S7b.1: oldest snapshots pruned past this (the newest is always kept)
    git_backup_cap_mb: float = 100  # S8e.1: a task-start .git over this is watched but not restorable


@dataclass(frozen=True)
class PathsConfig:
    # The product's bot workspaces live in the user's state folder (H1). The rest is the exam's
    # and exists in a source checkout only.
    workspaces: Path = field(default_factory=lambda: user_state_dir() / "workspaces")
    trials: Path = REPO_ROOT / "workspaces" / "_trials"
    results: Path = REPO_ROOT / "results" / "stage1"
    lab_ref: Path = REPO_ROOT / "lab_ref"


@dataclass(frozen=True)
class Profile:
    name: str
    tools: ToolsConfig
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    watchdog: WatchdogConfig = field(default_factory=WatchdogConfig)
    container: ContainerConfig = field(default_factory=ContainerConfig)
    undo: UndoConfig = field(default_factory=UndoConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    # H1 P-1: "exam" = today's rules byte-for-byte (G4 comparability); "product" = paths, secrets,
    # destruction + undo, network only (container mode only; the guard refuses it for parity).
    rules: Literal["exam", "product"] = "exam"


# --- Profiles ----------------------------------------------------------------------------


def parity_lab6() -> Profile:
    """The lab loop on a real disk. Anything the lab did not do is off."""
    return Profile(
        name="parity_lab6",
        tools=ToolsConfig(schemas=tuple(TOOLS_LAB6), shell_mode="emulated"),
        provider=ProviderConfig(retries_on_connection_error=0),  # the lab retried a 500 only
        # Owner decision, 2026-10-26: parse + nudge off (the lab read native tool_calls only), but still
        # record text-call attempts so G3 shows how many calls the old loop missed.
        loop=LoopConfig(text_call_parse=False, parse_fail_nudge=False, record_text_call_attempts=True,
                        dialect_aliases=False, refuse_cut_replies=False, max_calls_per_turn=None,
                        mark_untrusted_output=False),
        context=ContextConfig(
            tool_output_cap_chars=None,
            evict_tool_results_over_chars=None,
            compaction=False,
        ),
        watchdog=WatchdogConfig(
            stuck_nudge_after=None,
            stuck_end_after=None,
            repetition_min_unique_line_ratio=None,
            task_wall_clock_s=None,
        ),
        # Undo stays on: the model never sees it, and every disk run should be recoverable.
    )


def core4() -> Profile:
    """The new default: four tools, real shell in a container, undo on."""
    return Profile(
        name="core4",
        tools=ToolsConfig(schemas=tuple(TOOLS_CORE4), shell_mode="container"),
        loop=LoopConfig(system_prompt=f"{SYSTEM_SLIM} {CORE4_SEARCH_HINT}"),
    )


def core4_product() -> Profile:
    """core4 with the product rules (H1 P-1): what people will really run. Same tools, shell,
    masks and undo; only the guard's rule set differs."""
    return dataclasses.replace(core4(), name="core4_product", rules="product")


def with_run_settings(profile: Profile, *, num_ctx: int | None = None, max_turns: int | None = None,
                      num_predict: int | None = None) -> Profile:
    """S1-X X3: the run settings the runner's flags give (``--num-ctx`` / ``--max-turns`` / ``--num-predict``)
    on top of a profile. No setting given = the very same profile object, so a default run stays byte-identical
    (G3 parity compares its first request with the lab's). ``num_predict`` is for every model of the run: it
    clears ``num_predict_by_model``. Raises ValueError (a plain message) for a value that is not a whole
    number above 0, or a reply budget not smaller than the context."""
    given = {"num_ctx": num_ctx, "max_turns": max_turns, "num_predict": num_predict}
    if all(v is None for v in given.values()):
        return profile
    for name, v in given.items():
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 1):
            raise ValueError(f"{name} must be a whole number above 0 (got {v!r})")
    prov = profile.provider
    if num_ctx is not None:
        prov = dataclasses.replace(prov, num_ctx=num_ctx)
    if num_predict is not None:
        prov = dataclasses.replace(prov, num_predict=num_predict, num_predict_by_model=())
    biggest = max([prov.num_predict, *(n for _m, n in prov.num_predict_by_model)])
    if biggest >= prov.num_ctx:
        raise ValueError(f"num_predict ({biggest}) must be smaller than num_ctx ({prov.num_ctx})")
    loop = profile.loop if max_turns is None else dataclasses.replace(profile.loop, max_turns=max_turns)
    return dataclasses.replace(profile, provider=prov, loop=loop)


PROFILES = {"parity_lab6": parity_lab6, "core4": core4, "core4_product": core4_product}


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name]()
    except KeyError:
        raise KeyError(f"unknown profile {name!r}; known: {sorted(PROFILES)}") from None
