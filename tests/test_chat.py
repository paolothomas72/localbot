"""BRIEF C3: ``localbot chat``, the first terminal chat the owner can use.

Everything outside the code under test is a fake: the keyboard, the screen, Ollama (the real provider behind a
scripted transport), the container (a scripted shell that records start / close), the container runtime check and
the PC's home folder (a temp folder). The real undo store, the real loop and the real work-folder rules run.
One test (the last section) needs a real Podman or Docker and carries the ``container`` marker by hand; the host
suite never runs it and never probes a runtime for it."""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from localbot import __main__ as cli
from localbot import chat
from localbot.config import ContainerConfig, ProviderConfig
from localbot.guard.undo import UndoError
from localbot.guard.workspace_fs import CaseSensitivityError, is_case_sensitive, new_work_folder
from tests.fakes import FakeShell, RoutedTransport, ScriptedTransport, call, provider, reply

REPO = Path(__file__).resolve().parents[1]
SECRET = "plum-orchard-7731-kestrel-fake"
TAGS = [{"name": "qwen3:8b", "size": 5_200_000_000}, {"name": "llama3.1:8b", "size": 4_900_000_000}]
SEED = {"notes.txt": "N\n", "keep.txt": "keep\n"}


# --- the fakes --------------------------------------------------------------------------------------------


class Keys:
    """The keyboard: the next line each time it is asked; None (end of input) when the lines run out. A line that is
    an exception is raised instead (Ctrl+C)."""

    def __init__(self, lines):
        self.lines, self.prompts = list(lines), []

    def __call__(self, prompt: str):
        self.prompts.append(prompt)
        if not self.lines:
            return None
        nxt = self.lines.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt


class FakeContainer(FakeShell):
    """A scripted shell that also records its life in a shared log."""

    def __init__(self, script, log, made):
        super().__init__(script)
        self.log = log
        made.append(self)

    def start(self):
        self.log.append("start")

    def close(self):
        self.log.append("close")


class World:
    """One chat run's surroundings. ``run(*args)`` is ``localbot chat`` with those arguments."""

    def __init__(self, tmp_path, keys, replies=None, chat_transport=None, tags=TAGS, seed=None, shell_script=None,
                 runtime="podman"):
        self.home = tmp_path / "home"
        self.home.mkdir(exist_ok=True)
        self.out = io.StringIO()
        self.keys = Keys(keys)
        self.log, self.shells, self.work_made = [], [], []
        self.transport = chat_transport or ScriptedTransport(replies or [])
        self.routed = RoutedTransport(self.transport, tags=tags)
        self.seed = SEED if seed is None else seed
        self.shell_script = shell_script or {}
        self.restores = []
        self.deps = chat.Deps(
            home=self.home,
            runtime=lambda: runtime,
            provider=lambda cfg: provider(self.routed, cfg),
            shell=self._shell,
            sweep=lambda cfg: self.log.append("sweep"),
            new_work=self._new_work,
            make_store=chat.default_make_store,
            quiet=self._quiet,
            commit=lambda: "abc1234",
            read_line=self.keys,
            out=self.out,
            now=lambda: datetime(2026, 11, 5, 12, 30, 15, tzinfo=timezone.utc),
        )

    # the work folder: made on first run only, seeded with the starting files
    def _new_work(self, path: Path) -> Path:
        self.work_made.append(path)
        path.mkdir(parents=True)
        for rel, text in self.seed.items():
            (path / rel).parent.mkdir(parents=True, exist_ok=True)
            (path / rel).write_text(text, encoding="utf-8", newline="\n")
        return path

    def _shell(self, cfg, work):
        assert isinstance(cfg, ContainerConfig)
        return FakeContainer(self.shell_script_for(work), self.log, self.shells)

    def shell_script_for(self, work: Path):
        def rm(*names):
            def run():
                for n in names:
                    (work / n).unlink()
                return ""
            return run

        def overwrite():
            (work / "notes.txt").write_text("CHANGED\n", encoding="utf-8", newline="\n")
            return ""
        base = {"rm notes.txt": rm("notes.txt"), "rm .env": rm(".env"), "overwrite": overwrite, "ls": "keep.txt\nnotes.txt"}
        base.update(self.shell_script)
        return base

    def _quiet(self) -> bool:
        self.log.append("quiet")
        return True

    @property
    def bot(self) -> Path:
        return self.home / ".localbot" / "bots" / "main"

    @property
    def work(self) -> Path:
        return self.bot / "work"

    def run(self, *args) -> int:
        bot, model = "main", None
        it = iter(args)
        for a in it:  # the two options of ``localbot chat`` (the real parser is tested in the command-line test)
            if a == "--bot":
                bot = next(it)
            elif a == "--model":
                model = next(it)
        return chat.chat_main(bot, model, self.deps)

    @property
    def text(self) -> str:
        return self.out.getvalue()

    def records(self):
        """Every session record (one per model picked), as a list of parsed lines."""
        out = []
        for f in sorted((self.bot / "sessions").glob("*.jsonl")):
            out.append([json.loads(line) for line in f.read_text(encoding="utf-8").splitlines()])
        return out

    def record_text(self) -> str:
        return "".join(f.read_text(encoding="utf-8") for f in sorted((self.bot / "sessions").glob("*.jsonl")))

    def events(self, name=None):
        evs = [r["event"] for rec in self.records() for r in rec if r["kind"] == "event"]
        return [e for e in evs if name is None or e["event"] == name]


def _qwen_first():
    return "1"  # the picker lists qwen3:8b first


def _used_model(w, i=0):
    return w.transport.bodies[i]["model"]


# --- 1. no container runtime ------------------------------------------------------------------------------


def test_c3_no_container_runtime_is_a_plain_refusal_and_no_task_starts(tmp_path, monkeypatch):
    w = World(tmp_path, ["1", "hello"], runtime=None)
    started = []
    monkeypatch.setattr(chat, "run_task", lambda *a, **k: started.append(1))
    code = w.run()
    assert code != 0 and not started and not w.shells and not w.transport.bodies
    assert "Podman or Docker" in w.text and "Nothing was started" in w.text
    assert not (w.home / ".localbot").exists()  # not even a folder was made
    assert w.keys.prompts == []  # and nobody was asked anything


# --- 2. an undo store is always attached; the container shell is the only shell ---------------------------


def test_c3_no_undo_store_refuses_to_start(tmp_path, monkeypatch):
    w = World(tmp_path, ["1", "hello"], replies=[reply("hi")])
    w.deps.make_store = lambda cfg, bot, work: None
    started = []
    monkeypatch.setattr(chat, "run_task", lambda *a, **k: started.append(1))
    code = w.run()
    assert code != 0 and not started and not w.shells and not w.transport.bodies
    assert "undo" in w.text.lower() and "refus" in w.text.lower()


def test_c3_an_undo_store_that_cannot_be_made_refuses_to_start(tmp_path, monkeypatch):
    w = World(tmp_path, ["1", "hello"], replies=[reply("hi")])

    def boom(cfg, bot, work):
        raise UndoError("the undo store must be outside the workspace")
    w.deps.make_store = boom
    started = []
    monkeypatch.setattr(chat, "run_task", lambda *a, **k: started.append(1))
    assert w.run() != 0 and not started and not w.shells
    assert "undo" in w.text.lower()


def test_c3_the_loop_gets_core4_product_the_container_shell_a_store_and_the_hooks_never_the_emulated_shell(
        tmp_path, monkeypatch):
    w = World(tmp_path, ["1", "hello", "/quit"], replies=[reply("hi")])
    seen = {}
    real = chat.run_task

    def spy(profile, prov, model, user, ws, **kw):
        seen.update(profile=profile, model=model, user=user, kw=kw)
        return real(profile, prov, model, user, ws, **kw)
    monkeypatch.setattr(chat, "run_task", spy)
    assert w.run() == 0
    p, kw = seen["profile"], seen["kw"]
    assert p.name == "core4_product" and p.rules == "product" and p.tools.shell_mode == "container"
    assert kw["shell"] is w.shells[0] and kw["undo"] is not None and kw["undo"].root.name == "undo"
    assert callable(kw["next_user"]) and callable(kw["on_event"]) and callable(kw["ask_delete"])
    assert not kw.get("allow_delete") and seen["user"] == "hello" and seen["model"] == "qwen3:8b"
    # a chat has no whole-conversation clock (the person's thinking time and a long chat must not end it)
    assert p.watchdog.task_wall_clock_s is None and p.loop.trial_deadline_s >= 24 * 3600


def test_c3_the_bot_folder_layout_and_undo_store_location(tmp_path):
    w = World(tmp_path, ["1", "hello", "/quit"], replies=[reply("hi")])
    assert w.run() == 0
    assert w.work.is_dir() and (w.bot / "sessions").is_dir()
    assert (w.bot / "undo" / "meta.json").is_file()  # store_dir = ...\bots, bot = main  ->  bots\main\undo
    assert w.work_made == [w.work]


# --- 3. models --------------------------------------------------------------------------------------------


def test_c3_models_are_listed_numbered_and_picked_by_number(tmp_path):
    w = World(tmp_path, ["2", "hello", "/quit"], replies=[reply("hi")])
    assert w.run() == 0
    assert "1. qwen3:8b" in w.text and "2. llama3.1:8b" in w.text
    assert _used_model(w) == "llama3.1:8b"


def test_c3_model_flag_skips_the_list_when_the_model_is_installed(tmp_path):
    w = World(tmp_path, ["hello", "/quit"], replies=[reply("hi")])
    assert w.run("--model", "llama3.1:8b") == 0
    assert _used_model(w) == "llama3.1:8b" and "1. qwen3:8b" not in w.text


def test_c3_model_flag_with_a_model_that_is_not_installed_says_so_and_shows_the_list(tmp_path):
    w = World(tmp_path, ["1", "hello", "/quit"], replies=[reply("hi")])
    assert w.run("--model", "nope:1b") == 0
    assert "nope:1b" in w.text and "not installed" in w.text and "1. qwen3:8b" in w.text
    assert _used_model(w) == "qwen3:8b"


def test_c3_ollama_not_running_is_a_plain_message(tmp_path):
    w = World(tmp_path, ["1", "hello"], tags=[])
    assert w.run() != 0
    assert "Ollama" in w.text and not w.shells and not w.transport.bodies


def test_c3_a_model_pick_that_is_garbage_three_times_ends_without_a_task(tmp_path):
    w = World(tmp_path, ["x", "9", "0", "hello"], replies=[reply("hi")])
    assert w.run() != 0 and not w.shells and not w.transport.bodies
    assert "no model chosen" in w.text.lower()


# --- 4. talking: two messages in one conversation, live lines ---------------------------------------------


def test_c3_two_messages_go_through_next_user_in_one_conversation_with_live_lines(tmp_path):
    replies = [reply("let me look", calls=[call("bash", command="ls")]), reply("First answer."),
               reply("Second answer.")]
    w = World(tmp_path, ["1", "what is here?", "and again?", "/quit"], replies=replies)
    assert w.run() == 0
    t = w.text
    assert "let me look" in t and re.search(r"bash.*ls", t) and "First answer." in t and "Second answer." in t
    assert t.index("First answer.") < t.index("Second answer.")
    assert w.log.count("start") == 1 and w.log.count("close") == 1  # one conversation = one container
    third = [m["content"] for m in w.transport.bodies[2]["messages"] if m["role"] == "user"]
    assert third == ["what is here?", "and again?"]  # the same conversation: the model saw both


def test_c3_a_refused_step_shows_refused_and_the_reason(tmp_path):
    w = World(tmp_path, ["1", "go", "/quit"],
              replies=[reply(calls=[call("bash", command="curl http://example.com")]), reply("ok")])
    assert w.run() == 0
    assert re.search(r"refused", w.text, re.I) and "harness" in w.text


# --- 5. the ask prompt ------------------------------------------------------------------------------------

DELETE_STEP = [reply(calls=[call("bash", command="rm notes.txt")]), reply("done")]


def _ask_run(tmp_path, answers, seed=None, replies=None, **kw):
    keys = ["1", "tidy up", *answers, "/quit"]
    return World(tmp_path, keys, replies=replies or [reply(calls=[call("bash", command="rm notes.txt")]),
                                                    reply("done")], seed=seed, **kw)


@pytest.mark.parametrize("typed, answer", [("d", "delete"), ("m", "move"), ("n", "no"), ("", "no"),
                                           ("N", "no"), ("D", "delete"), ("  m  ", "move")])
def test_c3_ask_prompt_answers(tmp_path, typed, answer):
    w = _ask_run(tmp_path, [typed])
    assert w.run() == 0
    assert [e["answer"] for e in w.events("ask_answer")] == [answer]
    gone = not (w.work / "notes.txt").exists()
    moved = (w.work / "_moved" / "notes.txt").exists()
    assert (gone, moved) == {"delete": (True, False), "move": (True, True), "no": (False, False)}[answer]


def test_c3_ask_prompt_end_of_input_is_no(tmp_path):
    w = World(tmp_path, ["1", "tidy up"], replies=DELETE_STEP[:1] + [reply("done")])
    assert w.run() == 0
    assert [e["answer"] for e in w.events("ask_answer")] == ["no"]
    assert (w.work / "notes.txt").read_text(encoding="utf-8") == "N\n"


def test_c3_ask_prompt_ctrl_c_at_the_prompt_is_no_and_the_chat_goes_on(tmp_path):
    w = _ask_run(tmp_path, [KeyboardInterrupt()])
    assert w.run() == 0
    assert [e["answer"] for e in w.events("ask_answer")] == ["no"]
    assert (w.work / "notes.txt").exists()


def test_c3_ask_prompt_garbage_three_times_is_no(tmp_path):
    w = _ask_run(tmp_path, ["x", "yes", "delete it"])
    assert w.run() == 0
    assert [e["answer"] for e in w.events("ask_answer")] == ["no"]
    assert (w.work / "notes.txt").exists()
    assert sum(1 for p in w.keys.prompts if "[d]" in p) == 3 and w.keys.lines == []


def test_c3_ask_prompt_garbage_then_a_real_answer_uses_the_real_answer(tmp_path):
    w = _ask_run(tmp_path, ["what", "d"])
    assert w.run() == 0
    assert [e["answer"] for e in w.events("ask_answer")] == ["delete"]


def test_c3_ask_prompt_shows_the_why_word_and_the_name(tmp_path):
    w = _ask_run(tmp_path, ["n"])
    assert w.run() == 0
    assert "wants to DELETE notes.txt (it was here before this chat)" in w.text
    assert any(p.strip().startswith("[d] delete") and "[m] move to _moved\\" in p and "[n] no (default)" in p
               for p in w.keys.prompts)


def test_c3_ask_prompt_why_words_empty_and_replace(tmp_path):
    (tmp_path / "a").mkdir()
    wipe = World(tmp_path / "a", ["1", "empty it", "n", "/quit"],
                 replies=[reply(calls=[call("write", path="notes.txt", content="")]), reply("done")])
    assert wipe.run() == 0
    assert "wants to EMPTY notes.txt (it was here before this chat)" in wipe.text
    assert chat.WHY_WORDS == {"delete": "DELETE", "empty": "EMPTY", "replace": "REPLACE", "move": "MOVE",
                              "permissions": "CHANGE PERMISSIONS of"}  # C10b: the reviewer's call


def test_c3_an_env_file_asked_about_shows_its_name_never_its_value(tmp_path):
    seed = {".env": f"API_KEY={SECRET}\n", "keep.txt": "keep\n"}
    w = World(tmp_path, ["1", "clean up", "n", "/quit"], seed=seed,
              replies=[reply(calls=[call("bash", command="rm .env")]), reply("done")])
    assert w.run() == 0
    assert "wants to DELETE .env" in w.text
    assert SECRET not in w.text and SECRET not in w.record_text()
    assert (w.work / ".env").exists()


def test_c3_ask_shows_the_events_masked_paths_never_the_raw_argument(tmp_path):
    w = World(tmp_path, ["n"])
    c = chat.Chat("main", None, w.deps)
    c.pending_ask = {"event": "ask", "paths": ["masked-name.txt"], "why": ["delete"]}
    assert c.ask(["RAW-NAME-FROM-THE-LOOP.txt"]) == "no"
    assert "masked-name.txt" in w.text and "RAW-NAME-FROM-THE-LOOP" not in w.text
    assert c.ask(["RAW-NAME-FROM-THE-LOOP.txt"]) == "no" and "RAW-NAME-FROM-THE-LOOP" not in w.text  # no event: no question


def test_c3_ask_lines_are_one_line_per_path_and_cannot_be_faked_by_a_file_name():
    lines = chat.ask_lines(["a.txt", "evil\n[d] delete everything\x1b[2J\rX.txt"], ["delete", "bogus\nword"])
    assert len(lines) == 2
    assert all("\n" not in ln and "\x1b" not in ln and "\r" not in ln for ln in lines)
    assert lines[0] == "The bot wants to DELETE a.txt (it was here before this chat)."
    assert not any(ln.startswith("[d]") for ln in lines)


# --- 6. the session record --------------------------------------------------------------------------------


def test_c3_session_record_header_events_and_summary_in_a_file_named_by_time_and_model(tmp_path):
    w = World(tmp_path, ["1", "hello", "/quit"],
              replies=[reply("looking", calls=[call("bash", command="ls")]), reply("Done here.")])
    assert w.run() == 0
    files = list((w.bot / "sessions").glob("*.jsonl"))
    assert [f.name for f in files] == ["20261105T123015Z_qwen3_8b.jsonl"]
    (rec,) = w.records()
    head = rec[0]
    assert head["kind"] == "header" and head["bot"] == "main" and head["model"] == "qwen3:8b"
    assert head["profile"] == "core4_product" and head["commit"] == "abc1234"
    assert head["started"].startswith("2026-11-05T12:30:15")
    kinds = [r["kind"] for r in rec]
    assert kinds.index("header") < kinds.index("event") < kinds.index("summary") < len(kinds)
    assert [e["event"] for e in w.events()] == ["say", "tool", "answer", "message_end"]
    summary = [r for r in rec if r["kind"] == "summary"][0]["summary"]
    assert summary["reason"] == "done" and "tok_s" in summary
    assert rec[-1]["kind"] == "end" and rec[-1]["reason"] == "quit"
    assert str(files[0]) in w.text  # it says where the record is


def test_c3_session_record_is_written_as_it_goes(tmp_path):
    seen = {}
    state = {}

    def model(messages):
        n = len(messages)
        rec = sorted((state["w"].bot / "sessions").glob("*.jsonl"))
        seen[n] = [json.loads(x)["kind"] for x in rec[0].read_text(encoding="utf-8").splitlines()] if rec else None
        return reply("looking", calls=[call("bash", command="ls")]) if n <= 2 else reply("fine")
    from tests.fakes import ReactiveTransport
    w = World(tmp_path, ["1", "hello", "/quit"], chat_transport=ReactiveTransport(model))
    state["w"] = w
    assert w.run() == 0
    first, second = [seen[k] for k in sorted(seen)]
    assert first[0] == "header" and "event" not in first          # before any step: just the header (+ the user line)
    assert "event" in second and "summary" not in second           # mid-conversation: events are on disk already


def test_c3_a_secret_read_in_chat_is_in_neither_the_screen_nor_the_record(tmp_path):
    seed = {".env": f"API_KEY={SECRET}\n", "keep.txt": "keep\n"}
    w = World(tmp_path, ["1", f"my key is {SECRET}, read .env", "/quit"], seed=seed,
              replies=[reply(calls=[call("read", path=".env")]),
                       reply(f"The key is {SECRET} and API_KEY={SECRET}")])
    assert w.run() == 0
    assert SECRET not in w.text and SECRET not in w.record_text()
    assert "[REDACTED]" in w.text
    assert any(r["kind"] == "user" for r in w.records()[0])  # what the person typed is recorded, masked


def test_c3_the_record_has_one_file_per_model_and_per_conversation_summaries(tmp_path):
    w = World(tmp_path, ["1", "one", "/new", "two", "/model", "2", "three", "/quit"],
              replies=[reply("a"), reply("b"), reply("c")])
    assert w.run() == 0
    by_model = {rec[0]["model"]: rec for rec in w.records()}  # the files sort by name, not by time
    first, second = by_model["qwen3:8b"], by_model["llama3.1:8b"]
    assert [r["kind"] for r in first].count("summary") == 2
    assert [r["kind"] for r in second].count("summary") == 1
    assert len(list((w.bot / "sessions").glob("*.jsonl"))) == 2


# --- 7. slash commands ------------------------------------------------------------------------------------


def test_c3_undo_closes_the_container_restores_and_starts_again(tmp_path):
    replies = [reply(calls=[call("bash", command="overwrite")]), reply("changed it"), reply("hello again")]
    w = World(tmp_path, ["1", "change the notes", "/undo", "hi", "/quit"], replies=replies)
    assert w.run() == 0
    assert w.log == ["sweep", "start", "close", "quiet", "start", "close"]
    assert (w.work / "notes.txt").read_text(encoding="utf-8") == "N\n"
    assert "restored" in w.text.lower() and "hello again" in w.text


def test_c3_undo_with_a_number_and_with_nothing_to_undo_and_with_junk(tmp_path):
    w = World(tmp_path, ["1", "/undo 3", "/undo x", "/undo 0", "/quit"], replies=[])
    assert w.run() == 0
    t = w.text.lower()
    assert t.count("nothing was restored") == 3 and w.log.count("quiet") <= 1
    assert w.shells == []  # no message was ever sent: no container was ever started


def test_c3_undo_that_the_store_refuses_says_why_and_goes_on(tmp_path):
    w = World(tmp_path, ["1", "change", "/undo", "hi", "/quit"],
              replies=[reply(calls=[call("bash", command="overwrite")]), reply("ok"), reply("hello")])
    w.deps.quiet = lambda: False
    assert w.run() == 0
    assert "refusing to restore" in w.text and (w.work / "notes.txt").read_text(encoding="utf-8") == "CHANGED\n"


def test_c3_files_lists_the_bots_folder(tmp_path):
    w = World(tmp_path, ["1", "/files", "/quit"], seed={"a.txt": "A\n", "sub/b.txt": "B\n"})
    assert w.run() == 0
    assert "a.txt" in w.text and "sub/b.txt" in w.text


def test_c3b_files_cleans_file_names_a_direction_flip_and_an_escape_cannot_reach_the_screen(tmp_path):
    seed = {"a\u202eb.txt": "x\n", "plain.txt": "p\n"}          # U+202E flips the text direction of what follows it
    probe = tmp_path / "probe"
    probe.mkdir()
    try:
        (probe / "e\x1b[2Jf.txt").write_text("y", encoding="utf-8")
        seed["e\x1b[2Jf.txt"] = "y\n"                              # ESC is not allowed in Windows names
    except OSError:
        pass
    w = World(tmp_path, ["1", "/files", "/quit"], seed=seed)
    assert w.run() == 0
    assert "plain.txt" in w.text and "b.txt" in w.text
    assert "\u202e" not in w.text and "\x1b" not in w.text


def test_c3_new_starts_a_fresh_conversation_in_the_same_folder(tmp_path):
    w = World(tmp_path, ["1", "one", "/new", "two", "/quit"], replies=[reply("a"), reply("b")])
    assert w.run() == 0
    assert w.log.count("start") == 2 and w.log.count("close") == 2
    second = [m["content"] for m in w.transport.bodies[1]["messages"] if m["role"] == "user"]
    assert second == ["two"] and w.work_made == [w.work]


def test_c3_model_picks_again_and_starts_a_fresh_conversation(tmp_path):
    w = World(tmp_path, ["1", "one", "/model", "2", "two", "/quit"], replies=[reply("a"), reply("b")])
    assert w.run() == 0
    assert [_used_model(w, 0), _used_model(w, 1)] == ["qwen3:8b", "llama3.1:8b"]
    assert [m["content"] for m in w.transport.bodies[1]["messages"] if m["role"] == "user"] == ["two"]


def test_c3_stats_counts_the_session(tmp_path):
    fast = reply("ok", calls=[call("bash", command="ls")])
    fast["eval_count"], fast["eval_duration"] = 30, 1_000_000_000
    done = reply("done")
    done["eval_count"], done["eval_duration"] = 30, 1_000_000_000
    w = World(tmp_path, ["1", "look", "/stats", "/quit"], replies=[fast, done])
    assert w.run() == 0
    t = w.text
    assert re.search(r"messages you sent: 1\b", t) and re.search(r"tool steps: 1\b", t)
    assert re.search(r"refused or undone: 0\b", t) and re.search(r"asked about files: 0\b", t)
    assert "30.0 tokens/s" in t and re.search(r"session time: ", t)


def test_c3_stats_counts_refusals_and_asks_and_their_answers(tmp_path):
    w = _ask_run(tmp_path, ["m", "/stats"])
    assert w.run() == 0
    t = w.text
    assert re.search(r"asked about files: 1 \(delete 0, move 1, no 0\)", t)


def test_c3_help_lists_every_command_and_unknown_slash_commands_are_not_messages(tmp_path):
    w = World(tmp_path, ["1", "/help", "/bogus", "/quit"])
    assert w.run() == 0
    for cmd in ("/files", "/stats", "/new", "/model", "/undo", "/help", "/quit"):
        assert cmd in w.text
    assert "unknown command /bogus" in w.text and w.shells == [] and not w.transport.bodies


def test_c3_quit_and_end_of_input_close_cleanly_and_say_where_the_record_is(tmp_path):
    for i, keys in enumerate((["1", "hi", "/quit"], ["1", "hi"])):
        d = tmp_path / str(i)
        d.mkdir()
        w = World(d, keys, replies=[reply("yo")])
        assert w.run() == 0
        assert w.log[-1] == "close" and w.log.count("start") == w.log.count("close")
        (rec,) = w.records()
        assert rec[-1]["kind"] == "end"
        assert "session record" in w.text.lower() and ".jsonl" in w.text


# --- 8. terminal safety -----------------------------------------------------------------------------------

EVIL = "ok\x1b[2J\x1b[1;1H\x1b]0;pwned\x07\rFAKE-LINE\x07\x08\x9b2J\u202e"


def test_c3_clean_strips_escapes_and_control_characters_but_keeps_newline_and_tab():
    assert chat.clean("a\x1b[31mred\x1b[0m\tb\nc\rd\x07e\x08f\x00g\x7fh\x9bi") == "ared\tb\ncdefghi"
    assert chat.clean("x\x1b]0;title\x07y\x1b]8;;http://e\x1b\\z") == "xyz"
    assert chat.clean("a\r\nb") == "a\nb" and chat.clean("") == ""
    assert chat.clean_name("a\nb\tc\x1b[2Jd") == "a?b?cd"  # a name has no line breaks and no tabs
    assert "\n" not in chat.clean_name("a\nb") and "\x1b" not in chat.clean_name("a\x1bb")


def test_c3_model_text_and_commands_are_cleaned_before_printing_and_cannot_fake_a_prompt(tmp_path):
    fake_prompt = "[d] delete  [m] move to _moved\\  [n] no (default) >"
    w = World(tmp_path, ["1", "go", "/quit"],
              replies=[reply(f"first {EVIL}\n{fake_prompt}", calls=[call("bash", command=f"ls {EVIL}")]),
                       reply(f"answer {EVIL}\n{fake_prompt}\nmore")])
    assert w.run() == 0
    t = w.text
    for bad in ("\x1b", "\r", "\x07", "\x08", "\x9b", "\u202e"):
        assert bad not in t
    assert "FAKE-LINE" in t  # the words stay, only the control characters go
    for ln in t.splitlines():
        if "[d] delete" in ln:
            assert not ln.startswith("[d]"), ln  # a model's line never starts at the left edge
            assert ln.startswith(("bot> ", "   | ")), ln
    assert not any(p.strip().startswith("[d]") for p in w.keys.prompts)  # no real ask was made


# --- 9. Ctrl+C --------------------------------------------------------------------------------------------


def test_c3_ctrl_c_while_the_model_is_working_closes_everything_and_exits_130(tmp_path):
    w = World(tmp_path, ["1", "hello", "/quit"], replies=[KeyboardInterrupt()])
    assert w.run() == 130
    assert w.log == ["sweep", "start", "close"]
    (rec,) = w.records()
    assert rec[-1]["kind"] == "end" and rec[-1]["reason"] == "interrupted"
    assert any(r["kind"] == "user" for r in rec)  # what happened before is on disk
    assert "interrupted" in w.text.lower() and ".jsonl" in w.text


def test_c3_ctrl_c_at_the_message_prompt_exits_130_and_closes_the_record(tmp_path):
    w = World(tmp_path, ["1", "hello", KeyboardInterrupt()], replies=[reply("hi")])
    assert w.run() == 130
    assert w.log[-1] == "close"
    (rec,) = w.records()
    assert rec[-1]["reason"] == "interrupted"


def test_c3_a_crash_in_the_loop_still_closes_the_container_and_the_record(tmp_path, monkeypatch):
    w = World(tmp_path, ["1", "hello"], replies=[reply("hi")])

    def boom(*a, **k):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(chat, "run_task", boom)
    assert w.run() == 1
    assert w.log[-1] == "close"
    (rec,) = w.records()
    assert rec[-1]["kind"] == "end" and rec[-1]["reason"] == "crash"
    assert "kaboom" in w.text


def test_c3_a_container_that_will_not_start_is_a_plain_message(tmp_path):
    from localbot.tools.shell_container import ContainerError
    w = World(tmp_path, ["1", "hello"], replies=[reply("hi")])

    class Bad(FakeContainer):
        def start(self):
            raise ContainerError("podman run failed: no such image")
    w.deps.shell = lambda cfg, work: Bad({}, w.log, w.shells)
    assert w.run() == 1
    assert "no such image" in w.text and not w.transport.bodies
    assert w.log[-1] == "close"


# --- 10. the work folder ----------------------------------------------------------------------------------


def test_c3_the_work_folder_is_made_on_first_run_and_reused_after(tmp_path):
    w = World(tmp_path, ["1", "hi", "/quit"], replies=[reply("a")])
    assert w.run() == 0 and w.work_made == [w.work]
    (w.work / "mine.txt").write_text("keep me\n", encoding="utf-8")
    w2 = World(tmp_path, ["1", "hi", "/quit"], replies=[reply("b")])
    assert w2.run() == 0
    assert w2.work_made == [] and (w2.work / "mine.txt").read_text(encoding="utf-8") == "keep me\n"
    assert len(list((w2.bot / "sessions").glob("*.jsonl"))) == 2  # same second, same model: a second file name


def test_c3_a_work_folder_that_cannot_be_made_is_a_plain_message_and_nothing_starts(tmp_path):
    w = World(tmp_path, ["1", "hi"], replies=[reply("a")])

    def fail(path):
        raise CaseSensitivityError(f"{path}: the case-sensitive flag did not stick")
    w.deps.new_work = fail
    assert w.run() != 0
    assert "case-sensitive" in w.text and "work folder" in w.text.lower()
    assert not w.shells and not w.transport.bodies


def test_c3_the_real_work_folder_is_case_sensitive(tmp_path):
    try:
        new_work_folder(tmp_path / "probe")
    except Exception as exc:  # this PC cannot make case-sensitive folders: nothing to test here
        pytest.skip(f"cannot make a case-sensitive folder here: {type(exc).__name__}")
    w = World(tmp_path, ["1", "hi", "/quit"], replies=[reply("a")])
    w.deps.new_work = chat.default_new_work
    assert w.run() == 0 and is_case_sensitive(w.work)


# --- the command line -------------------------------------------------------------------------------------


def test_c3_the_command_line_parses_chat_and_leaves_undo_alone(monkeypatch):
    got = []
    monkeypatch.setattr("localbot.chat.chat_main", lambda bot, model, deps=None: got.append((bot, model)) or 0)
    assert cli.main(["chat"]) == 0 and cli.main(["chat", "--bot", "work-1", "--model", "m:1"]) == 0
    assert got == [("main", None), ("work-1", "m:1")]


@pytest.mark.parametrize("name", ["a b", "../x", "x/y", "x\\y", "", "con", "NUL", "a.b", "x" * 41, "é", "main\n"])
def test_c3_bad_bot_names_are_refused(tmp_path, name):
    w = World(tmp_path, ["1", "hi"], replies=[reply("a")])
    assert w.run("--bot", name) != 0
    assert "letters, digits" in w.text and not (w.home / ".localbot").exists() and not w.shells


@pytest.mark.parametrize("name", ["main", "work-1", "A_b", "x" * 40])
def test_c3_good_bot_names_are_accepted(name):
    assert chat.valid_bot_name(name)


def test_c3b_a_trailing_line_break_is_not_a_good_bot_name():
    # Python's $ also matches before a final newline; the name check must not.
    assert not chat.valid_bot_name("main\n") and not chat.valid_bot_name("main\r\n") and chat.valid_bot_name("main")


# --- G3: one chat per bot at a time -----------------------------------------------------------------------

HOLDER = ("import sys, time\nfrom pathlib import Path\nfrom localbot import chat\n"
          "lk = chat.acquire_bot_lock(Path(sys.argv[1]))\nprint('held', flush=True)\ntime.sleep(90)\n")


def _hold(lock_path: Path) -> subprocess.Popen:
    """Another PROCESS holding the bot's lock (one process can lock twice on some systems, so a thread would prove
    nothing)."""
    p = subprocess.Popen([sys.executable, "-X", "utf8", "-c", HOLDER, str(lock_path)], stdout=subprocess.PIPE,
                         text=True, cwd=REPO, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert p.stdout.readline().strip() == "held"
    return p


def _stop(p: subprocess.Popen) -> None:
    """Kill the holder and everything it started (a venv's python.exe may be a launcher with the real python as its
    child). Only ever called with our own Popen, before it is waited on, so the pid cannot have been reused."""
    import psutil
    try:
        procs = psutil.Process(p.pid).children(recursive=True) + [psutil.Process(p.pid)]
    except psutil.Error:
        procs = []
    for q in procs:
        try:
            q.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(procs, timeout=15)
    p.wait(timeout=30)
    p.stdout.close()


BUSY = ("Bot 'main' is already open in another window. Close that chat first, or use --bot with another name. "
        "Nothing was started.")


def test_c3b_a_second_window_on_the_same_bot_is_refused_and_nothing_is_made(tmp_path):
    w = World(tmp_path, ["1", "hello"], replies=[reply("hi")])
    holder = _hold(w.bot / "chat.lock")
    try:
        code = w.run()
    finally:
        _stop(holder)
    assert code == 2 and BUSY in w.text
    assert not w.shells and not w.work_made and not w.transport.bodies and w.log == [] and w.keys.prompts == []
    assert sorted(x.name for x in w.bot.iterdir()) == ["chat.lock"]  # no work folder, undo store or session record


def test_c3b_another_bot_name_is_not_blocked_and_the_lock_frees_when_the_holder_exits(tmp_path):
    w = World(tmp_path, ["1", "/quit"], replies=[])
    holder = _hold(w.bot / "chat.lock")
    try:
        assert w.run("--bot", "other") == 0                       # a chat on "main" never blocks one on "other"
        w2 = World(tmp_path, ["1", "/quit"])
        assert w2.run() == 2 and BUSY in w2.text
    finally:
        _stop(holder)
    w3 = World(tmp_path, ["1", "/quit"])                            # the OS dropped the lock with the process
    assert w3.run() == 0 and BUSY not in w3.text


def test_c3b_the_lock_is_held_for_the_whole_chat_and_let_go_at_every_exit(tmp_path, monkeypatch):
    lock = tmp_path / "home" / ".localbot" / "bots" / "main" / "chat.lock"
    w = World(tmp_path, ["1", "/files", "/quit"])
    seen = []
    keys = w.deps.read_line

    def spy(prompt):
        try:
            chat.acquire_bot_lock(lock).release()
            seen.append("free")
        except chat.BotBusy:
            seen.append("busy")
        return keys(prompt)
    w.deps.read_line = spy
    assert w.run() == 0
    assert seen and set(seen) == {"busy"}                          # held while the chat asks for lines
    chat.acquire_bot_lock(lock).release()                            # and free after a normal exit
    w = World(tmp_path, ["1", "hello"], replies=[reply("hi")])
    monkeypatch.setattr(chat, "run_task", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert w.run() == 1
    chat.acquire_bot_lock(lock).release()                            # and free after a crash
    w = World(tmp_path, ["1", KeyboardInterrupt()])
    assert w.run() == 130
    chat.acquire_bot_lock(lock).release()                            # and after Ctrl+C


def test_c3b_a_refused_start_does_not_leave_the_lock_behind(tmp_path):
    w = World(tmp_path, ["1", "hello"], runtime=None)
    assert w.run() != 0
    assert not (w.home / ".localbot").exists()                       # no runtime: not even a folder
    w = World(tmp_path, ["1", "hello"])
    w.deps.make_store = lambda cfg, bot, work: None
    assert w.run() != 0
    chat.acquire_bot_lock(w.bot / "chat.lock").release()


# --- 11. the install script -------------------------------------------------------------------------------

SCRIPT = REPO / "scripts" / "install_command.ps1"
PS = shutil.which("powershell") or shutil.which("pwsh")
needs_ps = pytest.mark.skipif(PS is None, reason="no PowerShell on this machine")


def _ps(command: str, **env) -> str:
    e = dict(os.environ, SCRIPT=str(SCRIPT), LOCALBOT_INSTALL_TEST="1", **env)
    r = subprocess.run([PS, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command",
                        ". $env:SCRIPT; " + command], capture_output=True, text=True, env=e, timeout=120)
    assert r.returncode == 0, r.stderr
    return r.stdout.rstrip("\r\n")


@needs_ps
def test_c3_install_path_function_adds_the_folder_once_and_keeps_the_rest():
    cmd = "Add-LocalbotPathEntry -Current $env:CUR -Folder $env:FOLDER"
    folder = r"D:\home\bob\.localbot\bin"
    assert _ps(cmd, CUR=r"C:\a;C:\Program Files\b;%USERPROFILE%\c", FOLDER=folder) == \
        r"C:\a;C:\Program Files\b;%USERPROFILE%\c;" + folder
    assert _ps(cmd, CUR=r"C:\a;" + folder, FOLDER=folder) == r"C:\a;" + folder             # run twice = one entry
    assert _ps(cmd, CUR=r"C:\a;" + folder.upper() + "\\;C:\\z", FOLDER=folder) == r"C:\a;" + folder.upper() + "\\;C:\\z"
    assert _ps(cmd, CUR="", FOLDER=folder) == folder
    assert _ps(cmd, CUR=r"C:\a;;", FOLDER=folder) == r"C:\a;" + folder                     # no empty entries made


@needs_ps
def test_c3b_the_cmd_text_sets_pythonpath_and_runs_python_with_dash_p():
    py, repo = r"D:\home\bob\localbot\.venv\Scripts\python.exe", r"D:\home\bob\localbot"
    text = _ps("Get-LocalbotCmdText -Python $env:PY -Repo $env:REPO", PY=py, REPO=repo)
    assert text.replace("\r", "").split("\n") == [
        "@echo off", "setlocal", f'set "PYTHONPATH={repo}"', f'"{py}" -P -m localbot %*']


@needs_ps
@pytest.mark.parametrize("bad", ['D:\\a%b\\repo', 'D:\\a"b\\repo'])
def test_c3b_a_repo_or_python_path_with_a_percent_or_a_quote_is_refused(bad):
    for which in ("repo", "python"):
        py = bad if which == "python" else r"D:\home\bob\py.exe"
        repo = bad if which == "repo" else r"D:\home\bob\localbot"
        r = subprocess.run([PS, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command",
                            ". $env:SCRIPT; Get-LocalbotCmdText -Python $env:PY -Repo $env:REPO"],
                           capture_output=True, text=True, timeout=120,
                           env=dict(os.environ, SCRIPT=str(SCRIPT), LOCALBOT_INSTALL_TEST="1", PY=py, REPO=repo))
        assert r.returncode != 0 and "a % or a double quote" in (r.stderr + r.stdout)


@needs_ps
def test_c3_install_writes_the_localbot_cmd_with_the_right_text(tmp_path):
    py, repo = r"D:\home\bob\localbot\.venv\Scripts\python.exe", r"D:\home\bob\localbot"
    out = tmp_path / "bin"
    _ps("Write-LocalbotCmd -BinDir $env:BIN -Python $env:PY -Repo $env:REPO", BIN=str(out), PY=py, REPO=repo)
    data = (out / "localbot.cmd").read_bytes()
    want = f'@echo off\r\nsetlocal\r\nset "PYTHONPATH={repo}"\r\n"{py}" -P -m localbot %*\r\n'
    assert data == want.encode("ascii") and not data.startswith(b"\xef\xbb\xbf")


@needs_ps
def test_c3b_the_import_check_finds_code_that_is_only_on_pythonpath_and_ignores_the_current_folder(tmp_path):
    repo, other, temp = tmp_path / "repo", tmp_path / "other", tmp_path / "temp"
    (repo / "lbtest_pkg_xyz").mkdir(parents=True)
    (repo / "lbtest_pkg_xyz" / "__init__.py").write_text("X = 1\n", encoding="utf-8")
    other.mkdir()
    temp.mkdir()
    (temp / "lbtest_pkg_xyz.py").write_text("X = 2\n", encoding="utf-8")      # sits in the folder the check runs from
    cmd = "Test-LocalbotImport -Python $env:PY -Repo $env:REPO -Module lbtest_pkg_xyz"
    env = dict(PY=sys.executable, TEMP=str(temp), TMP=str(temp))
    assert _ps(cmd, REPO=str(repo), **env) == "True"        # importable only through PYTHONPATH
    assert _ps(cmd, REPO=str(other), **env) == "False"      # the current folder (-P) is not a way in


def _regtest(initial, kind, folder, runs=1, key=r"HKCU:\Software\LocalBotInstallTest"):
    """Run Set-LocalbotUserPath on a temporary registry key; returns [raw Path value, its type]."""
    cmd = ("$k = $env:KEY; Remove-Item -LiteralPath $k -Recurse -ErrorAction SilentlyContinue; "
           "New-Item -Path $k | Out-Null; try { "
           "if ($env:INITIAL -ne '<none>') { New-ItemProperty -LiteralPath $k -Name Path -Value $env:INITIAL "
           "-PropertyType $env:KIND | Out-Null }; "
           "for ($i = 0; $i -lt [int]$env:RUNS; $i++) { Set-LocalbotUserPath -Folder $env:FOLDER -Key $k }; "
           "$it = Get-Item -LiteralPath $k; "
           "$it.GetValue('Path', '', 'DoNotExpandEnvironmentNames'); $it.GetValueKind('Path') "
           "} finally { Remove-Item -LiteralPath $k -Recurse -ErrorAction SilentlyContinue }")
    return _ps(cmd, KEY=key, INITIAL=initial, KIND=kind, FOLDER=folder, RUNS=str(runs)).splitlines()


@needs_ps
@pytest.mark.skipif(os.name != "nt", reason="the registry is Windows only")
def test_c3b_the_user_path_is_changed_without_expanding_it_or_changing_its_type():
    folder = r"D:\home\bob\.localbot\bin"
    for runs in (1, 2):                                      # run twice = one entry
        raw, kind = _regtest(r"%USERPROFILE%\x;C:\y", "ExpandString", folder, runs)
        assert raw == r"%USERPROFILE%\x;C:\y;" + folder and kind == "ExpandString"
    raw, kind = _regtest(r"C:\a;C:\y", "String", folder)       # a plain value stays a plain value
    assert raw == r"C:\a;C:\y;" + folder and kind == "String"
    raw, kind = _regtest("<none>", "ExpandString", folder)     # no value yet: made expandable
    assert raw == folder and kind == "ExpandString"
    raw, kind = _regtest(r"%USERPROFILE%\x;" + folder, "ExpandString", folder)   # already there: nothing changes
    assert raw == r"%USERPROFILE%\x;" + folder and kind == "ExpandString"


@needs_ps
@pytest.mark.skipif(os.name != "nt", reason="the registry is Windows only")
def test_c3b_the_installer_end_to_end_makes_a_command_that_runs_without_pip_from_any_folder(tmp_path):
    home, work = tmp_path / "home", tmp_path / "cwd"
    home.mkdir()
    work.mkdir()
    (work / "argparse.py").write_text("raise SystemExit(99)\n", encoding="utf-8")   # must never shadow real code
    key = r"HKCU:\Software\LocalBotInstallTestE2E"
    cmd = ("$k = $env:KEY; Remove-Item -LiteralPath $k -Recurse -ErrorAction SilentlyContinue; "
           "New-Item -Path $k | Out-Null; try { "
           "Install-LocalbotCommand -Python $env:PY -EnvKey $k "
           "} finally { Remove-Item -LiteralPath $k -Recurse -ErrorAction SilentlyContinue }")
    out = _ps(cmd, KEY=key, PY=sys.executable, USERPROFILE=str(home))
    assert "open a new PowerShell window, then type: localbot chat" in out and "pip" not in out.lower()
    cmd_file = home / ".localbot" / "bin" / "localbot.cmd"
    assert f'set "PYTHONPATH={REPO}"' in cmd_file.read_text(encoding="ascii")
    r = subprocess.run(["cmd", "/c", str(cmd_file), "chat", "--help"], capture_output=True, text=True, cwd=work,
                       timeout=120, env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    assert r.returncode == 0 and "usage" in r.stdout.lower(), (r.stdout, r.stderr)


@needs_ps
@pytest.mark.skipif(os.name != "nt", reason="the registry is Windows only")
def test_c3b_a_python_that_cannot_start_local_bot_stops_the_install_before_anything_is_written(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    fake = tmp_path / "fakepython.cmd"
    fake.write_text("@exit /b 1\r\n", encoding="ascii")        # exists, but cannot import anything
    key = r"HKCU:\Software\LocalBotInstallTestFail"
    cmd = ("$k = $env:KEY; Remove-Item -LiteralPath $k -Recurse -ErrorAction SilentlyContinue; "
           "New-Item -Path $k | Out-Null; try { "
           "try { Install-LocalbotCommand -Python $env:PY -EnvKey $k; 'went on' } catch { $_.Exception.Message }; "
           "'path value written: ' + (@((Get-Item -LiteralPath $k).GetValueNames()) -contains 'Path') "
           "} finally { Remove-Item -LiteralPath $k -Recurse -ErrorAction SilentlyContinue }")
    out = _ps(cmd, KEY=key, PY=str(fake), USERPROFILE=str(home))
    assert "cannot start Local Bot" in out and "went on" not in out and "pip" not in out.lower()
    assert "path value written: False" in out and not (home / ".localbot").exists()


@needs_ps
def test_c3b_the_installer_stops_with_a_plain_message_and_changes_nothing_when_python_cannot_start_local_bot(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    r = subprocess.run([PS, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
                        "-Python", str(tmp_path / "no-such-python.exe")], capture_output=True, text=True, timeout=120,
                       env=dict(os.environ, USERPROFILE=str(home), LOCALBOT_INSTALL_TEST=""))
    assert r.returncode != 0 and "pip" not in (r.stdout + r.stderr).lower()
    assert "no-such-python.exe" in r.stdout + r.stderr
    assert not (home / ".localbot").exists()


@needs_ps
def test_c3_sourcing_the_script_changes_nothing_and_the_script_names_only_the_bin_folder():
    assert _ps("'sourced'") == "sourced"
    src = SCRIPT.read_text(encoding="utf-8")
    assert "open a new PowerShell window, then type: localbot chat" in src
    # the PATH is edited as the raw registry value; never read expanded and never written back as plain text
    assert "SetEnvironmentVariable" not in src and "GetEnvironmentVariable" not in src
    assert "DoNotExpandEnvironmentNames" in src and "GetValueKind" in src and "SendMessageTimeout" in src
    assert "pip" not in src.lower().replace("pipeline", "") and " -P -m localbot" in src
    assert "Scripts" not in src.replace("\\Scripts\\python.exe", "")


# --- 12. the real container (the owner runs this; the host suite never does) -----------------------------------


def _runtime_or_skip(runtime: str) -> str:
    from tests import runtimes
    if not runtimes.is_up(runtime):
        pytest.skip(f"{runtime} is not up")
    return runtime


def _seeded(work: Path, seed: dict) -> Path:
    """Write the starting files into a folder that ``chat.default_new_work`` really made (it makes an EMPTY one)."""
    for rel, text in seed.items():
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        (work / rel).write_text(text, encoding="utf-8", newline="\n")
    return work


def test_c3d_seeded_makes_the_real_folder_and_writes_the_starting_files(tmp_path):
    """The helper the real-container tests use: the real make-folder step, then the seed files (parents made)."""
    try:
        new_work_folder(tmp_path / "probe")
    except Exception as exc:
        pytest.skip(f"cannot make a case-sensitive folder here: {type(exc).__name__}")
    work = _seeded(chat.default_new_work(tmp_path / "w"), {"a/b.txt": "x", ".env": "K=1\n"})
    assert work == tmp_path / "w"
    assert (work / "a" / "b.txt").read_text(encoding="utf-8") == "x"
    assert (work / ".env").read_bytes() == b"K=1\n"


def _real_container_chat(tmp_path, runtime, keys, replies, seed):
    """A chat on a REAL container shell (Podman or Docker); everything else is faked as in the other tests."""
    from localbot.tools.shell_container import ContainerShell
    _runtime_or_skip(runtime)
    try:
        new_work_folder(tmp_path / "probe")
    except Exception as exc:
        pytest.skip(f"cannot make a case-sensitive folder here: {type(exc).__name__}")
    w = World(tmp_path, keys, replies=replies, seed=seed)
    w.deps.runtime = lambda: runtime
    w.deps.sweep = lambda cfg: ContainerShell.sweep(cfg)
    w.deps.new_work = lambda path: _seeded(chat.default_new_work(path), seed)   # the real folder AND its seed files
    w.deps.quiet = lambda: True
    w.deps.shell = lambda cfg, work: ContainerShell(cfg, work)
    return w


@pytest.mark.container
@pytest.mark.parametrize("runtime", ["podman", "docker"])
def test_c3c_container_rm_env_in_a_real_chat_asks_then_the_mask_keeps_it(tmp_path, runtime):
    """``.env`` was there when the chat started, so the delete guard asks BEFORE ``rm .env`` runs (same as the
    host tests). Even when the person answers "d" (delete), the real container's read-only mask over ``.env``
    makes ``rm`` fail, so the host file keeps its exact bytes and its value is never shown."""
    seed = {".env": f"API_KEY={SECRET}\n", "keep.txt": "keep\n"}
    replies = [reply(calls=[call("bash", command="rm .env")]), reply("done")]
    w = _real_container_chat(tmp_path, runtime, ["1", "tidy up", "d", "/quit"], replies, seed)
    assert w.run() == 0, w.text
    assert (w.work / "keep.txt").read_text(encoding="utf-8") == "keep\n", "the chat did not run on the seeded folder"
    asks = w.events("ask")
    assert len(asks) == 1 and asks[0]["place"] == "precheck", asks          # asked once, before the command ran
    assert "wants to DELETE .env" in w.text
    rm = [e for e in w.events("tool") if "rm .env" in e["args"]]
    assert len(rm) == 1, "the person said delete, so the command must have run in the container"
    failed = rm[0]["result"].lower()
    assert "busy" in failed or "cannot remove" in failed or "read-only" in failed, rm[0]["result"]
    assert (w.work / ".env").read_bytes() == seed[".env"].encode("utf-8")     # exactly its seed bytes
    assert SECRET not in w.text and SECRET not in w.record_text()


@pytest.mark.container
@pytest.mark.parametrize("runtime", ["podman", "docker"])
def test_c3c_container_a_secret_file_already_in_moved_is_masked_in_a_real_container(tmp_path, runtime):
    """A secret-named file that an earlier chat moved to ``_moved/`` is masked like any other: a fresh container
    starts with every secret-named entry covered, ``_moved/.env`` included."""
    from localbot.tools.shell_container import MASK_NOTICE
    seed = {"_moved/.env": f"API_KEY={SECRET}\n", "keep.txt": "keep\n"}
    replies = [reply(calls=[call("bash", command="cat _moved/.env")]), reply("done")]
    w = _real_container_chat(tmp_path, runtime, ["1", "show it", "/quit"], replies, seed)
    assert w.run() == 0, w.text
    assert (w.work / "keep.txt").read_text(encoding="utf-8") == "keep\n", "the chat did not run on the seeded folder"
    assert (w.work / "_moved" / ".env").read_text(encoding="utf-8") == f"API_KEY={SECRET}\n", "no _moved/.env was seeded"
    tools = [e for e in w.events("tool") if "cat _moved/.env" in e["args"]]
    assert len(tools) == 1
    assert MASK_NOTICE.strip() in tools[0]["result"] and SECRET not in tools[0]["result"]
    assert SECRET not in w.text and SECRET not in w.record_text()
    assert (w.work / "_moved" / ".env").read_text(encoding="utf-8") == f"API_KEY={SECRET}\n"   # host file unchanged
