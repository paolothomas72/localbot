"""config.py: profiles exist, and parity values still match the frozen lab source.

The lab files are read with ``ast`` (never imported or executed), so lab_ref/ stays untouched.
"""
import ast
from pathlib import Path

import pytest

from localbot import config

LAB = Path(__file__).resolve().parents[1] / "lab_ref"


def _lab_constants(rel: str) -> dict:
    """Module-level NAME = <literal> assignments from a lab file, without running it."""
    tree = ast.parse((LAB / rel).read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def _lab_kwdefaults(rel: str, func: str) -> dict:
    tree = ast.parse((LAB / rel).read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func)
    names = [a.arg for a in fn.args.kwonlyargs]
    return {n: ast.literal_eval(d) for n, d in zip(names, fn.args.kw_defaults) if d is not None}


def test_system_slim_matches_lab():
    assert config.SYSTEM_SLIM == _lab_constants("scripts/run_lh_product.py")["SYSTEM_SLIM"]


def test_lab6_tools_match_lab():
    assert config.TOOLS_LAB6 == _lab_constants("scripts/run_all_fit.py")["TOOLS"]


def test_parity_loop_values_match_lab():
    lab = _lab_constants("scripts/run_lh_product.py")
    p = config.get_profile("parity_lab6")
    assert p.loop.max_turns == lab["MAX_TURNS"]
    assert p.loop.task_timeout_s == lab["TASK_TIMEOUT_S"]
    assert p.loop.trial_deadline_s == lab["DEADLINE_TRIAL_S"]
    assert p.loop.system_prompt == lab["SYSTEM_SLIM"]


def test_parity_provider_values_match_lab():
    chat = _lab_kwdefaults("lib/ollama_chat.py", "chat")
    p = config.get_profile("parity_lab6").provider
    assert p.temperature == chat["temperature"]
    assert p.num_predict == chat["num_predict"]
    assert p.keep_alive == chat["keep_alive"]
    assert p.num_ctx == _lab_constants("lib/fit_roster.py")["DEFAULT_CTX"]
    assert p.stream is False


def test_parity_turns_off_what_the_lab_never_did():
    p = config.get_profile("parity_lab6")
    assert p.tools.names == ("read", "write", "edit", "glob", "grep", "bash_exec")
    assert p.tools.shell_mode == "emulated"
    assert p.loop.text_call_parse is False and p.loop.parse_fail_nudge is False
    assert p.loop.record_text_call_attempts is True  # observe only (owner decision, 2026-10-26)
    assert p.context.compaction is False
    assert p.context.tool_output_cap_chars is None
    assert p.context.evict_tool_results_over_chars is None


def test_core4_defaults():
    p = config.get_profile("core4")
    assert p.tools.names == ("read", "write", "edit", "bash")
    assert p.tools.shell_mode == "container"
    assert p.undo.enabled  # which tools snapshot: loop._MUTATING (H1: before_tools was never read)
    assert p.context.compaction and (p.context.compaction_trigger, p.context.compaction_target) == (0.80, 0.60)
    assert p.context.tool_output_cap_chars == 4000
    assert p.container.runtime == "podman"
    assert ("--network", "none") == p.container.run_flags[0:2]


def test_core4_system_prompt_adds_search_hint():
    c4, p6 = config.get_profile("core4"), config.get_profile("parity_lab6")
    assert p6.loop.system_prompt == config.SYSTEM_SLIM                       # parity: unchanged
    assert c4.loop.system_prompt == f"{config.SYSTEM_SLIM} {config.CORE4_SEARCH_HINT}"


def test_unknown_profile():
    with pytest.raises(KeyError):
        config.get_profile("nope")


def test_no_personal_path_in_config():
    assert "Downloads" not in Path(config.__file__).read_text(encoding="utf-8")


# H1-J J7a: Track L's files carry no owner name and no personal or home path (publish scan). The
# words are built here so this file does not match itself.
_L_FILES = ["localbot/config.py", "localbot/loop.py", "localbot/context.py", "localbot/watchdog.py",
            "localbot/telemetry.py", "localbot/provider/ollama.py", "localbot/parse/dialects.py",
            "localbot/parse/fn_call.py", "localbot/parse/types.py", "localbot/exam/runner.py",
            "localbot/exam/packs/injection.json", "tests/test_config.py", "tests/test_loop.py",
            "tests/test_provider.py", "tests/test_fn_call.py", "tests/test_dialects.py", "tests/test_watchdog.py",
            "tests/test_context.py", "tests/test_tool_args.py", "tests/test_injection_pack.py",
            "tests/test_injection_runner.py", "tests/test_runner.py"]
_SCRUB = [("Pa" + "olo").lower(), "c:" + "\\users", "c:" + "/users", "/" + "home/", "kings" + "ky"]


@pytest.mark.parametrize("rel", _L_FILES)
def test_j7a_no_owner_name_or_home_path_in_track_l_files(rel):
    text = (LAB.parent / rel).read_text(encoding="utf-8").lower()
    hits = [w for w in _SCRUB if w in text]
    assert hits == [], f"{rel}: {hits}"


def test_h1_dead_knobs_are_gone():
    """H1 loop P2: knobs nothing reads only pretend to configure something. Removed:
    UndoConfig.before_tools (loop._MUTATING decides), UndoConfig.git_exe / LOCALBOT_GIT (undo has
    used file copies, not git, since S7b.1), ProviderConfig.require_full_gpu / unload_after_model
    (the runner always checks the lane and unloads), LoopConfig.majority_of (runner.majority is
    the lab's >= 2 of 3)."""
    p = config.get_profile("core4")
    for obj, name in [(p.undo, "before_tools"), (p.undo, "git_exe"), (p.provider, "require_full_gpu"),
                      (p.provider, "unload_after_model"), (p.loop, "majority_of")]:
        assert not hasattr(obj, name), f"{type(obj).__name__}.{name} is still there"
    assert "LOCALBOT_GIT" not in Path(config.__file__).read_text(encoding="utf-8")


def test_state_dir_is_configurable(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALBOT_STATE_DIR", str(tmp_path))
    assert config.get_profile("core4").undo.store_dir == tmp_path


@pytest.mark.parametrize("platform,env,expected", [
    ("win32", {"LOCALAPPDATA": r"C:\U\x\AppData\Local"}, Path(r"C:\U\x\AppData\Local") / "localbot"),
    ("linux", {"XDG_STATE_HOME": "/srv/state"}, Path("/srv/state") / "localbot"),
    # H1-J (audit): absolute by POSIX rules, whatever OS runs the test ("C:\x" is relative on Linux)
    ("linux", {"XDG_STATE_HOME": "C:\\x"}, Path.home() / ".local" / "state" / "localbot"),
    ("linux", {}, Path.home() / ".local" / "state" / "localbot"),
    ("linux", {"XDG_STATE_HOME": "relative/not/allowed"}, Path.home() / ".local" / "state" / "localbot"),
    ("darwin", {}, Path.home() / "Library" / "Application Support" / "localbot"),
])
def test_h1_state_dir_default_is_the_users_state_dir(platform, env, expected):
    """H1 (Track P's cross-track request): the default state dir (undo snapshots, bot state) is
    the user's own state folder, not ``<source tree>/state``: an installed package has no repo."""
    assert config.user_state_dir(env, platform) == expected
    assert config.user_state_dir({"LOCALBOT_STATE_DIR": "/s", **env}, platform) == Path("/s")


def test_h1_defaults_point_outside_the_source_tree(monkeypatch, tmp_path):
    monkeypatch.delenv("LOCALBOT_STATE_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "la"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    p = config.get_profile("core4")
    for path in (p.undo.store_dir, p.paths.workspaces):
        assert config.REPO_ROOT not in Path(path).parents and Path(path) != config.REPO_ROOT / "state"
    assert p.paths.workspaces == p.undo.store_dir / "workspaces"
    # exam-only paths stay in the source checkout (they hold the lab code and the evidence)
    assert p.paths.lab_ref == config.REPO_ROOT / "lab_ref" and p.paths.results == config.REPO_ROOT / "results" / "stage1"


@pytest.mark.parametrize("url", ["http://10.0.0.5:11434", "https://example.com"])
def test_ollama_url_must_be_localhost(monkeypatch, url):
    monkeypatch.setenv("LOCALBOT_OLLAMA_URL", url)
    with pytest.raises(ValueError):
        config.get_profile("core4")


# --- H1-I item 6: the rules knob and the core4_product profile ----------------------------------


def test_h1i_rules_knob_and_core4_product_profile():
    import dataclasses
    from localbot.config import get_profile
    assert get_profile("parity_lab6").rules == "exam" and get_profile("core4").rules == "exam"
    prod = get_profile("core4_product")
    assert prod.rules == "product" and prod.name == "core4_product"
    assert dataclasses.replace(prod, name="core4", rules="exam") == get_profile("core4")
