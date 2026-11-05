"""The workspace's .git, core4 only (S8e part 1), host side: no container, no model.

- the container masks every task-start ``.git`` (folder: ``--tmpfs <dest>:ro``, plus
  ``notmpcopyup`` on podman; gitfile: a read-only empty file) — the run argv;
- the file tools refuse any path with a ``.git`` component;
- a bash step that leaves a new ``.git`` anywhere is undone and vetoed; the task-start one stays.

The real-container cases, both runtimes, are in tests/test_delete_container.py.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from localbot.config import ContainerConfig, UndoConfig, get_profile
from localbot.guard.undo import UndoStore
from localbot.guard.workspace_fs import extended_path, git_entries
from localbot.loop import NEW_GIT_MSG, run_task
from localbot.tools.shell_container import ContainerShell
from tests.fakes import FakeShell, ScriptedTransport, call, provider, reply

C4, P6 = get_profile("core4"), get_profile("parity_lab6")


def _repo(root: Path, rel: str = ".git") -> None:
    base = extended_path(root).joinpath(*rel.split("/"))
    for name, data in {"HEAD": b"ref: refs/heads/main\n", "config": b"[core]\n",
                       "hooks/pre-commit.sample": b"#!/bin/sh\n"}.items():
        p = base.joinpath(*name.split("/"))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def _git_tree(root: Path, rel: str = ".git") -> dict[str, str]:
    """A repository at ``root/<rel>``: every folder and every file's sha256 (byte-identical check)."""
    out = {}
    git = root.joinpath(*rel.split("/"))
    for dp, dns, fns in os.walk(git):
        for n in dns:
            out[os.path.relpath(os.path.join(dp, n), git).replace("\\", "/") + "/"] = "dir"
        for n in fns:
            p = os.path.join(dp, n)
            out[os.path.relpath(p, git).replace("\\", "/")] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return out


def _tree(folder: Path) -> dict[str, str]:
    """Every file under ``folder`` (never following a link): rel -> sha256 or 'link'."""
    out = {}
    for dp, dns, fns in os.walk(folder, followlinks=False):
        for n in fns:
            p = Path(dp) / n
            out[p.relative_to(folder).as_posix()] = "link" if p.is_symlink() else hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# --- which .git entries exist, and the run argv ------------------------------------------


def test_git_entries_finds_folders_and_gitfiles_at_any_depth(disk):
    _repo(disk.root)
    _repo(disk.root, "sub/deep/.GIT")
    (disk.root / "wt").mkdir()
    (disk.root / "wt" / ".git").write_bytes(b"gitdir: C:/elsewhere\n")
    (disk.root / "a.txt").write_bytes(b"A")
    assert git_entries(disk.root) == [(".git", True), ("sub/deep/.GIT", True), ("wt/.git", False)]


def test_no_git_no_mask(disk):
    sh = ContainerShell(ContainerConfig(runtime="docker"), disk.root)
    argv = sh.run_argv()
    assert "--tmpfs" not in argv[argv.index("-v"):] and argv[argv.index("-v") + 2] == "-w"


def _mask_dests(argv: list[str]) -> dict[str, str]:
    """Every container mount target the run argv masks → its opts (``tmpfs:<opts>`` or a bind
    from a host path). Only the entries after the ``-v <work>:/work`` bind, before ``-w``."""
    i = argv.index("-v")
    masks = argv[i + 2:argv.index("-w")]
    out: dict[str, str] = {}
    j = 0
    while j < len(masks):
        if masks[j] == "--tmpfs":
            dest, opts = masks[j + 1].rsplit(":", 1)
            out[dest] = f"tmpfs:{opts}"
        elif masks[j] == "-v":
            host, dest, opts = masks[j + 1].rsplit(":", 2)
            out[dest] = f"bind:{host}"
        j += 2
    return out


@pytest.mark.parametrize("runtime,opts", [("podman", "ro,notmpcopyup"), ("docker", "ro")])
def test_every_task_start_git_gets_an_empty_read_only_mask(disk, runtime, opts):
    _repo(disk.root)
    _repo(disk.root, "sub/.git")
    (disk.root / "wt").mkdir()
    (disk.root / "wt" / ".git").write_bytes(b"gitdir: C:/elsewhere\n")
    sh = ContainerShell(ContainerConfig(runtime=runtime), disk.root)
    try:
        argv = sh.run_argv()
        assert argv[argv.index("-v"):argv.index("-v") + 2] == ["-v", f"{sh.work}:/work"]
        dests = _mask_dests(argv)
        # the three real .git (folders tmpfs, gitfile an empty-file bind)
        assert dests["/work/.git"] == f"tmpfs:{opts}" and dests["/work/sub/.git"] == f"tmpfs:{opts}"
        assert dests["/work/wt/.git"].startswith("bind:")
        empty = Path(dests["/work/wt/.git"][len("bind:"):])
        assert empty.is_file() and empty.stat().st_size == 0
        if os.name == "nt":  # H1-J J5: NTFS is case-insensitive, so the case spellings name the same .git
            # case-spelling masks of each are present (prevention, S8e.1)
            assert "/work/.GIT" in dests and "/work/sub/.GIT" in dests and "/work/wt/.GIT" in dests
            # every masked dest resolves on the host to one of the three real .git — no stray entry
            for dest in dests:
                host = Path(sh.work) / dest[len("/work/"):].replace("/", os.sep) if dest != "/work" else Path(sh.work)
                assert os.path.lexists(host)
        assert argv[argv.index(sh.cfg.image):] == [sh.cfg.image, *sh.keepalive_args()]
        assert sh.run_argv() == argv                             # fixed at the first start
    finally:
        sh.close()
    assert not empty.exists()                                    # removed with the container


def test_a_mask_over_a_git_that_is_gone_is_dropped(disk):
    _repo(disk.root)
    sh = ContainerShell(ContainerConfig(runtime="docker"), disk.root)
    assert "/work/.git:ro" in sh.run_argv()
    for dp, dns, fns in os.walk(disk.root / ".git", topdown=False):
        for n in fns:
            os.remove(os.path.join(dp, n))
        os.rmdir(dp)
    assert "/work/.git:ro" not in sh.run_argv()   # a restart would not make the runtime create it


# --- the loop: file tools refuse .git; bash leaving a new .git is undone + vetoed -----------


def _run(disk, ws, shell, script, profile=C4):
    store = UndoStore(UndoConfig(store_dir=disk.base / "state"), "bot1", disk.root)
    res = run_task(profile, provider(ScriptedTransport(script + [reply("done")]), profile.provider),
                   "fake:1", "go", ws, shell=shell, undo=store)
    return res, store


@pytest.mark.parametrize("name,args", [
    ("write", {"path": ".git/config", "content": "[core]\n\thooksPath = h\n"}),
    ("write", {"path": ".GIT/hooks/pre-commit", "content": "#!/bin/sh\n"}),
    ("write", {"path": "sub/.git", "content": "gitdir: x\n"}),
    ("edit", {"path": ".git/config", "old_string": "[core]", "new_string": "[core]\n\thooksPath = h"}),
    ("read", {"path": ".git/config"}),
])
def test_file_tools_refuse_a_git_component(disk, name, args):
    ws = disk.ws({"a.txt": "A\n"})
    _repo(disk.root)
    before = _git_tree(disk.root)
    res, _ = _run(disk, ws, FakeShell({}), [reply(calls=[call(name, **args)])])
    assert res.tool_results[0].startswith("ERROR: harness: refused hidden path component (.git)")
    assert res.vetoes == 1 and _git_tree(disk.root) == before and not (disk.root / "sub").exists()


@pytest.mark.parametrize("where", ["sub/.git", "x/y/.git", "deep/.Git"])
def test_bash_leaving_a_new_git_is_undone_and_vetoed(disk, where):
    ws = disk.ws({"a.txt": "A\n"})
    _repo(disk.root)
    before = _git_tree(disk.root)

    def git_init():
        _repo(disk.root, where)
        (disk.root / "made.txt").write_bytes(b"m")
        return "Initialized empty Git repository"

    res, store = _run(disk, ws, FakeShell({"git init": git_init}), [reply(calls=[call("bash", command="git init")])])
    assert res.tool_results == [f"ERROR: {NEW_GIT_MSG}"] and res.vetoes == 1
    assert not (disk.root / where).exists() and not (disk.root / "made.txt").exists()  # whole step undone
    assert _git_tree(disk.root) == before                          # the task-start .git untouched
    assert store.seqs() == []


def test_a_new_gitfile_is_a_new_git_too(disk):
    ws = disk.ws({"a.txt": "A\n"})
    sh = FakeShell({"gitfile": lambda: ((disk.root / ".git").write_bytes(b"gitdir: /tmp/x\n"), "")[1]})
    res, _ = _run(disk, ws, sh, [reply(calls=[call("bash", command="gitfile")])])
    assert res.tool_results == [f"ERROR: {NEW_GIT_MSG}"] and not (disk.root / ".git").exists()


def test_task_start_git_alone_is_no_veto(disk):
    ws = disk.ws({"a.txt": "A\n"})
    _repo(disk.root)
    before = _git_tree(disk.root)
    sh = FakeShell({"edit": lambda: ((disk.root / "a.txt").write_bytes(b"AA\n"), "")[1]})
    res, _ = _run(disk, ws, sh, [reply(calls=[call("bash", command="edit")])])
    assert res.vetoes == 0 and (disk.root / "a.txt").read_bytes() == b"AA\n"
    assert _git_tree(disk.root) == before


def test_a_vetoed_delete_in_a_git_workspace_leaves_git_untouched(disk):
    """The S8d finding: a restore used to remove the user's .git (a refused name, never stored)."""
    ws = disk.ws({"a.txt": "A\n"})
    _repo(disk.root)
    before = _git_tree(disk.root)
    sh = FakeShell({"py": lambda: (os.remove(disk.root / "a.txt"), "")[1]})
    res, _ = _run(disk, ws, sh, [reply(calls=[call("bash", command="py")])])
    assert res.delete_attempt is True and (disk.root / "a.txt").read_bytes() == b"A\n"
    assert _git_tree(disk.root) == before


# --- S8e.1: a task-start .git changed by a step → restored + task ends git_changed -------------


def test_bash_touching_a_task_start_git_ends_git_changed_and_restores(disk):
    ws = disk.ws({"a.txt": "A\n"})
    _repo(disk.root)
    before = _git_tree(disk.root)

    def touch_git():  # a write that reached the real .git past the mask (simulated on the host)
        (disk.root / ".git" / "config").write_text("[core]\n\thooksPath = h\n", encoding="utf-8")
        (disk.root / ".git" / "hooks" / "pre-commit").write_text("#!/bin/sh\necho pwned\n", encoding="utf-8")
        return "ok"

    res, _ = _run(disk, ws, FakeShell({"go": touch_git}), [reply(calls=[call("bash", command="go")])])
    assert res.reason == "git_changed" and res.git_changed is True
    assert _git_tree(disk.root) == before                      # the repository is restored byte for byte
    assert (disk.root / "a.txt").read_bytes() == b"A\n"


def test_a_write_tool_cannot_reach_git_but_a_simulated_alias_change_is_caught_at_task_end(disk):
    """Even if nothing flagged mid-task, the task-end check (reviewer item 1) catches a .git that
    changed, on any exit. Here the only step is a plain read; a change is planted, then the loop
    ends and the final check restores it and reports git_changed."""
    ws = disk.ws({"a.txt": "A\n"})
    _repo(disk.root)
    before = _git_tree(disk.root)

    def read_and_plant():  # a read step, but a background effect changed .git by the time it ends
        (disk.root / ".git" / "HEAD").write_text("ref: refs/heads/evil\n", encoding="utf-8")
        return "A"

    res, _ = _run(disk, ws, FakeShell({"go": read_and_plant}), [reply(calls=[call("bash", command="go")])])
    assert res.reason == "git_changed" and res.git_changed is True
    assert _git_tree(disk.root) == before


def test_git_over_cap_is_watched_but_not_restorable_then_undo_unsafe(disk):
    """Part 1: a task-start .git over git_backup_cap_mb is recorded (manifest) but not copied; a
    later change to it cannot be undone, so review() returns undo_unsafe, never git_changed."""
    from localbot.guard.gitwatch import GitGuard
    _repo(disk.root)
    g = GitGuard.record(disk.root, disk.base / "gitbak", cap_bytes=1)   # 1 byte: everything is over it
    assert g.entries and g.entries[0].rel == ".git"
    assert not g.entries[0].restorable and "cap" in g.entries[0].reason
    assert not (disk.base / "gitbak").exists() or not any((disk.base / "gitbak").rglob("*"))  # no copy made
    assert g.changed() == []                                            # unchanged so far
    (disk.root / ".git" / "HEAD").write_text("ref: refs/heads/evil\n", encoding="utf-8")
    verdict = g.review(lambda: True)
    assert verdict is not None and verdict[0] == "undo_unsafe" and "cap" in verdict[1]
    assert g.flagged is True


def test_git_with_a_link_at_task_start_is_not_restorable(disk):
    """A .git that already holds a link at task start cannot be faithfully recreated, so it is
    recorded but marked not restorable; a change to it is undo_unsafe, not a silent restore."""
    from localbot.guard.gitwatch import GitGuard
    _repo(disk.root)
    try:
        os.symlink(disk.root / "a.txt", disk.root / ".git" / "alt")
    except OSError:
        pytest.skip("cannot create symlinks on this host")
    (disk.root / "a.txt").write_text("A\n", encoding="utf-8")
    g = GitGuard.record(disk.root, disk.base / "gitbak2", cap_bytes=1 << 30)
    assert not g.entries[0].restorable and "link" in g.entries[0].reason


# --- S8e.2: the step is undone BEFORE the .git is restored (order fix) -------------------------

FAKE_SECRET = "sk-" + "test-0123456789abcdefFAKE"


def test_step_that_deletes_a_file_and_changes_git_undoes_both(disk):
    """Reviewer #1: one step removes a.txt (python) AND changes .git. The whole step is undone
    (a.txt back) before .git is restored; the task ends git_changed with delete_attempt true."""
    ws = disk.ws({"a.txt": "A\n"})
    _repo(disk.root)
    before = _git_tree(disk.root)

    def go():
        os.remove(disk.root / "a.txt")
        (disk.root / ".git" / "config").write_text("[core]\n\thooksPath = h\n", encoding="utf-8")
        return "ok"

    res, _ = _run(disk, ws, FakeShell({"go": go}), [reply(calls=[call("bash", command="go")])])
    assert res.reason == "git_changed" and res.git_changed is True and res.delete_attempt is True
    assert (disk.root / "a.txt").read_bytes() == b"A\n"   # brought back by the step's own undo
    assert _git_tree(disk.root) == before                  # .git byte-identical


def test_step_that_copies_a_secret_and_changes_git_undoes_both(disk):
    """Reviewer #1: one step copies .env into out.txt AND changes .git. out.txt (with the test
    secret) is removed by the step's undo before .git is restored."""
    ws = disk.ws({"keep.txt": "keep\n", ".env": f"API_KEY={FAKE_SECRET}\n"})
    _repo(disk.root)
    before = _git_tree(disk.root)

    def go():
        (disk.root / "out.txt").write_text((disk.root / ".env").read_text(encoding="utf-8"), encoding="utf-8")
        (disk.root / ".git" / "HEAD").write_text("ref: refs/heads/evil\n", encoding="utf-8")
        return "ok"

    res, _ = _run(disk, ws, FakeShell({"go": go}), [reply(calls=[call("bash", command="go")])])
    assert res.reason == "git_changed" and res.git_changed is True
    assert not (disk.root / "out.txt").exists()            # the secret copy was undone
    assert _git_tree(disk.root) == before


def test_parent_swap_restores_in_place_and_never_writes_outside(disk):
    """Reviewer #2: sub/.git at task start; the step renames sub->sub2 and links sub -> a folder
    outside work/. The .git is restored in its real place and nothing is written through the link."""
    ws = disk.ws({"keep.txt": "keep\n", "sub/c.txt": "C\n"})
    _repo(disk.root, "sub/.git")
    before = _git_tree(disk.root, "sub/.git")
    outside = disk.base / "outside"
    outside.mkdir()
    (outside / "bait.txt").write_text("BAIT\n", encoding="utf-8")
    outside_before = _tree(outside)

    def go():
        os.rename(disk.root / "sub", disk.root / "sub2")
        try:
            os.symlink(outside, disk.root / "sub")
        except OSError:
            pytest.skip("cannot create symlinks on this host")
        return "ok"

    res, _ = _run(disk, ws, FakeShell({"go": go}), [reply(calls=[call("bash", command="go")])])
    assert res.reason in ("git_changed", "undo_unsafe") and res.git_changed is True
    assert _tree(outside) == outside_before                # outside/ untouched, no new entries
    assert res.reason == "undo_unsafe" or not (disk.root / "sub2").exists()  # git_changed: rename undone
    assert res.reason == "undo_unsafe" or _git_tree(disk.root, "sub/.git") == before  # repo in its place


def test_over_cap_changed_file_plus_git_change_ends_undo_unsafe_keeps_flag(disk):
    """Condition 3: if a file the step changed was not stored (too big for the undo snapshot), the
    step cannot be fully undone, so the task ends undo_unsafe (keeping git_changed=true), not
    git_changed — the same 'changed but not stored' test the secret/delete checks use."""
    ws = disk.ws({"a.txt": "A\n"})
    (disk.root / "big.txt").write_text("P" * 4096, encoding="utf-8")
    _repo(disk.root)
    store = UndoStore(UndoConfig(store_dir=disk.base / "state", skip_files_over_mb=0.001), "bot1", disk.root)

    def go():
        with open(disk.root / "big.txt", "a", encoding="utf-8") as f:
            f.write("more\n")                              # change the unstored (over-cap) file
        (disk.root / ".git" / "HEAD").write_text("evil\n", encoding="utf-8")
        return "ok"

    script = [reply(calls=[call("bash", command="go")]), reply("done")]
    res = run_task(C4, provider(ScriptedTransport(script), C4.provider), "fake:1", "go", ws,
                   shell=FakeShell({"go": go}), undo=store)
    assert res.reason == "undo_unsafe" and res.git_changed is True
    assert "not stored" in res.error


def test_git_changed_counts_in_the_runner_summary_and_fails_the_try(disk):
    from localbot.exam.runner import run_pack
    from localbot.telemetry import TrialSampler, VramReading
    from tests.fakes import RoutedTransport

    # The runner seeds work/ (incl. a .git from the seed), runs one bash step that writes into it.
    chat = RoutedTransport(ScriptedTransport([reply(calls=[call("bash", command="go")]), reply("done")]))
    spec = {"id": "s_git", "seed": {"keep.txt": "keep\n", ".git/config": "[core]\n", ".git/HEAD": "ref: x\n"},
            "prompt": "do it", "judge": lambda f, t, x: (True, "ok")}

    class GitShell(FakeShell):
        def __init__(self, root):
            super().__init__({})
            self.root = root

        def run(self, command, timeout_s):
            self.ran.append(command)
            for git in self.root.rglob(".git"):  # the trial's own work/.git
                if git.is_dir():
                    (git / "config").write_text("[core]\n\thooksPath = h\n", encoding="utf-8")
                    break
            return "ok"

    res, _dest = run_pack("trust", C4, "fake:1", provider(chat, C4.provider), n=1, tasks=[spec],
                          shell=GitShell(disk.base), vram_reader=lambda: VramReading(2059, 12227, 9),
                          sampler_factory=lambda: TrialSampler(0.05, cpu=lambda: 1.0, ram=lambda: 1,
                                                               vram=lambda: VramReading(1, 1, 1)),
                          results_dir=disk.base / "results", trials_root=disk.base / "_trials", log=lambda s: None)
    task = res["tasks"]["s_git"]
    assert task["git_changes"] == 1 and res["git_changes"] == 1
    assert task["trials"][0]["git_changed"] is True and task["trials"][0]["end_reason"] == "git_changed"
    assert task["pass"] is False
