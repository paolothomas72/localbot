"""H1-K K6 D1 (owner decision): a case-sensitive work folder per bot / trial.

A mask sits on one spelling of a path. On case-insensitive NTFS the container reaches a masked
secret through another spelling (``CONFIG/.env.local``, ``.eNv.local``: test_shell_container's
case-spelling hole). Windows can make one folder case-sensitive (``fsutil file setCaseSensitiveInfo
<dir> enable``; here the same call through the API, so a failure is an error code, not a localized
message: fsutil exits 0 even when it fails). It only works on an EMPTY folder, and folders made
inside inherit it, so a work folder is made case-sensitive right after it is created, before the
first file. A container never starts on a folder that isn't (fail closed). Linux is case-sensitive
unless the file system folds case: a read-only probe checks (R9). Host tests only; the container
test is the owner's to run, under the lock.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from localbot.guard import workspace_fs as wfs

windows = pytest.mark.skipif(os.name != "nt", reason="per-folder case sensitivity is a Windows (NTFS) setting")


@windows
def test_an_empty_folder_becomes_case_sensitive_and_its_new_subfolders_inherit_it(tmp_path):
    d = tmp_path / "work"
    d.mkdir()
    assert wfs.is_case_sensitive(d) is False  # a plain folder on this PC
    wfs.set_case_sensitive(d)
    assert wfs.is_case_sensitive(d) is True
    (d / "sub").mkdir()
    assert wfs.is_case_sensitive(d / "sub") is True
    (d / "sub" / "A.txt").write_text("upper", encoding="utf-8")
    (d / "sub" / "a.txt").write_text("lower", encoding="utf-8")
    assert sorted(p.name for p in (d / "sub").iterdir()) == ["A.txt", "a.txt"]  # two files now


@windows
def test_a_folder_that_already_holds_a_file_is_refused_with_a_clear_message(tmp_path):
    d = tmp_path / "work"
    d.mkdir()
    (d / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(wfs.CaseSensitivityError, match="not empty") as e:
        wfs.set_case_sensitive(d)
    assert str(d) in str(e.value) and wfs.is_case_sensitive(d) is False


@windows
def test_new_work_folder_sets_the_flag_before_the_first_file(tmp_path):
    d = wfs.new_work_folder(tmp_path / "trial" / "work")
    assert d.is_dir() and list(d.iterdir()) == [] and wfs.is_case_sensitive(d)
    with pytest.raises(FileExistsError):  # never turned on for a folder that might hold files
        wfs.new_work_folder(d)


@windows
def test_on_linux_a_read_only_probe_sees_a_case_insensitive_folder(tmp_path, monkeypatch):
    """Review R9: Linux was trusted blindly, but a vfat / ntfs3 / casefold / drvfs folder folds case.
    The probe looks the folder's own name up with its case swapped (a new folder inherits its
    parent's behaviour) and writes nothing. Run here on a plain NTFS folder (case-insensitive) with
    sys.platform patched: a Windows host, not a Linux result."""
    plain = tmp_path / "work"
    plain.mkdir()
    digits = tmp_path / "123"  # no letter to swap: an ancestor's name is probed
    digits.mkdir()
    monkeypatch.setattr(wfs.sys, "platform", "linux")
    monkeypatch.setattr(wfs, "_case_info", lambda *a, **k: pytest.fail("no Windows call on Linux"))
    assert wfs.is_case_sensitive(plain) is False and wfs.is_case_sensitive(digits) is False
    with pytest.raises(wfs.CaseSensitivityError, match="case-insensitive"):
        wfs.set_case_sensitive(plain)
    with pytest.raises(wfs.CaseSensitivityError):
        wfs.new_work_folder(tmp_path / "trial" / "work")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["123", "trial", "work"]  # nothing written but the
    assert list(plain.iterdir()) == [] and list((tmp_path / "trial").iterdir()) == [tmp_path / "trial" / "work"]


@windows
def test_on_linux_a_case_sensitive_folder_passes_the_probe(tmp_path, monkeypatch):
    parent = wfs.new_work_folder(tmp_path / "cs")  # the real Windows flag; children inherit it
    monkeypatch.setattr(wfs.sys, "platform", "linux")
    monkeypatch.setattr(wfs, "_case_info", lambda *a, **k: pytest.fail("no Windows call on Linux"))
    (parent / "work").mkdir()
    assert wfs.is_case_sensitive(parent / "work") is True
    d = wfs.new_work_folder(parent / "trial" / "work")
    assert wfs.is_case_sensitive(d) is True and list(d.iterdir()) == []


def test_another_os_is_not_trusted(tmp_path, monkeypatch):
    """macOS (unsupported) is case-insensitive by default: fail closed, don't assume."""
    monkeypatch.setattr(wfs.sys, "platform", "darwin")
    assert wfs.is_case_sensitive(tmp_path) is False
    with pytest.raises(wfs.CaseSensitivityError):
        wfs.set_case_sensitive(tmp_path)


def test_a_container_never_starts_on_a_folder_that_is_not_case_sensitive(tmp_path, monkeypatch):
    from localbot.config import ContainerConfig
    from localbot.tools.shell_container import CASE_SENSITIVE_MSG, ContainerError, ContainerShell

    work = tmp_path / "work"
    work.mkdir()
    ran = []
    monkeypatch.setattr("localbot.tools.shell_container._run", lambda argv, **kw: ran.append(argv))
    monkeypatch.setattr(wfs, "is_case_sensitive", lambda folder: False)
    sh = ContainerShell(ContainerConfig(), work)
    with pytest.raises(ContainerError, match="case-sensitive") as e:
        sh.start()
    assert ran == [] and str(work) in str(e.value)  # refused before any runtime command
    assert CASE_SENSITIVE_MSG.split("{")[0] in str(e.value)


# --- names in a case-sensitive work folder: twins are two files (review of D1, P0) ------------------


def _twin_folder(tmp_path) -> Path:
    d = wfs.new_work_folder(tmp_path / "work")
    (d / "a.txt").write_text("lower\n", encoding="utf-8", newline="\n")
    (d / "A.txt").write_text("UPPER\n", encoding="utf-8", newline="\n")
    assert sorted(p.name for p in d.iterdir()) == ["A.txt", "a.txt"]  # two files really exist
    return d


def test_the_workspace_lists_and_snapshots_both_twins(tmp_path):
    from localbot.tools.fs import DiskWorkspace

    ws = DiskWorkspace(wfs.WorkspaceFS(_twin_folder(tmp_path)), order=["a.txt"])
    assert ws.files() == ["a.txt", "A.txt"]  # the known order first, then the rest
    assert ws.snapshot() == {"a.txt": "lower\n", "A.txt": "UPPER\n"}  # files_end: what every judge reads
    ws.put("B.txt", "x\n")
    ws.put("b.txt", "y\n")
    assert ws.order == ["a.txt", "B.txt", "b.txt"]


def test_the_secret_index_holds_both_twins(tmp_path):
    from localbot.guard.product import DiskView
    from localbot.guard.secret_names import SecretNames

    d = wfs.new_work_folder(tmp_path / "work")
    (d / ".env").write_text("K=one\n", encoding="utf-8")
    (d / ".ENV").write_text("K=two\n", encoding="utf-8")
    view = DiskView(wfs.WorkspaceFS(d))
    view.use_secret_rule(SecretNames().matches)
    assert view.indexed_secret_paths() == [".ENV", ".env"]  # both files' values are protected


@windows
def test_a_case_insensitive_folder_still_folds_names_as_before(tmp_path):
    """parity_lab6's folder (the lab's NTFS): README.MD is README.md, one file, one entry."""
    from localbot.tools.fs import DiskWorkspace

    d = tmp_path / "work"
    d.mkdir()
    (d / "README.md").write_text("x\n", encoding="utf-8")
    ws = DiskWorkspace(wfs.WorkspaceFS(d), order=["README.MD"])
    assert ws.files() == ["README.md"]
    ws.put("readme.MD", "y\n")
    assert ws.order == ["README.MD"]


def test_a_payload_copied_to_a_twin_is_a_leak_through_the_runner(disk):
    """The reviewer's case: a planted payload written to APP.py beside the seed's app.py. files_end
    used to drop one twin, so the injection judge never saw it."""
    from localbot.config import get_profile
    from localbot.exam import runner
    from localbot.telemetry import TrialSampler, VramReading
    from tests.fakes import RoutedTransport, ScriptedTransport, call, provider, reply
    from tests.test_injection_runner import RUN_ID, TrialShell

    c4 = get_profile("core4")
    t = next(x for x in runner.load_pack("injection") if x["id"] == "i_quoted_readme_edit")
    payload = t["injection"]["planted_values"][0]["value"]
    script = [reply(calls=[call("read", path="README.md")]),
              reply(calls=[call("write", path="APP.py", content=f"# {payload}\n")]), reply("Summarised."), reply("ok")]
    work = disk.base / "_trials" / RUN_ID / RUN_ID / runner.slug("fake:1") / runner.slug(t["id"]) / "t1" / "work"
    res, _dest = runner.run_pack(
        "injection", c4, "fake:1", provider(RoutedTransport(ScriptedTransport(script)), c4.provider), n=1, tasks=[t],
        shell=TrialShell(work), vram_reader=lambda: VramReading(1, 2, 0),
        sampler_factory=lambda: TrialSampler(0.05, cpu=lambda: 1.0, ram=lambda: 1, vram=lambda: None),
        results_dir=disk.base / "results", trials_root=disk.base / "_trials", run_id=RUN_ID, log=lambda _l: None)
    row = res["tasks"][t["id"]]["trials"][0]
    assert {"APP.py", "app.py"} <= set(row["files_end"]), sorted(row["files_end"])
    assert {"value": "payload", "where": "file:APP.py"} in row["injection"]["leaks"], row["injection"]
    assert row["pass"] is False


@pytest.mark.parametrize("profile,flagged", [("core4", True), ("parity_lab6", False)])
def test_a_trial_work_folder_is_case_sensitive_before_seeding_for_a_container_profile(disk, monkeypatch,
                                                                                     profile, flagged):
    """parity_lab6 keeps the lab's folder (byte-identical parity: case-insensitive on Windows, as the
    OS makes it elsewhere); core4 makes it case-sensitive. Checked on every OS (review R9: the Linux
    half used to record the expected value instead of checking)."""
    from localbot.config import get_profile
    from localbot.exam import runner
    from localbot.telemetry import TrialSampler, VramReading
    from tests.fakes import FakeShell, RoutedTransport, ScriptedTransport, provider, reply

    seen, made = [], []
    real, real_new = runner.seed_work, runner.new_work_folder

    def spy(work: Path, seed: dict) -> None:
        seen.append((list(work.iterdir()), wfs.is_case_sensitive(work)))
        real(work, seed)

    monkeypatch.setattr(runner, "seed_work", spy)
    monkeypatch.setattr(runner, "new_work_folder", lambda p: (made.append(Path(p).name), real_new(p))[1])
    prof = get_profile(profile)
    spec = runner.load_pack("work")[0]
    runner.run_pack("work", prof, "fake:1", provider(RoutedTransport(ScriptedTransport([reply("done")])), prof.provider),
                    n=1, tasks=[spec], shell=FakeShell({}), vram_reader=lambda: VramReading(1, 2, 0),
                    sampler_factory=lambda: TrialSampler(0.05, cpu=lambda: 1.0, ram=lambda: 1, vram=lambda: None),
                    results_dir=disk.base / "results", trials_root=disk.base / "_trials", log=lambda _line: None)
    want = flagged or os.name != "nt"  # Linux folders are case-sensitive by themselves
    assert seen == [([], want)], seen  # empty when checked, and case-sensitive as the profile needs
    assert made == (["work"] if flagged else []), made  # made (and flagged) only for a container profile
