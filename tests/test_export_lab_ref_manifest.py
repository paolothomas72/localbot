"""BRIEF 004g: the public copy's own check lists must pass.

The export holds back lab_ref/results_sept/ (local paths), so the repo's lab_ref/MANIFEST.sha256 would show 41
failures to a stranger. The export now writes its own manifest (the repo's lines for the files it holds, byte for
byte) and its own lab_ref/README.md (said plainly: held back, and why). The repo's files do not change."""
from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "export_public.py"
MANIFEST, README = "lab_ref/MANIFEST.sha256", "lab_ref/README.md"

# 004h: these tests build an export from the repo's git history; the public copy has neither the exporter nor git.
pytestmark = pytest.mark.skipif(not SCRIPT.exists(), reason="the exporter is not in this tree (the public copy)")


@pytest.fixture(scope="module")
def ep():
    spec = importlib.util.spec_from_file_location("export_public_004g", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def export(ep, tmp_path_factory):
    """An export of the real repo commit, built the way the existing export tests build one."""
    out = tmp_path_factory.mktemp("export_004g") / "pub"
    assert ep.main(["build", "--repo", str(REPO), "--rev", "HEAD", "--out", str(out)]) == 0
    return out


def _at_head(rel: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{rel}"], capture_output=True, check=True).stdout


def _check_lists(out: Path) -> dict[str, list[str]]:
    """Every fingerprint list in the export, run the way its own text says (from the folder it sits in; the
    docs/CODE_*.sha256 list from the export root). Returns list -> problems (missing file or wrong bytes)."""
    lists = [p for p in out.rglob("*") if p.is_file() and (p.name.endswith(".sha256") or p.name == "SHA256SUMS")]
    found = {}
    for lst in sorted(lists):
        rel = lst.relative_to(out).as_posix()
        base = out if rel.startswith("docs/CODE_") else lst.parent
        problems, lines = [], lst.read_bytes().decode("utf-8").splitlines()
        for line in lines:
            digest, name = line.split("  ", 1)
            target = base / name
            if not target.is_file():
                problems.append(f"missing: {name}")
            elif hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                problems.append(f"differs: {name}")
        found[rel] = problems if lines else ["empty list"]
    return found


def test_a1_every_fingerprint_list_in_the_export_passes(export):
    found = _check_lists(export)
    for must in (MANIFEST, "results/SHA256SUMS", "docs/CODE_61fb534.sha256"):
        assert must in found, must
    assert {k: v for k, v in found.items() if v} == {}


def test_a2_the_export_manifest_is_the_repos_lines_for_the_files_it_holds_nothing_added(export):
    repo_lines = _at_head(MANIFEST).splitlines(keepends=True)
    kept = [ln for ln in repo_lines if (export / "lab_ref" / ln.decode("utf-8").rstrip("\n").split("  ", 1)[1]).is_file()]
    dropped = [ln for ln in repo_lines if ln not in kept]
    assert (export / MANIFEST).read_bytes() == b"".join(kept)  # same bytes, same order, none added
    assert dropped and all(ln.decode("utf-8").split("  ", 1)[1].startswith("results_sept/") for ln in dropped)
    assert not (export / "lab_ref" / "results_sept").exists()


def test_a3_the_export_readme_says_results_sept_is_held_back_and_changes_nothing_else(export):
    repo = _at_head(README).decode("utf-8").split("\n")
    pub = (export / README).read_bytes().decode("utf-8").split("\n")
    n = len([ln for ln in _at_head(MANIFEST).decode("utf-8").splitlines() if ln.split("  ", 1)[1].startswith("results_sept/")])
    assert len(pub) == len(repo)
    changed = [(a, b) for a, b in zip(repo, pub) if a != b]
    assert len(changed) == 2 and all(a.startswith("- `results_sept/`") or a.startswith("- `MANIFEST.sha256`") for a, _ in changed)
    results_line = next(b for a, b in changed if a.startswith("- `results_sept/`"))
    manifest_line = next(b for a, b in changed if a.startswith("- `MANIFEST.sha256`"))
    assert results_line.startswith("- `results_sept/`") and "held back" in results_line
    assert str(n) in results_line and "local paths" in results_line
    assert "only the files" in manifest_line and "stops matching" in manifest_line and "edited the copy" in manifest_line
    assert "something edited the frozen copy" not in manifest_line


def test_a4_the_repos_own_lab_ref_files_are_unchanged(ep):
    for rel in (MANIFEST, README):
        assert (REPO / rel).read_bytes() == _at_head(rel), rel
    assert len(_at_head(MANIFEST).splitlines()) == 62
    assert "- `results_sept/` \u2014 the September 27\u201329 score files Local Bot must match or beat.".encode("utf-8") in _at_head(README)


def test_a5_the_scan_passes_and_the_new_text_has_no_name_or_path(ep, export):
    hits, n, _allowed = ep.scan_tree_ex(export)
    assert n > 0 and hits == []
    for rel in (MANIFEST, README):
        assert ep.scan_bytes(rel, (export / rel).read_bytes()) == [], rel


def test_a6_the_code_list_gives_each_rewritten_file_a_reason(export):
    added = (export / "docs" / "CODE_ADDED_AFTER_61fb534.md").read_text(encoding="utf-8")
    for rel in (MANIFEST, README):
        line = next(ln for ln in added.splitlines() if ln.startswith(f"- `{rel}`"))
        assert "written by the export" in line, line
    assert "lists only the files in this copy" in next(ln for ln in added.splitlines() if f"`{MANIFEST}`" in ln)
    assert "sha256 " + hashlib.sha256((export / MANIFEST).read_bytes()).hexdigest() in added


# --- the rewrite itself, on small made-up trees ----------------------------------------------------

def _readme(ep) -> bytes:
    return _at_head(README)


def _tree(ep, manifest: str, *names: str):
    files = [("lab_ref/" + n, "lab_ref/" + n, b"x\n") for n in names]
    files += [(README, README, _readme(ep)), (MANIFEST, MANIFEST, manifest.encode("utf-8"))]
    return files


def test_a7_a_manifest_line_for_a_file_the_copy_does_not_hold_is_dropped_only_under_results_sept(ep):
    h = "0" * 64
    ok = _tree(ep, f"{h}  LICENSE\n{h}  results_sept/a.json\n{h}  results_sept/b.json\n", "LICENSE")
    out = {d: b for _s, d, b in ep.lab_ref_public(ok)}
    assert out[MANIFEST] == f"{h}  LICENSE\n".encode("utf-8")
    assert b"2 files" in out[README]
    with pytest.raises(ep.ExportError):  # a file held back for another reason has no explanation in the README
        ep.lab_ref_public(_tree(ep, f"{h}  LICENSE\n{h}  lib/gone.py\n", "LICENSE"))


def test_a8_a_readme_that_no_longer_has_the_expected_lines_is_an_error_not_a_silent_pass(ep):
    h = "0" * 64
    files = _tree(ep, f"{h}  LICENSE\n{h}  results_sept/a.json\n", "LICENSE")
    files[-2] = (README, README, b"# lab_ref\n\nsomething else\n")
    with pytest.raises(ep.ExportError):
        ep.lab_ref_public(files)


def test_a9_a_tree_without_the_manifest_is_left_alone(ep):
    files = [("a.txt", "a.txt", b"a\n")]
    assert ep.lab_ref_public(list(files)) == files


def test_a10_a_full_copy_with_nothing_held_back_keeps_the_repos_files_untouched(ep):
    h = "0" * 64
    files = _tree(ep, f"{h}  LICENSE\n", "LICENSE")
    assert ep.lab_ref_public(list(files)) == files
