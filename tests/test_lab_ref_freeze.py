"""H1-K K6 P2 (owner decision): lab_ref/ re-frozen as a scrubbed copy. The export scan's 6 hits
(the owner's name and a personal path, in 4 files) are replaced with neutral text, nothing else
changes, and MANIFEST.sha256 is rewritten for the new bytes. The parity tests (lab request bytes,
lab guards, lab packs) stay green as the proof that nothing else moved."""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LAB = REPO / "lab_ref"
BEFORE = "352fbcb"  # lab_ref as frozen on 2026-10-26, unchanged until H1-K P2
SCRUBBED = {"LICENSE": [3], "README.md": [4], "lib/gap_pack.py": [3], "lib/sandbox_privacy_tasks.py": [43]}


def _manifest() -> dict[str, str]:
    out = {}
    for line in (LAB / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        sha, rel = line.split("  ", 1)
        out[rel] = sha
    return out


def _before(rel: str) -> bytes | None:
    try:
        r = subprocess.run(["git", "show", f"{BEFORE}:lab_ref/{rel}"], cwd=REPO, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def test_every_lab_ref_file_matches_its_manifest_hash():
    m = _manifest()
    if (LAB / "results_sept").is_dir():
        assert len(m) >= 60, len(m)  # the repo: every lab_ref file is listed
    else:
        # 004h: the public copy holds back results_sept/ and writes its own, shorter manifest: no line for it at all.
        assert len(m) >= 20, len(m)
        assert [rel for rel in m if rel.startswith("results_sept/")] == []
    # results_sept/ is held back from the public copy; any other missing file is a failure, not a skip.
    held_back = [rel for rel in m if rel.startswith("results_sept/") and not (LAB / rel).exists()]
    bad = [rel for rel, sha in m.items() if rel not in held_back and
           (not (LAB / rel).exists() or hashlib.sha256((LAB / rel).read_bytes()).hexdigest() != sha)]
    assert bad == []
    if held_back:
        pytest.skip(f"needs private files that are not in the public copy: lab_ref/{held_back[0]} and {len(held_back) - 1} more")


@pytest.mark.skipif(not (REPO / "scripts" / "export_public.py").exists(), reason="the scanner is not in this tree")
def test_lab_ref_holds_no_name_or_personal_path():
    sys.path.insert(0, str(REPO))
    try:
        from scripts import export_public as ep
    finally:
        sys.path.remove(str(REPO))
    # The lab_ref files the public tree takes (docs/public/export_allowlist.txt); results_sept/ is held back.
    exported = [LAB / "LICENSE", LAB / "README.md", LAB / "MANIFEST.sha256", LAB / "paths.py",
                *sorted((LAB / "lib").rglob("*")), *sorted((LAB / "scripts").rglob("*"))]
    files = [p for p in exported if p.is_file() and "__pycache__" not in p.parts]
    assert len(files) >= 20, len(files)
    hits = [f"{p.relative_to(LAB).as_posix()}:{h.line} {h.pattern}" for p in files
            for h in ep.scan_bytes(f"lab_ref/{p.relative_to(LAB).as_posix()}", p.read_bytes())
            if h.pattern in ("name", "personal-path")]
    assert hits == []


def _escape_target(text: str) -> str:
    """The second path p_escape_write's prompt names (the one P2 scrubbed), as the model reads it."""
    m = re.search(r"also to (\S+) with content", text)
    assert m, "p_escape_write prompt changed shape"
    return m.group(1).replace("\\\\", "\\")


def test_r7_the_scrubbed_prompt_keeps_the_original_refusal(tmp_path):
    """Review R7: P2 first put ``%USERPROFILE%`` where the owner's user-folder path was. The path
    layer refuses that as an environment-variable reference, not a drive-letter path: p_escape_write
    would test another rule. The scrubbed path must be refused exactly like the original."""
    from localbot.guard.workspace_fs import WorkspaceError, WorkspaceFS

    old = _before("lib/sandbox_privacy_tasks.py")
    if old is None:
        pytest.skip(f"{BEFORE} is not in this repository")
    new_line = (LAB / "lib/sandbox_privacy_tasks.py").read_bytes().decode("utf-8").split("\n")[42]
    old_line = old.decode("utf-8").split("\n")[42]
    fs = WorkspaceFS(tmp_path)
    why = []
    for line in (old_line, new_line):
        with pytest.raises(WorkspaceError) as e:
            fs.resolve(_escape_target(line))
        why.append(str(e.value))
    assert why[1] == why[0] and "drive-letter" in why[1], why[1]


def test_the_refreeze_changed_only_the_scrubbed_lines():
    if _before("LICENSE") is None:
        pytest.skip(f"{BEFORE} is not in this repository")
    for rel, line_nos in SCRUBBED.items():
        old = _before(rel).decode("utf-8").split("\n")
        new = (LAB / rel).read_bytes().decode("utf-8").split("\n")  # bytes: 3 of these files are CRLF
        assert len(old) == len(new), rel
        changed = [i + 1 for i, (a, b) in enumerate(zip(old, new)) if a != b]
        assert changed == line_nos, (rel, changed)
    for rel in _manifest():  # every other lab_ref file: byte-identical to the 2026-10-26 freeze
        if rel not in SCRUBBED:
            assert (LAB / rel).read_bytes() == _before(rel), rel
