"""H1-K K4: every docs/h1/H1*.md track log is folded into docs/BUILD_LOG.md, verbatim, in the
order the logs reached main, under one H1 section; docs/h1/ keeps the originals and nothing that
was in BUILD_LOG before changes. Both files are private (the public export denies them), so
these tests skip in a tree without them."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "docs" / "BUILD_LOG.md"
H1 = REPO / "docs" / "h1"
pytestmark = pytest.mark.skipif(not (LOG.exists() and H1.is_dir()), reason="the private build log is not in this tree")

# (file, the main commit that brought it, the main commit of the text folded): the merge for a track
# log, the commit itself for H1_I. Review R10: H1_L and H1_I were changed on main after they arrived
# (by d8b66da and 473ff95); the fold holds that last text, so the label names its commit too.
MERGE_ORDER = [("H1_T.md", "22da8f1", "22da8f1"), ("H1_G.md", "3fdb97e", "3fdb97e"),
               ("H1_C.md", "5bd259b", "5bd259b"), ("H1_L.md", "f635dc6", "d8b66da"),
               ("H1_I.md", "28e828a", "473ff95"), ("H1_P.md", "6d6976c", "6d6976c"),
               ("H1J_C.md", "a9bd672", "a9bd672"), ("H1J_G.md", "cedde2d", "cedde2d"),
               ("H1J_L.md", "d8b66da", "d8b66da"), ("H1J_T.md", "352fbcb", "352fbcb"),
               ("H1K.md", "5f4e575", "5f4e575"), ("H1L.md", "f653fc2", "f653fc2"),
               ("H1M.md", "d96c416", "d96c416")]
NOT_YET: set[str] = set()  # H1M.md was folded by the reviewer right after its merge (d96c416)
BEFORE = "352fbcb"  # BUILD_LOG as it was when H1-K started
SECTION = "# H1 — Stage 1 hardening: the track logs (folded from docs/h1/, in merge order, verbatim; H1-K K4)"


def _marker(name: str, arrived: str, final: str) -> str:
    text = "" if final == arrived else f"; this text from {final}"
    return f"<!-- folded from docs/h1/{name} (on main by {arrived}{text}) -->"


def test_the_merge_order_names_every_track_log():
    on_disk = {p.name for p in H1.glob("H1*.md")} - NOT_YET
    assert on_disk and on_disk == {name for name, _, _ in MERGE_ORDER}


def test_every_track_log_is_folded_verbatim_once_in_merge_order():
    log = LOG.read_text(encoding="utf-8")
    assert log.count(SECTION) == 1
    start = log.index(SECTION)
    at = []
    for name, arrived, final in MERGE_ORDER:
        text = (H1 / name).read_text(encoding="utf-8")
        assert text.strip() and log.count(text) == 1, name  # verbatim, exactly once
        marker = _marker(name, arrived, final)
        assert log.count(marker) == 1 and log.index(marker) < log.index(text), (name, marker)
        at.append(log.index(text))
    assert start < at[0] and at == sorted(at)


def test_each_label_names_the_commit_of_the_folded_text():
    """Review R10: the text folded is byte for byte the file at the commit its label names."""
    for name, _arrived, final in MERGE_ORDER:
        r = subprocess.run(["git", "show", f"{final}:docs/h1/{name}"], cwd=REPO, capture_output=True, timeout=30)
        if r.returncode != 0:
            pytest.skip(f"{final} is not in this repository")
        assert r.stdout == (H1 / name).read_bytes(), (name, final)


def test_the_two_speed_limits_are_written_down_as_known_limits():
    """H1-K K6 D6 (owner decision): no stat-cached undo snapshots; capped reads go to Stage 2. No
    code: both are known limits, in the spec (§11) and in the public evidence page."""
    spec = (REPO / "docs" / "STAGE1_SPEC.md").read_text(encoding="utf-8")
    later = spec[spec.index("## 11. Open for later stages"):]
    evidence = (REPO / "docs" / "public" / "EVIDENCE.md").read_text(encoding="utf-8")
    for text in (later, evidence):
        assert "no stat-cached snapshots" in text.lower() or "no cache" in text.lower(), text[:400]
        assert "size cap" in text.lower(), text[:400]
    assert "D6" in later and "Stage 2" in later


def test_nothing_that_was_in_the_build_log_changed():
    try:
        before = subprocess.run(["git", "show", f"{BEFORE}:docs/BUILD_LOG.md"], cwd=REPO, capture_output=True,
                                timeout=30).stdout.decode("utf-8")
    except (OSError, subprocess.SubprocessError):
        pytest.skip("no git here")
    if not before:
        pytest.skip(f"{BEFORE} is not in this repository")
    assert LOG.read_text(encoding="utf-8").startswith(before)  # appended to, never reworded
