"""H1 Track P: the files a public repo needs (BRIEF_H1 §2 row P): an MIT LICENSE at the root, a
pyproject.toml whose ``localbot`` command and runtime dependencies match the package, and a CI
workflow that runs the host tests on Windows and Ubuntu and can't publish anything."""
from __future__ import annotations

import ast
import importlib
import re
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

import localbot

REPO = Path(__file__).resolve().parents[1]
PY = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8")) if (REPO / "pyproject.toml").exists() else {}
CI_FILE = REPO / ".github" / "workflows" / "ci.yml"

# Imported only inside try/except ImportError, never installed on purpose: with it, the text
# parser would take a repair step the lab never took (parse/fn_call.py step 5).
OPTIONAL_IMPORTS = {"json_repair"}


def _ci() -> str:
    return CI_FILE.read_text(encoding="utf-8")


def test_the_license_is_mit_at_the_root():
    text = (REPO / "LICENSE").read_text(encoding="utf-8")
    assert re.match(r"MIT License\n\nCopyright \(c\) 2026 \S", text)
    vendored = (REPO / "third_party" / "localharness-0.22.1" / "LICENSE").read_text(encoding="utf-8")
    assert text.split("Permission is hereby", 1)[1] == vendored.split("Permission is hereby", 1)[1]


def test_the_copyright_holder_is_the_contributors():
    """H1-K K6 P1 (owner decision): the LICENSE names "Local Bot contributors", not a person."""
    assert (REPO / "LICENSE").read_text(encoding="utf-8").splitlines()[2] == "Copyright (c) 2026 Local Bot contributors"


@pytest.mark.skipif(not (REPO / "scripts" / "export_public.py").exists(),
                    reason="the scanner is not in this tree (the public export denies it); its own scan runs there")
def test_no_fake_key_is_written_out_in_the_code():
    """P1: a fake test key is built at run time ("sk-" + "..."), so the published code holds no
    key-shaped literal and the export scan needs no exception for it. Same patterns as the scan."""
    sys.path.insert(0, str(REPO))
    try:
        from scripts import export_public as ep
    finally:
        sys.path.remove(str(REPO))
    denied = {"scripts/export_public.py", "tests/test_export_public.py"}  # they hold the patterns
    files = [p for top in ("localbot", "scripts", "tests") for p in sorted((REPO / top).rglob("*.py"))
             if "__pycache__" not in p.parts and p.relative_to(REPO).as_posix() not in denied]
    assert len(files) > 50, len(files)
    hits = [f"{p.relative_to(REPO).as_posix()}:{h.line} {h.pattern}" for p in files
            for h in ep.scan_bytes(p.relative_to(REPO).as_posix(), p.read_bytes())
            if h.pattern in ep._KEYS | {"private-key"}]
    assert hits == []


@pytest.mark.skipif(not (REPO / "scripts" / "export_public.py").exists(),
                    reason="the scanner is not in this tree (the public export denies it); its own scan runs there")
def test_no_owner_name_or_personal_path_in_the_exported_code():
    """H1-K K6: the final export scan found the owner's name in a new test's docstring; this keeps
    every exported .py (localbot/, scripts/, tests/) free of name / personal-path hits."""
    sys.path.insert(0, str(REPO))
    try:
        from scripts import export_public as ep
    finally:
        sys.path.remove(str(REPO))
    denied = {"scripts/export_public.py", "tests/test_export_public.py", "tests/test_result_tools.py"}
    files = [p for top in ("localbot", "scripts", "tests") for p in sorted((REPO / top).rglob("*.py"))
             if "__pycache__" not in p.parts and p.relative_to(REPO).as_posix() not in denied]
    assert len(files) > 50, len(files)
    hits = [f"{p.relative_to(REPO).as_posix()}:{h.line} {h.pattern}" for p in files
            for h in ep.scan_bytes(p.relative_to(REPO).as_posix(), p.read_bytes())
            if h.pattern in ("name", "personal-path", "pc-name", "email")]
    assert hits == []


def test_pyproject_names_mit_and_ships_both_notices():
    assert PY["project"]["license"] == "MIT"
    assert set(PY["project"]["license-files"]) == {"LICENSE", "localbot/NOTICE"}
    assert PY["tool"]["setuptools"]["package-data"]["localbot"] == ["NOTICE", "exam/packs/*.json"]  # + H1-K K3
    assert PY["build-system"]["build-backend"] == "setuptools.build_meta"


def test_the_localbot_command_runs_the_package_main():
    module, func = PY["project"]["scripts"]["localbot"].split(":")
    main = getattr(importlib.import_module(module), func)
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0


def test_the_version_comes_from_the_package():
    assert PY["project"]["dynamic"] == ["version"]
    assert PY["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "localbot.__version__"}
    Version(localbot.__version__)  # PEP 440, or this raises


def test_requires_python_admits_the_pinned_interpreter():
    pinned = (REPO / ".python-version").read_text(encoding="utf-8").strip()
    assert pinned in SpecifierSet(PY["project"]["requires-python"])


def _imports(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _guarded_imports(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and any(
                isinstance(h.type, ast.Tuple) and any(getattr(e, "id", "") == "ImportError" for e in h.type.elts)
                or getattr(h.type, "id", "") == "ImportError" for h in node.handlers):
            for stmt in node.body:
                names |= _imports(stmt)
    return names


def test_every_third_party_import_in_the_package_is_declared():
    declared = {re.split(r"[<>=!~ ;\[]", d, maxsplit=1)[0].lower().replace("-", "_")
                for d in PY["project"]["dependencies"]}
    found: set[str] = set()
    guarded: set[str] = set()
    for f in sorted((REPO / "localbot").rglob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        found |= _imports(tree)
        guarded |= _guarded_imports(tree)
    third = found - set(sys.stdlib_module_names) - {"localbot"}
    assert "psutil" in declared
    assert third - OPTIONAL_IMPORTS <= declared, third - OPTIONAL_IMPORTS - declared
    assert OPTIONAL_IMPORTS <= guarded  # an optional import never runs unguarded
    assert not OPTIONAL_IMPORTS & declared


def test_setuptools_finds_every_subpackage():
    assert PY["tool"]["setuptools"]["packages"]["find"]["include"] == ["localbot*"]
    for d in (REPO / "localbot").rglob("*"):
        if d.is_dir() and d.name != "__pycache__" and any(d.glob("*.py")):
            assert (d / "__init__.py").is_file(), d


def test_ci_runs_the_host_tests_on_windows_and_ubuntu():
    ci = _ci()
    assert "windows-latest" in ci and "ubuntu-24.04" in ci
    assert '-m "not container"' in ci and "fail-fast: false" in ci


def test_ci_is_read_only_and_never_publishes():
    ci = _ci()
    assert re.search(r"^permissions:\n  contents: read\n", ci, re.M)
    assert "persist-credentials: false" in ci
    for bad in ("secrets.", "git push", "pypi", "twine", "GITHUB_TOKEN", "write", "pull_request_target"):
        assert bad not in ci, bad


def test_ci_keeps_lf_line_endings_on_windows():
    ci = _ci()
    assert "git config --global core.autocrlf false" in ci
    assert ci.index("core.autocrlf false") < ci.index("actions/checkout")


# The first release of each action that runs on Node 24 (each tag's action.yml `runs.using`, read
# with `gh api` on 2026-11-01): Node 20 left GitHub's runners on 2026-09-23 (BRIEF_H1K K3).
NODE24_FROM = {"actions/checkout": 5, "actions/setup-python": 6}


def test_every_action_is_pinned_to_a_full_commit_sha_of_a_node24_release():
    """A tag can be moved to other code; a commit SHA can't (H1-J J8, Track P's 5f82d5b). The
    comment names the release, and that release runs on Node 24 (H1-K K3a)."""
    uses = re.findall(r"^\s*-?\s*uses:\s*(\S+)(.*)$", _ci(), re.M)
    assert sorted(ref.split("@")[0] for ref, _ in uses) == sorted(NODE24_FROM)
    for ref, rest in uses:
        assert re.fullmatch(r"actions/[a-z-]+@[0-9a-f]{40}", ref), ref
        m = re.fullmatch(r"\s*# v(\d+)\.\d+\.\d+", rest)
        assert m, rest
        assert int(m.group(1)) >= NODE24_FROM[ref.split("@")[0]], (ref, rest)  # not a Node 20 release


def _build(tmp_path: Path) -> list[str]:
    """Build the wheel and the sdist from a copy of the tree (setuptools writes build/ and
    *.egg-info next to pyproject.toml) and return every name in both archives."""
    import shutil
    import subprocess
    import tarfile
    import zipfile

    src = tmp_path / "src"
    shutil.copytree(REPO / "localbot", src / "localbot", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy(REPO / name, src / name)
    out = tmp_path / "dist"
    for hook in ("build_sdist", "build_wheel"):  # one process per hook, as a PEP 517 frontend runs them
        code = f"import sys, setuptools.build_meta as b; b.{hook}(sys.argv[1])"
        r = subprocess.run([sys.executable, "-c", code, str(out)], cwd=src, capture_output=True, text=True,
                           timeout=300)
        assert r.returncode == 0, (hook, r.stdout[-2000:] + r.stderr[-2000:])
    (wheel,), (sdist,) = list(out.glob("*.whl")), list(out.glob("*.tar.gz"))  # setuptools logs to stdout
    with zipfile.ZipFile(wheel) as z:
        names = [f"wheel:{n}" for n in z.namelist()]
    with tarfile.open(sdist) as t:
        names += [f"sdist:{n.split('/', 1)[1]}" for n in t.getnames() if "/" in n]
    return names


@pytest.mark.skipif(importlib.util.find_spec("setuptools") is None,
                    reason="needs setuptools (requirements-dev.txt pins it; the build backend)")
def test_the_injection_pack_ships_in_the_wheel_and_the_sdist(tmp_path):
    """H1-K K3b: `--pack injection` reads localbot/exam/packs/injection.json next to runner.py, so
    an installed package needs it (setuptools ships only .py files unless package-data says so)."""
    names = _build(tmp_path)
    assert any(n.startswith("wheel:localbot/") for n in names) and any(n.startswith("sdist:") for n in names)
    assert "wheel:localbot/exam/packs/injection.json" in names, sorted(n for n in names if "exam" in n)
    assert "sdist:localbot/exam/packs/injection.json" in names


def test_the_injection_pack_is_in_the_public_export():
    allow_file = REPO / "docs" / "public" / "export_allowlist.txt"
    if not allow_file.exists():
        pytest.skip("needs a private file that is not in the public copy: docs/public/export_allowlist.txt")
    allow = allow_file.read_text(encoding="utf-8").splitlines()
    assert "localbot/exam/packs/injection.json" in [ln.split("#")[0].strip() for ln in allow]


def test_ci_installs_the_package_and_runs_its_command():
    ci = _ci()
    assert "python -m pip install -r requirements-dev.txt" in ci and "python -m pip install ." in ci
    assert "localbot --help" in ci
