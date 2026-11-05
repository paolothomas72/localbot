"""Test-only container images (``git/`` is the S8d build context). Host tests import the tag
from here, so they never import a container test file (whose import probes the runtimes).

H1-J J6 (audit P2: the test image was run by tag): the base in ``git/Containerfile`` is pinned by
digest, the build is offline, and the container tests run the image by the full id the build
printed, never by the tag, so a re-tag or another build under the same tag can't change what
they test.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

GIT_IMAGE = "localhost/localbot-test-git:s8d"  # S8d test-only image; never ContainerConfig.image
CONTAINERFILE = Path(__file__).resolve().parent / "git" / "Containerfile"

_FULL_ID = re.compile(r"(sha256:)?[0-9a-f]{64}")


class ImageBuildError(RuntimeError):
    """``built`` is False when the build itself failed (e.g. the base image is not on this PC),
    True when it ran but printed no full image id to pin."""

    def __init__(self, msg: str, built: bool) -> None:
        super().__init__(msg)
        self.built = built


def build_argv(runtime: str) -> list[str]:
    """Offline: never pull, no network; ``-q`` prints only the image id."""
    pull = "--pull=never" if runtime == "podman" else "--pull=false"
    return [runtime, "build", pull, "--network", "none", "-q", "-t", GIT_IMAGE,
            "-f", str(CONTAINERFILE), str(CONTAINERFILE.parent)]


def image_id(stdout: str) -> str | None:
    """The full image id ``build -q`` printed last (docker: ``sha256:<64 hex>``, podman: 64 hex),
    as printed; None for anything else (a short id or a tag is not a pin)."""
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    return lines[-1] if lines and _FULL_ID.fullmatch(lines[-1]) else None


def build_git_image(runtime: str, run=subprocess.run) -> str:
    """Build the git test image and return its full id (the reference to run it by)."""
    r = run(build_argv(runtime), capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise ImageBuildError(f"test git image not built on {runtime} (base image missing?): "
                              f"{(r.stderr or '')[-300:]}", built=False)
    pinned = image_id(r.stdout or "")
    if pinned is None:
        raise ImageBuildError(f"{runtime} built the test git image but printed no full id to pin: "
                              f"{(r.stdout or '')[-200:]!r}", built=True)
    return pinned
