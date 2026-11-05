"""The secret-file names (H1 G-3): one list for core4's secret index (a file whose values may not be
copied into another file or shown in chat), the write tool's secrets-path rule, the same rule on
files a bash step made, and Track C's container masking (a secret file is not mounted into the
container). Parity (emulated) does not use it: it keeps the lab's rule exactly
(``product.is_secret_name``).

A name matches when its case-folded base name matches one of ``globs`` (``fnmatchcase``) and none
of ``templates`` — so ``.env.local`` is a secret and ``.env.example`` is not. The first three globs
are the lab's rule (``endswith("secret.txt" | ".env" | "secrets")`` on the path is the same as
those suffixes on the base name). ``SecretNames(globs=…)`` takes another list (configurable).

``matches_path_rule`` is the same list for the write rule (write tool + bash-made files) minus
``INDEX_ONLY``: the lab indexed ``secret.txt`` but never refused writing it, and the exam's privacy
judges score what a model does to ``secret.txt``, so refusing it would change exam results.
"""
from __future__ import annotations

from fnmatch import fnmatchcase

from localbot.guard.workspace_fs import fold_name

LAB_GLOBS = ("*.env", "*secret.txt", "*secrets")
SECRET_GLOBS = (
    *LAB_GLOBS,
    ".env.*",                                                     # .env.local, .env.production…
    "secrets.*", "credentials", "credentials.*", ".git-credentials",  # H1-K D4: any extension (fail toward hiding)
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "id_ecdsa_sk", "id_ed25519_sk",
    "*.pem", "*.key", "*.p12", "*.pfx", "*.ppk",
    ".npmrc", ".pypirc", ".netrc", "_netrc", ".pgpass", ".htpasswd",
)
TEMPLATE_GLOBS = ("*.example", "*.sample", "*.template", "*.dist", "*.tmpl", "*.pub")
INDEX_ONLY = ("*secret.txt",)


class SecretNames:
    """``matches(path)``: is this a secret file by name. ``path`` is workspace-relative, ``/`` or
    ``\\`` separated, any case."""

    def __init__(self, globs: tuple[str, ...] = SECRET_GLOBS,
                 templates: tuple[str, ...] = TEMPLATE_GLOBS) -> None:
        self.globs = tuple(fold_name(g) for g in globs)
        self.templates = tuple(fold_name(g) for g in templates)
        self.path_globs = tuple(g for g in self.globs if g not in INDEX_ONLY)

    def matches(self, path: str) -> bool:
        """A secret file by name: indexed, its values never copied or shown."""
        return self._match(path, self.globs)

    def matches_path_rule(self, path: str) -> bool:
        """A path the write tool refuses (and a bash step may not make): ``matches`` minus
        ``INDEX_ONLY``."""
        return self._match(path, self.path_globs)

    def _match(self, path: str, globs: tuple[str, ...]) -> bool:
        base = fold_name(path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1])
        base = base.rstrip(". ")  # H1 P2: Win32 drops a trailing '.' / space: '.env ' opens '.env'
        if not base or any(fnmatchcase(base, t) for t in self.templates):
            return False
        return any(fnmatchcase(base, g) for g in globs)


DEFAULT = SecretNames()
