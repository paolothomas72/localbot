"""S8f.2 — capture the exact first /api/chat request of each task, without sending it (read-only).

Why: on trust ``s_clean_list`` Ornith's first tool flipped from glob (Local Bot 36be41b, 3/3) to
bash_exec (4884b27 and a72c51e, 0/3 each) while the lab loop still mostly picks glob. The choice
is made on the first model turn, before any tool result, so this tool captures that request —
the bytes that would go on the wire — on each side, and diffs them.

No model is called and nothing leaves the process: ``urllib.request.urlopen`` is replaced by a
recorder that refuses every request, and every socket connect is refused. Each task therefore
ends at its first request (one captured /api/chat per task); other requests the code makes
(``/api/ps``, ``/api/tags``, ``/api/version``, the unload) are recorded and refused too.

Sides — the real code paths, nothing rebuilt by hand:
  localbot  ``localbot.exam.runner.run_pack`` → ``run_task`` → ``OllamaProvider``, wired as
            ``python -m localbot.exam.runner`` wires it, with the code from ``--repo`` (default:
            this checkout; point it at another checkout, e.g. a worktree at 36be41b, to capture
            that commit with this script). Trial folders as in a real run (deleted by run_pack);
            the result JSON and the undo stores go to a temp dir, never results/ or state/.
  lab       ``lab_ref/scripts/run_lh_product.py --slim`` (work), ``run_lh_safety.py`` (trust),
            ``run_lh_bands.py --privacy`` (privacy): each script's ``run_trial`` → ``loop`` →
            ``ollama_chat.chat``. Imported read-only with bytecode off; nothing written to lab_ref/.

Capture output (JSON): per pack and task the URL, method, headers, length and sha256 of the exact
body bytes, and the body (text and parsed) with test secrets redacted — the lab's fake canary in
s_no_secret's prompt, values of secret-named seed files. The sha256 is of the unredacted bytes.

Usage:
  python -I scripts/first_request.py capture --side localbot [--repo DIR] [--profile parity_lab6]
         --pack work|trust|privacy|all --model MODEL --out FILE
  python -I scripts/first_request.py capture --side lab --pack all --model MODEL --out FILE
  python -I scripts/first_request.py diff A.json B.json [C.json ...] [--task ID ...] [--show]
Exit: capture 0; diff 0 when the only differences are in DOCUMENTED, else 1.
"""
from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parents[1]
PACKS = ("work", "trust", "privacy")
REFUSED = "first_request.py: recorded, not sent (no model call)"
REFUSED_END = "refused"
MISSING = "<missing>"
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
# Differences allowed between Local Bot's and the lab's first request, each with its reason.
DOCUMENTED = {
    "url.host": "the lab posts to http://localhost:11434 (ollama_chat.CHAT); Local Bot to "
                "http://127.0.0.1:11434 (config default LOCALBOT_OLLAMA_URL, loopback only). Same "
                "local Ollama, port and path; the host is not part of the request body.",
}
_CANARY = re.compile(r"CANARY_[A-Za-z0-9_\-]+")
_SECRET_FILE = re.compile(r"secret|\.env|token|passw|cred|id_rsa|\.ssh|\.pem|\.key", re.I)


# --- recorder ----------------------------------------------------------------------------------


class Refused(RuntimeError):
    """Raised for every request: the recorder never sends anything. No "500" in the text, so
    neither loop retries."""


class Recorder:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.connects: list[str] = []

    def urlopen(self, req: Any, *_a: Any, **_kw: Any) -> Any:
        if isinstance(req, str):
            url, method, data, headers = req, "GET", None, {}
        else:
            url, method, data, headers = req.full_url, req.get_method(), req.data, dict(req.header_items())
        self.requests.append({"url": url, "method": method, "headers": headers, "data": data})
        raise Refused(REFUSED)

    @contextlib.contextmanager
    def installed(self):
        real_open = urllib.request.urlopen
        real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

        def connect(_sock, address):
            self.connects.append(str(address))
            raise ConnectionRefusedError(REFUSED)

        def connect_ex(_sock, address):
            self.connects.append(str(address))
            return errno.ECONNREFUSED

        urllib.request.urlopen = self.urlopen
        socket.socket.connect, socket.socket.connect_ex = connect, connect_ex
        try:
            yield self
        finally:
            urllib.request.urlopen = real_open
            socket.socket.connect, socket.socket.connect_ex = real_connect, real_connect_ex

    def chats(self, start: int) -> list[dict]:
        return [r for r in self.requests[start:] if urlsplit(r["url"]).path == "/api/chat"]

    def others(self) -> list[dict]:
        return [{"url": r["url"], "method": r["method"]} for r in self.requests
                if urlsplit(r["url"]).path != "/api/chat"]


# --- redaction -----------------------------------------------------------------------------------


def secret_strings(seed: dict) -> list[str]:
    """Values a secret-named seed file holds (whole lines, and the VALUE of KEY=VALUE lines)."""
    found: set[str] = set()
    for path, body in (seed or {}).items():
        if not _SECRET_FILE.search(path):
            continue
        for line in str(body).splitlines():
            s = line.strip()
            if len(s) >= 8:
                found.add(s)
            parts = re.split(r"[=:]", s, maxsplit=1)
            if len(parts) == 2 and len(v := parts[1].strip().strip("'\"")) >= 6:
                found.add(v)
    return sorted(found, key=len, reverse=True)


def _forms(secrets: list[str]) -> list[str]:
    """Each secret as raw text and as it appears inside a JSON string (json.dumps escapes)."""
    out = {f for s in secrets for f in (s, json.dumps(s)[1:-1], json.dumps(s, ensure_ascii=False)[1:-1])}
    return sorted(out, key=len, reverse=True)


def redact_text(text: str, secrets: list[str]) -> tuple[str, int]:
    n = 0
    for s in _forms(secrets):
        if s and s in text:
            n += text.count(s)
            text = text.replace(s, "[REDACTED]")
    text, k = _CANARY.subn("[REDACTED]", text)
    return text, n + k


def redact_obj(obj: Any, secrets: list[str]) -> Any:
    if isinstance(obj, str):
        return redact_text(obj, secrets)[0]
    if isinstance(obj, list):
        return [redact_obj(x, secrets) for x in obj]
    if isinstance(obj, dict):
        return {k: redact_obj(v, secrets) for k, v in obj.items()}
    return obj


# --- capture -------------------------------------------------------------------------------------


def _row(task: dict, req: dict, end: str) -> dict:
    data: bytes = req["data"] or b""
    raw = data.decode("utf-8")
    body = json.loads(raw)
    secrets = secret_strings(task.get("seed") or {})
    text, n = redact_text(raw, secrets)
    msgs = body.get("messages") or []
    return {
        "task": task["id"], "url": req["url"], "method": req["method"], "headers": req["headers"],
        "body_len": len(data), "body_sha256": hashlib.sha256(data).hexdigest(),
        "user_is_prompt": len(msgs) >= 2 and msgs[1].get("content") == task["prompt"],
        "redactions": n, "body_text": text, "body": redact_obj(body, secrets), "end": end,
    }


def _rows(pack: str, tasks: list[dict], chats: list[dict], ends: dict[str, str]) -> list[dict]:
    if len(chats) != len(tasks):
        raise SystemExit(f"{pack}: {len(chats)} /api/chat requests for {len(tasks)} tasks "
                         "(expected exactly one per task: the first, refused)")
    return [_row(t, r, ends.get(t["id"], MISSING)) for t, r in zip(tasks, chats)]


def _git(repo: Path, *args: str) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def manifest_ok(lab: Path) -> bool:
    """Every file in lab_ref/MANIFEST.sha256 still has its listed hash."""
    for line in (lab / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(maxsplit=1)
        p = lab / rel.strip()
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            return False
    return True


def _log(line: str) -> None:
    sys.stderr.write(line.encode("ascii", "backslashreplace").decode("ascii") + "\n")


def capture_localbot(repo: Path, profile_name: str, packs: list[str], model: str) -> dict:
    repo = repo.resolve()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repo))
    tmp = Path(tempfile.mkdtemp(prefix="s8f2_capture_"))
    os.environ["LOCALBOT_STATE_DIR"] = str(tmp / "state")  # undo stores: temp, not <repo>/state
    try:
        import localbot
        from localbot.config import get_profile
        from localbot.exam import runner
        from localbot.provider.ollama import OllamaProvider
        from localbot.telemetry import TrialSampler

        here = Path(localbot.__file__).resolve()
        if not here.is_relative_to(repo):
            raise SystemExit(f"localbot imported from {here}, not from --repo {repo}")
        profile = get_profile(profile_name)
        rec, out = Recorder(), {}
        with rec.installed():
            for pack in packs:
                tasks, start = runner.load_pack(pack), len(rec.requests)
                result, _dest = runner.run_pack(
                    pack, profile, model, OllamaProvider(profile.provider), n=1,
                    vram_reader=lambda: None, sampler_factory=lambda: TrialSampler(1.0, vram=None),
                    results_dir=tmp / "results", log=_log)
                ends = {}
                for tid, t in result["tasks"].items():
                    tr = t["trials"][0]
                    ok = tr.get("end_reason") == "provider_error" and REFUSED in (tr.get("error") or "")
                    ends[tid] = REFUSED_END if ok else f"{tr.get('end_reason')}: {tr.get('error')}"
                out[pack] = _rows(pack, tasks, rec.chats(start), ends)
        code = {"repo": str(repo), "commit": _git(repo, "rev-parse", "HEAD"),
                "localbot_dirty": bool(_git(repo, "status", "--porcelain", "--", "localbot")),
                "localbot_file": str(here), "lab_ref_manifest_ok": manifest_ok(repo / "lab_ref")}
        return {"side": "localbot", "profile": profile_name, "code": code, "packs": out,
                "other_requests": rec.others(), "connects_refused": rec.connects}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def capture_lab(lab: Path, packs: list[str], model: str) -> dict:
    lab = lab.resolve()
    sys.dont_write_bytecode = True  # nothing written into lab_ref/
    sys.argv = ["first_request.py", "--slim"]  # the gate's commands all run slim (spec §3)
    sys.path.insert(0, str(lab / "scripts"))
    import run_lh_product as P  # noqa: E402  (reads --slim from sys.argv at import)
    import run_lh_safety as S  # noqa: E402
    import run_lh_bands as B  # noqa: E402

    runs = {  # each script's own trial call, as its main() makes it
        "work": (lambda: P.run_trial(model, tools=list(P.TOOLS)), P.TASKS),
        "trust": (lambda: S.run_trial(model), S.TASKS),
        "privacy": (lambda: B.run_trial(model, B.PRIVACY), B.PRIVACY),
    }
    rec, out, printed = Recorder(), {}, io.StringIO()
    with rec.installed(), contextlib.redirect_stdout(printed):
        for pack in packs:
            fn, tasks = runs[pack]
            start = len(rec.requests)
            res = fn()
            ends = {tid: REFUSED_END if r.get("reason") == f"request failed: {REFUSED}" else str(r.get("reason"))
                    for tid, r in res.items()}
            out[pack] = _rows(pack, tasks, rec.chats(start), ends)
    for line in printed.getvalue().splitlines():
        _log(f"lab: {line}")
    code = {"lab_ref": str(lab), "system_is_slim": P.SYSTEM == P.SYSTEM_SLIM,
            "lab_ref_manifest_ok": manifest_ok(lab),
            "scripts": {"work": "run_lh_product.py --slim", "trust": "run_lh_safety.py",
                        "privacy": "run_lh_bands.py --privacy"}}
    return {"side": "lab", "profile": "lab", "code": code, "packs": out,
            "other_requests": rec.others(), "connects_refused": rec.connects}


def write_capture(cap: dict, model: str, dest: Path) -> None:
    cap = {"tool": "scripts/first_request.py (S8f.2)", "model": model, "refused_end": REFUSED_END,
           "documented": DOCUMENTED, **cap}
    text = json.dumps(cap, indent=2, ensure_ascii=False) + "\n"
    if _CANARY.search(text):
        raise SystemExit("refusing to write: a canary survived redaction")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


# --- diff ----------------------------------------------------------------------------------------


def request_for(cap: dict, pack: str, task: str) -> dict:
    for r in cap["packs"][pack]:
        if r["task"] == task:
            return r
    raise KeyError(f"{pack}/{task} not in capture ({cap.get('side')})")


def _d(field: str, a: Any, b: Any) -> dict:
    return {"field": field, "a": a, "b": b}


def _diff_dict(prefix: str, a: dict, b: dict) -> list[dict]:
    out = []
    for k in dict.fromkeys([*a, *b]):
        va, vb = a.get(k, MISSING), b.get(k, MISSING)
        if va != vb:
            out.append(_d(f"{prefix}.{k}", va, vb))
    return out


def _diff_messages(a: list, b: list) -> list[dict]:
    out = [_d("messages.count", len(a), len(b))] if len(a) != len(b) else []
    for i, (ma, mb) in enumerate(zip(a, b)):
        out += _diff_dict(f"messages[{i}]", ma, mb)
    return out


def _tool_fields(t: dict) -> dict:
    fn = t.get("function") if isinstance(t.get("function"), dict) else {}
    out = {f"function.{k}" if k not in ("name", "description", "parameters") else k: v for k, v in fn.items()}
    out.update({k: v for k, v in t.items() if k != "function"})
    return out


def _diff_tools(a: list, b: list) -> list[dict]:
    na = [(t.get("function") or {}).get("name") for t in a]
    nb = [(t.get("function") or {}).get("name") for t in b]
    out = []
    if na != nb:
        out.append(_d("tools.order" if sorted(map(str, na)) == sorted(map(str, nb)) else "tools.names", na, nb))
        pairs = [(i, a[i], b[nb.index(n)]) for i, n in enumerate(na) if n in nb]  # same name, any position
    else:
        pairs = [(i, ta, tb) for i, (ta, tb) in enumerate(zip(a, b))]
    for i, ta, tb in pairs:
        fa, fb = _tool_fields(ta), _tool_fields(tb)
        out += _diff_dict(f"tools[{i}]", fa, fb)
        pa, pb = fa.get("parameters"), fb.get("parameters")
        if pa == pb and json.dumps(pa) != json.dumps(pb):
            out.append(_d(f"tools[{i}].parameters.key_order", json.dumps(pa), json.dumps(pb)))
    return out


def _diff_body(a: dict, b: dict) -> list[dict]:
    out = []
    for k in dict.fromkeys([*a, *b]):
        va, vb = a.get(k, MISSING), b.get(k, MISSING)
        if MISSING in (va, vb) or not (isinstance(va, (dict, list)) and type(va) is type(vb)):
            if va != vb:
                out.append(_d(f"body.{k}", va, vb))
        elif k == "messages":
            out += _diff_messages(va, vb)
        elif k == "tools":
            out += _diff_tools(va, vb)
        elif isinstance(va, dict):
            out += _diff_dict(k, va, vb)
        elif va != vb:
            out.append(_d(f"body.{k}", va, vb))
    return out


def _first_diff(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def diff_requests(a: dict, b: dict) -> list[dict]:
    """Every difference between two captured requests, as {field, a, b}. Empty = same request."""
    out = []
    ua, ub = urlsplit(a["url"]), urlsplit(b["url"])
    if (ua.scheme, ua.port) != (ub.scheme, ub.port) or not {ua.hostname, ub.hostname} <= LOOPBACK:
        out.append(_d("url.server", f"{ua.scheme}://{ua.netloc}", f"{ub.scheme}://{ub.netloc}"))
    elif ua.hostname != ub.hostname:
        out.append(_d("url.host", ua.hostname, ub.hostname))
    if (ua.path, ua.query) != (ub.path, ub.query):
        out.append(_d("url.path", ua.path, ub.path))
    if a["method"] != b["method"]:
        out.append(_d("method", a["method"], b["method"]))
    ha = {k.lower(): v for k, v in a["headers"].items()}
    hb = {k.lower(): v for k, v in b["headers"].items()}
    out += _diff_dict("headers", ha, hb)
    if a["body_sha256"] == b["body_sha256"]:
        return out
    body = _diff_body(a["body"], b["body"])
    if not body:  # same JSON, different bytes: key order, spacing or escaping
        i = _first_diff(a["body_text"], b["body_text"])
        body = [_d("body.bytes", f"{a['body_len']} B, at {i}: {a['body_text'][max(0, i - 40):i + 40]!r}",
                   f"{b['body_len']} B, at {i}: {b['body_text'][max(0, i - 40):i + 40]!r}")]
    return out + body


def undocumented(diffs: list[dict]) -> list[dict]:
    return [d for d in diffs if d["field"] not in DOCUMENTED]


def _show(v: Any, other: Any) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    o = other if isinstance(other, str) else json.dumps(other, ensure_ascii=False)
    if len(s) <= 160:
        return repr(s)
    i = _first_diff(s, o)
    return f"[{len(s)} chars, differs at {i}] ...{s[max(0, i - 60):i + 100]!r}..."


def format_diffs(diffs: list[dict]) -> str:
    lines = []
    for d in diffs:
        tag = "documented" if d["field"] in DOCUMENTED else "DIFF"
        lines.append(f"    {tag} {d['field']}:\n      a: {_show(d['a'], d['b'])}\n      b: {_show(d['b'], d['a'])}")
    return "\n".join(lines)


def label(cap: dict) -> str:
    if cap["side"] == "localbot":
        return f"localbot@{(cap['code'].get('commit') or '?')[:7]}/{cap['profile']}"
    return "lab"


def run_diff(files: list[Path], only: list[str], show: bool) -> int:
    caps = [json.loads(Path(f).read_text(encoding="utf-8")) for f in files]
    names = [label(c) for c in caps]
    for c, n in zip(caps, names):
        print(f"{n}: model={c['model']} code={json.dumps(c['code'])}")
    compared = identical = bad = 0
    for pack in PACKS:
        if any(pack not in c["packs"] for c in caps):
            continue
        for task in [r["task"] for r in caps[0]["packs"][pack]]:
            if only and task not in only:
                continue
            rows = [request_for(c, pack, task) for c in caps]
            compared += 1
            print(f"\n== {pack}/{task}")
            for n, r in zip(names, rows):
                print(f"  {n}: {r['method']} {r['url']}  body {r['body_len']} B sha256 {r['body_sha256'][:16]}"
                      f"  redactions={r['redactions']} end={r['end']}")
            same = True
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    diffs = diff_requests(rows[i], rows[j])
                    if not diffs:
                        print(f"  {names[i]} vs {names[j]}: IDENTICAL (url, method, headers, body bytes)")
                        continue
                    if any(d["field"].startswith(("body", "messages", "tools", "options")) for d in diffs):
                        same = False
                    if undocumented(diffs):
                        bad += 1
                    body_same = rows[i]["body_sha256"] == rows[j]["body_sha256"]
                    print(f"  {names[i]} vs {names[j]}: body bytes {'IDENTICAL' if body_same else 'DIFFER'}")
                    print(format_diffs(diffs))
            identical += same
            if show:
                print(f"  body ({names[0]}, redacted):\n{rows[0]['body_text']}")
    print(f"\n{compared} task(s) compared; body identical on every side: {identical}; "
          f"pairs with an undocumented difference: {bad}")
    for field, why in DOCUMENTED.items():
        print(f"documented {field}: {why}")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="first_request.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--side", required=True, choices=["localbot", "lab"])
    c.add_argument("--repo", type=Path, default=HERE, help="localbot side: checkout whose code to capture")
    c.add_argument("--lab", type=Path, default=HERE / "lab_ref", help="lab side: lab_ref folder")
    c.add_argument("--profile", default="parity_lab6")
    c.add_argument("--pack", default="all", choices=[*PACKS, "all"])
    c.add_argument("--model", required=True)
    c.add_argument("--out", type=Path, required=True)
    d = sub.add_parser("diff")
    d.add_argument("files", nargs="+", type=Path)
    d.add_argument("--task", action="append", default=[])
    d.add_argument("--show", action="store_true", help="print the first file's (redacted) body per task")
    a = ap.parse_args(argv)
    if a.cmd == "diff":
        return run_diff(a.files, a.task, a.show)
    packs = list(PACKS) if a.pack == "all" else [a.pack]
    cap = (capture_localbot(a.repo, a.profile, packs, a.model) if a.side == "localbot"
           else capture_lab(a.lab, packs, a.model))
    write_capture(cap, a.model, a.out)
    n = sum(len(v) for v in cap["packs"].values())
    _log(f"wrote {a.out}: {cap['side']} {n} first request(s), {len(cap['other_requests'])} other request(s) "
         f"refused, {len(cap['connects_refused'])} connect(s) refused")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
