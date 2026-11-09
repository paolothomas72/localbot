"""queue_status: one plain line per model of a running (or finished) model queue. Read only.

    python scripts/queue_status.py --tag rerun1 [--runs-dir <path>] [--stall-minutes 75] [--broken-after 5]

Reads state/runs/queue_<tag>/ (state.json rows, else models/*/row.json; the pack logs under models/<name>/packs/).
Creates, changes and locks nothing, so it is safe to run while the queue runs.

``collect_status`` returns plain data and never prints or exits: job C18b (auto-skip of a stalled or broken model)
imports it. Exit codes of the command line: 0 no flag, 1 any flag, 2 run folder missing.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import model_queue as mq  # noqa: E402  (run_dir, _safe: the queue's own folder rules)
from localbot.config import ProviderConfig  # noqa: E402
from localbot.provider.ollama import gpu_fit, ps_row, urllib_transport  # noqa: E402

PsFn = Callable[[], list]

DEFAULT_STALL_MINUTES = 75  # one `long` try can run up to 3600 s
DEFAULT_BROKEN_AFTER = 5

# A finished try line (runner.py): "<model> <task> t1 pass=True ... infra=False". A line cut off before the
# infra value is not a try yet.
_TRY_RE = re.compile(r"\bt\d+\S*\s+pass=(?:True|False)\b.*\binfra=(True|False)\b")
_PACK_LOG_RE = re.compile(r"^packs_([^_]+)_([^_]+)_")
_STEP_BY_STATUS = {
    "done": "done",
    "capped": "capped",
    "could not download": "could not download",
    "over the card: Stage 2": "over the card",
    "error": "error",
}

EPILOG = (
    "Columns: 'errors' counts tries with infra=True in the newest pack log. While a pack runs the log has no\n"
    "no_result field, so a try that never ran is only caught here if it also set infra. 'no_result' comes from the\n"
    "'wrote ...' line that ends a finished pack and is shown for finished packs only; a pack finished before C17\n"
    "has no no_result field, and then the column says 'not recorded' (never 0).\n"
    "Flags: STALLED = last activity older than --stall-minutes; BROKEN = the newest --broken-after tries all\n"
    "infra=True; SPILL = the loaded model is not 100% on the GPU.\n"
    "Exit code: 0 no flag, 1 any flag, 2 run folder missing."
)


def default_ps_fn(transport=None) -> PsFn:
    """/api/ps read directly. OllamaProvider.ps() swallows every error and returns [], which would show an
    unreachable Ollama as 'not loaded'; this one lets the failure through."""
    def ps() -> list:
        url = ProviderConfig().base_url
        data = (transport or urllib_transport)(f"{url}/api/ps", None, 8)
        return list(data.get("models") or [])
    return ps


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None


def _load_rows(batch: Path) -> list[tuple[str, dict, Path]]:
    """(model, row, model folder). state.json rows first; if it is missing or does not parse, models/*/row.json."""
    data = _read_json(batch / "state.json")
    rows = data.get("rows") if isinstance(data, dict) else None
    out: list[tuple[str, dict, Path]] = []
    if isinstance(rows, dict):
        for name, row in rows.items():
            row = row if isinstance(row, dict) else {}
            model = str(row.get("model") or name)
            out.append((model, row, batch / "models" / mq._safe(model)))
        return out
    models = batch / "models"
    if models.is_dir():
        for folder in sorted(p for p in models.iterdir() if p.is_dir()):
            row = _read_json(folder / "row.json")
            row = row if isinstance(row, dict) else {}
            out.append((str(row.get("model") or folder.name), row, folder))
    return out


def _newest_mtime(folder: Path) -> float | None:
    newest = None
    if not folder.is_dir():
        return None
    for path in folder.rglob("*"):
        try:
            if path.is_file():
                m = path.stat().st_mtime
                newest = m if newest is None or m > newest else newest
        except OSError:
            continue
    return newest


def _scan_log(path: Path) -> dict:
    infra: list[bool] = []
    wrote = None
    for line in _read_text(path).splitlines():
        m = _TRY_RE.search(line)
        if m:
            infra.append(m.group(1) == "True")
        elif line.startswith("wrote ") and wrote is None:
            nr = re.search(r"\bno_result=(\d+)", line)
            ir = re.search(r"\binfra_reruns=(\d+)", line)
            dk = re.search(r"\bdisk \d+/(\d+)", line)  # C18b: the task total of the pack ("disk P/T")
            wrote = {"no_result": int(nr.group(1)) if nr else None,  # None = a pre-C17 line: not recorded
                     "infra_reruns": int(ir.group(1)) if ir else 0,
                     "total": int(dk.group(1)) if dk else None}
    return {"infra": infra, "wrote": wrote}


def _gpu(model: str, ps_models: list | None) -> tuple[str, str]:
    if ps_models is None:
        return "unknown", "unknown (Ollama not reachable)"
    if ps_row(ps_models, model) is None:
        return "not loaded", "not loaded"
    fit = gpu_fit(model, ps_models, None, None, None)
    if fit.lane == "spill":
        return "spill", fit.ps_says
    return "gpu", "all on the GPU"


def _record(model: str, row: dict, folder: Path, now: float, ps_models: list | None,
            stall_minutes: float, broken_after: int) -> dict:
    status = str(row.get("status") or "")
    logs = sorted((p for p in (folder / "packs").glob("packs_*.log") if p.is_file()),
                  key=lambda p: (p.stat().st_mtime, p.name)) if (folder / "packs").is_dir() else []
    scans = [_scan_log(p) for p in logs]
    newest = scans[-1] if scans else None
    infra = newest["infra"] if newest else []
    streak = 0
    for flag in reversed(infra):
        if not flag:
            break
        streak += 1
    pack = ""
    if logs:
        m = _PACK_LOG_RE.match(logs[-1].name)
        pack = m.group(2) if m else logs[-1].name
    if status in _STEP_BY_STATUS:
        step = _STEP_BY_STATUS[status]
    elif status.startswith("skipped:"):
        step = "skipped"  # C18b: the queue skipped this model; no flags, nothing left to watch
    elif logs:
        step = "full tests"
    else:
        step = "waiting"  # the quick check itself cannot be seen from the files (row.json is written after it)
    newest_file = _newest_mtime(folder)
    activity = None if newest_file is None else int(round((now - newest_file) / 60))
    gpu, gpu_text = _gpu(model, ps_models)
    flags: list[str] = []
    if step == "full tests" and activity is not None and activity > stall_minutes:
        flags.append("STALLED")
    if step == "full tests" and broken_after > 0 and len(infra) >= broken_after and all(infra[-broken_after:]):
        flags.append("BROKEN")
    if gpu == "spill":
        flags.append("SPILL")
    finished = [s["wrote"]["no_result"] for s in scans if s["wrote"]]
    no_result = None if any(v is None for v in finished) else sum(finished)
    # C18b: the no_result of the newest pack log, only when that pack has finished (None: running or not recorded)
    newest_pack_no_result = newest["wrote"]["no_result"] if newest and newest["wrote"] else None
    newest_pack_total = newest["wrote"]["total"] if newest and newest["wrote"] else None  # tasks ("disk P/T")
    return {
        "model": model, "step": step, "status": status, "pack": pack, "tries": len(infra),
        "pack_running": bool(newest) and newest["wrote"] is None,
        "no_result": no_result, "newest_pack_no_result": newest_pack_no_result,
        "newest_pack_total": newest_pack_total,
        "infra_reruns": sum(s["wrote"]["infra_reruns"] for s in scans if s["wrote"]),
        "last_activity_min": activity, "errors": sum(infra), "error_streak": streak,
        "gpu": gpu, "gpu_text": gpu_text, "flags": flags,
    }


def collect_status(runs_dir, tag: str, now: float, ps_fn: PsFn, stall_minutes: float = DEFAULT_STALL_MINUTES,
                   broken_after: int = DEFAULT_BROKEN_AFTER) -> list[dict]:
    """One record per model. Raises FileNotFoundError if the run folder is missing. Never prints, exits or writes."""
    batch = mq.run_dir(tag, Path(runs_dir))
    if not batch.is_dir():
        raise FileNotFoundError(str(batch))
    try:
        ps_models: list | None = list(ps_fn() or [])
    except Exception:
        ps_models = None
    return [_record(model, row, folder, now, ps_models, stall_minutes, broken_after)
            for model, row, folder in _load_rows(batch)]


def _ascii(text: object) -> str:
    return str(text).encode("ascii", "replace").decode("ascii")


def format_line(rec: dict) -> str:
    parts = [rec["model"], f"step: {rec['step']}"]
    if rec["pack"]:
        state = "running" if rec["pack_running"] else "finished"
        parts.append(f"pack: {rec['pack']} ({state}), {rec['tries']} tries done")
    act = rec["last_activity_min"]
    parts.append("last activity: " + ("none yet" if act is None else f"{act} min ago"))
    parts.append(f"errors (infra only while a pack runs): {rec['errors']}, newest in a row: {rec['error_streak']}")
    missing = "not recorded" if rec["no_result"] is None else rec["no_result"]
    parts.append(f"missing results (no_result, finished packs only): {missing}")
    parts.append(f"GPU: {rec['gpu_text']}")
    parts.append("flags: " + (" ".join(rec["flags"]) if rec["flags"] else "none"))
    return _ascii(" | ".join(parts))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="queue_status.py", description="One line per model of a model queue run. Read only.",
                                 epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="run folder name: state/runs/queue_<tag>/")
    ap.add_argument("--runs-dir", default=None, help="runs folder (default: LOCALBOT_RUNS_DIR, else state/runs)")
    ap.add_argument("--stall-minutes", type=float, default=DEFAULT_STALL_MINUTES,
                    help="STALLED when the last activity is older than this (default 75)")
    ap.add_argument("--broken-after", type=int, default=DEFAULT_BROKEN_AFTER,
                    help="BROKEN when this many newest tries in a row are infra=True (default 5)")
    return ap


def main(argv: list[str] | None = None, ps_fn: PsFn | None = None, now: float | None = None) -> int:
    args = build_parser().parse_args(argv)
    when = datetime.now(timezone.utc).timestamp() if now is None else now
    runs = Path(args.runs_dir) if args.runs_dir else mq.run_dir(args.tag).parent
    try:
        recs = collect_status(runs, args.tag, when, ps_fn or default_ps_fn(), args.stall_minutes, args.broken_after)
    except FileNotFoundError:
        print(_ascii(f"queue_status: no run folder for tag '{args.tag}' in {runs}"), file=sys.stderr)
        return 2
    stamp = datetime.fromtimestamp(when, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(_ascii(f"queue_status {stamp} tag={args.tag} models={len(recs)}"))
    for rec in recs:
        print(format_line(rec))
    return 1 if any(r["flags"] for r in recs) else 0


if __name__ == "__main__":
    sys.exit(main())
