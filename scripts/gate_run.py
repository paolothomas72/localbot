"""Run a gate plan one row at a time, and pick up after a restart.

No model is loaded here. Each row is one ``python -m localbot.exam.runner`` process.
G1 and G2 are pytest jobs, not rows in the plan. The batch script stays as it is.
"""
from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import runlock  # noqa: E402

REPO = SCRIPTS.parent
PACKS = frozenset({"work", "trust", "privacy", "injection", "long", "all"})
PROFILES = frozenset({"parity_lab6", "core4", "core4_product"})
WROTE = re.compile(r"^wrote (.+?\.json)(?:\s{2}|\s*$)")
LAST_STDOUT = ""
LAST_STDERR = ""


class PlanError(Exception):
    """The plan file is not usable. The message is the plain reason."""


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_porcelain() -> str:
    try:
        proc = subprocess.run(["git", "status", "--porcelain", "--", ".", ":(exclude)results"],
                              cwd=REPO, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"git status failed: {exc}"
    return proc.stdout if proc.returncode == 0 else proc.stderr or "git status failed"


def git_head() -> str:
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def gpu_visible() -> bool:
    try:
        proc = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def ollama_tags() -> dict[str, str] | None:
    """Tag to digest from ``ollama list``, or None when the command itself fails.

    A test may pass a set of tags; those count as present with no digest.
    """
    try:
        proc = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    found: dict[str, str] = {}
    for line in proc.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            found[parts[0]] = parts[1] if len(parts) > 1 else ""
    return found


def _digest_seen(tags, model: str) -> str | None:
    """The digest ``ollama list`` showed, or None when the tag is absent."""
    if isinstance(tags, dict):
        if model not in tags:
            return None
        return str(tags[model])
    if model not in tags:
        return None
    return ""


def launch(argv: list[str], log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        proc = subprocess.run(argv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def load_plan(path: Path | str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data["rows"] if isinstance(data, dict) else data
    seen: set[tuple] = set()
    out = []
    for index, row in enumerate(rows, 1):
        reason = _row_problem(row)
        if reason:
            raise PlanError(f"row {index}: {reason}")
        key = (row["gate"], row["model"], row["pack"], row["profile"], row["n"],
               row.get("num_ctx"), row.get("max_turns"), row.get("num_predict"), row.get("tasks"))
        if key in seen:
            raise PlanError(f"row {index}: duplicate row")
        seen.add(key)
        out.append(row)
    return out


def _row_problem(row: dict) -> str | None:
    if row.get("pack") not in PACKS:
        return f"unknown pack {row.get('pack')!r}"
    if row.get("profile") not in PROFILES:
        return f"unknown profile {row.get('profile')!r}"
    try:
        n = int(row["n"])
    except (KeyError, TypeError, ValueError):
        return "n must be at least 1"
    if n < 1:
        return "n must be at least 1"
    if "num_ctx" in row and row["num_ctx"] is not None:
        if int(row["num_ctx"]) <= 0 or int(row["num_ctx"]) % 1024 != 0:
            return "num_ctx must be a multiple of 1024"
    if "max_turns" in row and row["max_turns"] is not None and int(row["max_turns"]) < 0:
        return "max_turns must not be negative"
    if "num_predict" in row and row["num_predict"] is not None and int(row["num_predict"]) < 1:
        return "num_predict must be at least 1"
    for key in ("gate", "model"):
        if not row.get(key):
            return f"missing {key}"
    return None


def grouped(rows: list[dict]) -> list[dict]:
    """Keep gate order. Inside a gate, keep each model's rows together (first model first)."""
    gates: list[str] = []
    by_gate: dict[str, list[dict]] = {}
    for row in rows:
        by_gate.setdefault(row["gate"], [])
        if row["gate"] not in gates:
            gates.append(row["gate"])
        by_gate[row["gate"]].append(row)
    ordered = []
    for gate in gates:
        models: list[str] = []
        buckets: dict[str, list[dict]] = {}
        for row in by_gate[gate]:
            buckets.setdefault(row["model"], [])
            if row["model"] not in models:
                models.append(row["model"])
            buckets[row["model"]].append(row)
        for model in models:
            ordered.extend(buckets[model])
    return ordered


def filter_rows(rows: list[dict], gate: str | None, models: str | None) -> list[dict]:
    chosen = rows
    if gate:
        chosen = [r for r in chosen if r["gate"] == gate]
    if models:
        wanted = {m for m in models.split(",") if m}
        chosen = [r for r in chosen if r["model"] in wanted]
    return chosen


def command_argv(row: dict, python: str) -> list[str]:
    """The runner command. Omitted setting keys add no flag (the profile default stays)."""
    cmd = [str(python), "-m", "localbot.exam.runner",
           "--pack", row["pack"], "--profile", row["profile"], "--n", str(row["n"])]
    if row.get("num_ctx") is not None:
        cmd += ["--num-ctx", str(row["num_ctx"])]
    if row.get("max_turns") is not None:
        cmd += ["--max-turns", str(row["max_turns"])]
    if row.get("num_predict") is not None:
        cmd += ["--num-predict", str(row["num_predict"])]
    if row.get("tasks"):
        cmd += ["--tasks", str(row["tasks"])]
    cmd.append(row["model"])
    return cmd


def show_command(argv: list[str]) -> str:
    return " ".join(argv)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)


def efficiency(result: dict) -> dict:
    """Numbers already stored on one runner result. No new measurement."""
    calls = []
    reasons: Counter[str] = Counter()
    tok_s = []
    output = 0
    vram = []
    for task in (result.get("tasks") or {}).values():
        for trial in task.get("trials") or []:
            reasons[str(trial.get("end_reason") or "unknown")] += 1
            tel = trial.get("telemetry") or {}
            if tel.get("tok_s") is not None:
                tok_s.append(float(tel["tok_s"]))
            if tel.get("eval_count"):
                output += int(tel["eval_count"])
            if tel.get("vram_used_mib_peak") is not None:
                vram.append(int(tel["vram_used_mib_peak"]))
            for call in tel.get("calls") or []:
                calls.append(call)
    peaks = [c["prompt_eval_count"] for c in calls if c.get("prompt_eval_count") is not None]
    walls = [float(c["wall_s"]) for c in calls if c.get("wall_s") is not None]
    med = _median(walls)
    load = round(walls[0] - med, 3) if walls and med is not None else 0.0
    total = int(result.get("total") or 0)
    wall = float(result.get("wall_s") or 0)
    lane = ((result.get("gpu_fit") or {}).get("lane")) or "unknown"
    comparable = bool(result.get("timing_comparable", lane == "gpu"))
    return {
        "wall_s": wall,
        "tok_s": statistics.median(tok_s) if tok_s else None,
        "context_peak": max(peaks) if peaks else None,
        "output_tokens": output,
        "turns": len(calls),
        "end_reasons": dict(reasons),
        "gpu_lane": lane,
        "vram_peak_mib": max(vram) if vram else None,
        "time_per_task_s": round(wall / total, 4) if total else None,
        "model_load_s": load,
        "timing_comparable": comparable and lane == "gpu",
    }


def estimate_text(rows: list[dict], history: list[Path]) -> str:
    buckets: dict[tuple[str, str], list[float]] = {}
    for path in history:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for row in data.get("rows") or []:
            if row.get("status") not in (None, "ok"):
                continue
            if not row.get("timing_comparable", False):
                continue
            if row.get("wall_s") is None:
                continue
            buckets.setdefault((row["model"], row["pack"]), []).append(float(row["wall_s"]))
    lines = []
    seen = set()
    for row in rows:
        key = (row["model"], row["pack"])
        if key in seen:
            continue
        seen.add(key)
        values = buckets.get(key) or []
        shown = "unknown" if not values else str(_median(values))
        lines.append(f"{row['model']} {row['pack']} {shown}")
    return "\n".join(lines) + ("\n" if lines else "unknown\n")


def _commit_of(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    git = (data.get("run_env") or {}).get("git") or {}
    return str(git.get("commit") or "")


def _commit_matches(got: str, expected: str) -> bool:
    got, expected = got.lower(), expected.lower()
    return bool(got) and (got == expected or got.startswith(expected))


def _wrote_path(text: str) -> Path | None:
    for line in text.splitlines():
        match = WROTE.match(line.strip())
        if match:
            return Path(match.group(1))
    return None


def _row_key(row: dict) -> tuple:
    return (row["gate"], row["model"], row["pack"], row["profile"])


def _done_keys(batch: Path, commit: str) -> set[tuple]:
    done = set()
    if not batch.is_dir():
        return done
    items = []
    for journal in sorted(batch.glob("journal_*.jsonl")):
        for line in journal.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    for summary in sorted(batch.glob("summary_*.json")):
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        items.extend(payload.get("rows") or [])
    for item in items:
        if item.get("status") != "ok":
            continue
        result = _wrote_path(str(item.get("result") or ""))
        if result is None or not result.is_file():
            continue
        if not _commit_matches(_commit_of(result), commit):
            continue
        done.add((item.get("gate"), item.get("model"), item.get("pack"), item.get("profile")))
    return done


def _free_name(directory: Path, base: str, suffix: str) -> Path:
    path = directory / f"{base}{suffix}"
    if not path.exists():
        return path
    number = 2
    while True:
        candidate = directory / f"{base}_{number}{suffix}"
        if not candidate.exists():
            return candidate
        number += 1


def _runs_dir() -> Path:
    override = os.environ.get("LOCALBOT_RUNS_DIR")
    return Path(override) if override else REPO / "state" / "runs"


def _say(out: list[str], err: list[str], stream: str, text: str) -> None:
    (out if stream == "out" else err).append(text)


def main(argv: list[str] | None = None) -> int:
    global LAST_STDOUT, LAST_STDERR
    import argparse
    out: list[str] = []
    err: list[str] = []

    def finish(code: int) -> int:
        global LAST_STDOUT, LAST_STDERR
        LAST_STDOUT = "\n".join(out)
        LAST_STDERR = "\n".join(err)
        if LAST_STDOUT:
            print(LAST_STDOUT)
        if LAST_STDERR:
            print(LAST_STDERR, file=sys.stderr)
        return code

    parser = argparse.ArgumentParser(prog="gate_run")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--gate", default=None)
    parser.add_argument("--models", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--history", action="append", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    try:
        rows = load_plan(args.plan)
    except (OSError, json.JSONDecodeError, PlanError) as exc:
        _say(out, err, "err", str(exc))
        return finish(2)
    rows = filter_rows(rows, args.gate, args.models)
    if not rows:
        _say(out, err, "err", "no rows match the gate and model filters")
        return finish(1)
    rows = grouped(rows)
    if args.list:
        for row in rows:
            _say(out, err, "out", show_command(command_argv(row, args.python)))
        return finish(0)
    if args.estimate:
        history = [Path(p) for p in (args.history or [])]
        _say(out, err, "out", estimate_text(rows, history).rstrip("\n"))
        return finish(0)

    dirty = git_porcelain()
    if dirty.strip():
        _say(out, err, "err", "STOP: the git tree is not clean (uncommitted changes). Nothing ran.")
        return finish(2)
    head = git_head()
    if not _commit_matches(head, args.commit):
        _say(out, err, "err", f"STOP: HEAD is {head or 'unknown'}, not {args.commit}. Nothing ran.")
        return finish(2)
    try:
        lock = runlock.acquire("gate_run", cmd=["gate_run", args.tag or ""])
    except runlock.LockBusy as exc:
        _say(out, err, "err", f"STOP: busy: {runlock.describe(exc.holder)}. Nothing ran.")
        return finish(3)
    try:
        if not gpu_visible():
            _say(out, err, "err", "STOP: the GPU is not visible (nvidia-smi failed). Nothing ran.")
            return finish(2)
        tags = ollama_tags()
        if tags is None:
            _say(out, err, "err", "STOP: ollama list failed. Nothing ran.")
            return finish(2)
        tag = args.tag or ("gate_" + stamp())
        batch = _runs_dir() / tag
        batch.mkdir(parents=True, exist_ok=True)
        done = _done_keys(batch, args.commit)
        journal = _free_name(batch, "journal_" + stamp(), ".jsonl")
        recorded = []
        for row in rows:
            key = _row_key(row)
            if key in done:
                _say(out, err, "out", f"skip done {row['gate']} {row['model']} {row['pack']}")
                continue
            seen = _digest_seen(tags, row["model"])
            expected = str(row.get("digest") or "")
            if seen is None:
                _say(out, err, "out", f"SKIPPED {row['gate']} {row['model']}: not in ollama list")
                recorded.append({**_public_row(row), "status": "SKIPPED", "result": "not in ollama list",
                                 "digest_seen": ""})
                continue
            if expected and seen != expected:
                reason = f"digest differs: expected {expected}, found {seen}"
                _say(out, err, "out", f"SKIPPED {row['gate']} {row['model']}: {reason}")
                recorded.append({**_public_row(row), "status": "SKIPPED", "result": reason,
                                 "digest_seen": seen})
                continue
            log_path = _free_name(batch, _log_base(tag, row), ".log")
            rc = launch(command_argv(row, args.python), log_path, REPO)
            text = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
            result_path = _wrote_path(text)
            ok = rc == 0 and result_path is not None and result_path.is_file() and _commit_matches(
                _commit_of(result_path), args.commit)
            status = "ok" if ok else "FAILED"
            result_line = text.strip().splitlines()[-1] if text.strip() else "no 'wrote' line"
            item = {**_public_row(row), "status": status, "result": result_line, "log": str(log_path),
                    "exit": rc, "digest_seen": seen}
            if ok and result_path is not None:
                item.update(efficiency(json.loads(result_path.read_text(encoding="utf-8"))))
                item["result_file"] = str(result_path)
            recorded.append(item)
            with journal.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(item) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if ok:
                done.add(key)
        _write_snapshots(batch, recorded, journal)
        _say(out, err, "out", _gate_timing_text(recorded))
        return finish(0 if all(r["status"] in {"ok", "SKIPPED"} for r in recorded) or not recorded else 1)
    finally:
        lock.release()


def _public_row(row: dict) -> dict:
    return {"gate": row["gate"], "model": row["model"], "pack": row["pack"],
            "profile": row["profile"], "n": row["n"]}


def _log_base(tag: str, row: dict) -> str:
    model = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in row["model"])
    return f"{tag}_{row['gate']}_{row['pack']}_{row['profile']}_{model}"


def _write_snapshots(batch: Path, recorded: list[dict], journal: Path) -> None:
    previous = []
    for older in sorted(batch.glob("journal_*.jsonl")):
        if older == journal:
            continue
        for line in older.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    previous.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    rows = previous + recorded
    payload = {"rows": rows, "gates": _gate_timing(rows)}
    name = _free_name(batch, "summary_" + stamp(), ".json")
    name.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    tsv = name.with_suffix(".tsv")
    header = "gate\tmodel\tpack\tprofile\tstatus\twall_s\tresult"
    lines = [header]
    for row in rows:
        lines.append("\t".join(str(row.get(k, "")) for k in
                               ("gate", "model", "pack", "profile", "status", "wall_s", "result")))
    tsv.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    weekly = {
        "generated": stamp(),
        "runs": [r for r in rows if r.get("status") == "ok"],
        "not_tested": ["the host test suite", "the container attack pack"],
    }
    (batch / "weekly_data.json").write_text(json.dumps(weekly, indent=2) + "\n", encoding="utf-8", newline="\n")


def _gate_timing(rows: list[dict]) -> list[dict]:
    by: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        by.setdefault(row["gate"], []).append(row)
    out = []
    for gate, items in by.items():
        comparable = [r for r in items if r.get("timing_comparable")]
        walls = [float(r["wall_s"]) for r in items if r.get("wall_s") is not None]
        loads = [float(r["model_load_s"]) for r in comparable if r.get("model_load_s") is not None]
        slow = sorted(items, key=lambda r: float(r.get("wall_s") or 0), reverse=True)[:3]
        task_times = [float(r["time_per_task_s"]) for r in items if r.get("time_per_task_s") is not None]
        out.append({
            "gate": gate,
            "rows": len(items),
            "wall_s": round(sum(walls), 3) if walls else 0,
            "model_load_s": round(sum(loads), 3) if loads else 0,
            "time_per_task_s": _median(task_times),
            "slowest": [{"model": r["model"], "pack": r["pack"], "wall_s": r.get("wall_s")} for r in slow],
        })
    return out


def _gate_timing_text(rows: list[dict]) -> str:
    lines = []
    for gate in _gate_timing(rows):
        lines.append(
            f"gate {gate['gate']}: {gate['rows']} rows, wall {gate['wall_s']}s, "
            f"model-load {gate['model_load_s']}s, time per task {gate['time_per_task_s']}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
