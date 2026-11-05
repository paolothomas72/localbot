#!/usr/bin/env python3
"""History totals of the test evidence, recomputed from docs/runs.json and the result files.

Usage:  python -I scripts/evidence_history.py [ROOT]

ROOT is a checkout holding docs/runs.json and results/stage1/ (default: the checkout this script sits
in). Prints Markdown: the section 4.1 totals, the final run, and one row per round. Stdlib only and
read-only: it writes nothing and runs nothing. A file the manifest names but the folder lacks stops it
(exit 2, nothing on stdout).

Counting rules (fixed; the tests pin the published numbers):
- counted files = every file in runs[].files; tries = sum over those files of n x total;
  model time = sum of wall_s; models = distinct "model"; Ollama version = run_env.ollama_version.
- thrown out = the length of the manifest's "void" list (never counted above).
- the final run = every run whose name or code holds 61fb534; its lanes are gpu_fit.lane.
- a round = the runs that share one "code" (the program version the files record), in manifest order.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

FINAL = "61fb534"


class HistoryError(Exception):
    pass


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise HistoryError(f"cannot read {path.name}: {e}") from None


def summarize(root: Path) -> dict:
    root = Path(root)
    manifest = _load(root / "docs" / "runs.json")
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise HistoryError("docs/runs.json holds no runs")
    seen: dict[str, dict] = {}
    files = tries = 0
    secs = 0.0
    models: set = set()
    ollama: dict = {}
    rounds: dict[str, dict] = {}
    final = {"code": FINAL, "files": 0, "tries": 0, "minutes_raw": 0.0, "lanes": {}}
    for run in runs:
        is_final = FINAL in str(run.get("run", "")) or FINAL in str(run.get("code", ""))
        rnd = rounds.setdefault(run.get("code", "?"), {"code": run.get("code", "?"), "runs": 0, "files": 0,
                                                     "tries": 0, "hours": 0.0})
        rnd["runs"] += 1
        for name in run.get("files", []):
            if name in seen:
                raise HistoryError(f"{name} is listed twice in docs/runs.json")
            path = root / "results" / "stage1" / name
            if not path.is_file():
                raise HistoryError(f"results/stage1/{name} is missing (listed in run {run.get('run')!r})")
            d = _load(path)
            seen[name] = d
            t = int(d["n"]) * int(d["total"])
            w = float(d["wall_s"])
            files += 1
            tries += t
            secs += w
            models.add(d["model"])
            v = (d.get("run_env") or {}).get("ollama_version")
            ollama[v] = ollama.get(v, 0) + 1
            rnd["files"] += 1
            rnd["tries"] += t
            rnd["hours"] += w / 3600
            if is_final:
                lane = (d.get("gpu_fit") or {}).get("lane") or "unknown"
                final["files"] += 1
                final["tries"] += t
                final["minutes_raw"] += w / 60
                final["lanes"][lane] = final["lanes"].get(lane, 0) + 1
    final["minutes"] = int(final["minutes_raw"] + 0.5)
    return {"files": files, "tries": tries, "hours": secs / 3600, "models": len(models),
            "void": len(manifest.get("void", [])), "ollama": ollama, "rounds": list(rounds.values()),
            "final": final}


def _hm(minutes: int) -> str:
    return f"{minutes // 60} h {minutes % 60:02d} min" if minutes >= 60 else f"{minutes} min"


def render(s: dict) -> str:
    known = sorted((k, n) for k, n in s["ollama"].items() if k is not None)
    parts = [f"{k} ({n} files)" if i == 0 else f"{k} ({n})" for i, (k, n) in enumerate(known)]
    if None in s["ollama"]:
        parts.append(f"not recorded ({s['ollama'][None]} files)")
    f = s["final"]
    lanes = ", ".join(f"{k} {f['lanes'][k]}" for k in ("gpu", "spill", "unknown") if k in f["lanes"])
    lanes += "".join(f", {k} {n}" for k, n in sorted(f["lanes"].items()) if k not in ("gpu", "spill", "unknown"))
    out = ["### 4.1 Totals", "",
           f"- **{s['files']}** counted test files, **{s['tries']:,}** tries, **{s['hours']:.1f} hours** of model "
           f"time, **{s['models']}** models.",
           f"- **{s['void']}** more files were thrown out (listed in `docs/runs.json`, never counted above).",
           f"- Ollama version: {', '.join(parts)}.", "",
           f"### Final run ({f['code']})", "",
           f"- {f['files']} files, {f['tries']:,} tries, {_hm(f['minutes'])} of model time.",
           f"- Graphics card use per file (`gpu_fit.lane`): {lanes}.", "",
           "### Rounds (runs that share one program version)", "",
           "| version | runs | files | tries | hours |", "| --- | --- | --- | --- | --- |"]
    out += [f"| {r['code']} | {r['runs']} | {r['files']} | {r['tries']:,} | {r['hours']:.1f} |"
            for r in s["rounds"]]
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        print("usage: evidence_history.py [ROOT]", file=sys.stderr)
        return 2
    root = Path(argv[0]) if argv else Path(__file__).resolve().parents[1]
    try:
        text = render(summarize(root))
    except (HistoryError, KeyError, TypeError, ValueError) as e:
        print(f"evidence_history: {e}", file=sys.stderr)
        return 2
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))