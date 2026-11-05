"""G3 comparison (docs/STAGE1_SPEC.md §8, S6): today's baseline vs Local Bot parity_lab6.

    python scripts/compare_parity.py [REPO] [--model MODEL ...]

Baseline = results/stage1/baseline/lh_{product,safety,privacy}_*.json (the lab scripts, unchanged);
parity  = results/stage1/{work,trust,privacy}_parity_lab6_<model>_<stamp>.json (latest per pack and
model). Prints markdown: per task baseline x/3 vs parity lab-judge x/3 vs parity disk x/3, the
pack-level counts (call_source, missed text calls, vetoes, end reasons, no_call_reason,
negative-check hits, host changes, secret_seen, infra re-runs, harness errors, path-layer refusals
the lab dict would have written as a key = disk-vs-dict, gpu_fit, VRAM, tok/s), and for every task
whose numbers differ, each try's reason, tools, vetoes and calls. Secrets are redacted again
before printing (the files are already redacted).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

LABPFX = {"work": "lh_product_", "trust": "lh_safety_", "privacy": "lh_privacy_"}
RED = re.compile(r"CANARY_[A-Za-z0-9_-]+|DECOY_[0-9a-f]+|sk-[A-Za-z0-9-]+")
# Path-layer refusals that are new on disk (the lab dict would have stored the name as a key).
DISK_ONLY = ("refused control or reserved character", "refused path component ending in",
             "refused reserved device name", "refused 8.3 short-name", "refused ':' in path component",
             "refused hidden path component", "refused reparse point", "refused environment-variable",
             "refused empty or '.' path component")


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def red(s) -> str:
    return RED.sub("[R]", str(s))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", nargs="?", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--model", action="append", default=None)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):  # Windows console is cp1252; model text is not
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = Path(a.repo)
    base: dict[tuple[str, str], tuple[str, dict]] = {}
    for f in sorted((root / "results/stage1/baseline").glob("lh_*.json")):
        pack = next(k for k, v in LABPFX.items() if f.name.startswith(v))
        for m, row in load(f)["models"].items():
            base[(pack, m)] = (f.name, row)
    par: dict[tuple[str, str], tuple[str, dict]] = {}
    for f in sorted((root / "results/stage1").glob("*_parity_lab6_*.json")):
        d = load(f)
        par[(d["pack"], d["model"])] = (f.name, d)  # sorted by name = by stamp: latest wins
    models = a.model or sorted({m for _, m in par})
    for m in models:
        for pack in LABPFX:
            if (pack, m) not in par or (pack, m) not in base:
                continue
            pname, d = par[(pack, m)]
            bname, b = base[(pack, m)]
            report(pack, m, bname, b, pname, d)
    return 0


def report(pack: str, m: str, bname: str, b: dict, pname: str, d: dict) -> None:
    print(f"\n#### {pack.upper()} — {m}  (baseline `{bname}` vs parity `{pname}`)\n")
    print("| task | baseline today (lab loop, dict) | parity lab-judge (disk) | parity disk_pass | difference |")
    print("| --- | --- | --- | --- | --- |")
    for tid, bt in b["tasks"].items():
        pt = d["tasks"][tid]
        diff = []
        if pt["lab_judge_majority"] != bt["majority"]:
            diff.append("judge x/3 differs")
        if pt["majority"] != pt["lab_judge_majority"]:
            diff.append("disk checks differ")
        cell = "**" + "; ".join(diff) + "**" if diff else ""
        print(f"| {tid} | {bt['majority']} | {pt['lab_judge_majority']} | {pt['majority']} | {cell} |")
    print(f"| **score** | **{b['passed']}/{b['total']}** | **{d['passed_lab_judge']}/{d['total']}** | "
          f"**{d['passed']}/{d['total']}** | |")
    trials = [t for tt in d["tasks"].values() for t in tt["trials"]]
    cs = Counter(c for t in trials for c in t["call_source"])
    tca = sum(len(t["text_call_attempts"]) for t in trials)
    reasons = Counter(t["end_reason"] for t in trials)
    notes = Counter(t["end_note"] for t in trials if t.get("end_note"))
    nocall = Counter(t["no_call_reason"] for t in trials if t["no_call_reason"])
    neg = sum(1 for t in trials if not t["negative"]["ok"])
    veto = Counter(v.split(". ")[0][:60] for t in trials for v in t["veto_texts"])
    disk_only = [(tid, t["k"], c["args"].get("path"), c["result"][:90]) for tid, tt in d["tasks"].items()
                 for t in tt["trials"] for c in t.get("calls", []) if any(x in c["result"] for x in DISK_ONLY)]
    toks = [t["telemetry"]["tok_s"] for t in trials if t["telemetry"]["tok_s"]]
    vram = max((t["telemetry"].get("vram_used_mib_peak") or 0) for t in trials)
    walls = [t["telemetry"]["wall_s"] for t in trials]
    g = d["gpu_fit"]
    print(f"\n- tries {len(trials)}; call_source {dict(cs)}; text-call attempts (lab would miss) {tca}")
    print(f"- vetoes {sum(veto.values())}: {dict(veto.most_common())}")
    print(f"- end reasons {dict(reasons)}{'; notes ' + str(dict(notes)) if notes else ''}; no_call_reason {dict(nocall)}")
    print(f"- host_change {d['host_changes']}; negative-check hits {neg}; secret_seen {d['secret_seen']}; "
          f"infra re-runs {d['infra_reruns']}; harness errors {d.get('harness_errors', 0)}")
    print(f"- disk-vs-dict path refusals (tries where the lab would have written the name as a key): "
          f"{len({(x[0], x[1]) for x in disk_only})} tries / {len(disk_only)} calls"
          + "".join(f"\n  - {tid} t{k}: path={p!r} → {r}" for tid, k, p, r in disk_only))
    print(f"- gpu_fit {g['lane']} ({g['ps_says']}; {'; '.join(g['reasons']) or 'no disagreement'}); "
          f"timing_comparable {d['timing_comparable']}; VRAM before {d['vram_before_mib']} MiB, peak {vram} MiB; "
          f"tok/s mean {sum(toks) / max(1, len(toks)):.1f}; task wall mean {sum(walls) / max(1, len(walls)):.1f}s; "
          f"pack wall {d['wall_s']}s")
    for tid, bt in b["tasks"].items():
        pt = d["tasks"][tid]
        if pt["lab_judge_majority"] == bt["majority"] and pt["majority"] == pt["lab_judge_majority"]:
            continue
        print(f"\n  {tid}:")
        for i, s in enumerate(bt["trials"], 1):
            print(f"    base t{i}: pass={s['pass']} reason={red(s['reason'])[:110]} tools={s.get('tools')}")
        for t in pt["trials"]:
            print(f"    par  t{t['k']}{t['tag']}: disk={t['pass']} judge={t['lab_judge_pass']} "
                  f"reason={red(t['judge_reason'])[:110]} end={t['end_reason']} vetoes={t['vetoes']}")
            for c in t.get("calls", []):
                print(f"        {c['name']} {red(json.dumps(c['args'], ensure_ascii=False))[:120]} -> {red(c['result'])[:90]!r}")


if __name__ == "__main__":
    raise SystemExit(main())
