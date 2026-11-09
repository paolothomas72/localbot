"""Recount every score in the 9 Nov 2026 report from tries.csv and compare with scores.csv.

Rules (the same as the test runner):
- a try with harness_error=1 never counts as a pass or a fail;
- a task with 2 or more harness-error tries (of 3) has no result and is left out;
- otherwise a task passes when at least 2 of its 3 tries passed (proofs: 1 try, it must pass);
- the first-try score uses try 1 only, and only when try 1 had no harness error.
Run from the repository's top folder:  python docs/reports/2026-11-09/recount.py
"""
import csv, os
from collections import defaultdict
here = os.path.dirname(os.path.abspath(__file__))
tasks = defaultdict(list)
with open(os.path.join(here, "tries.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        tasks[(r["source"], r["run"], r["model"], r["test_set"], r["task"])].append(r)
got = defaultdict(lambda: [0, 0, 0, 0])  # passed, scored, planned, first-try passed
for (source, run, model, pack, task), tries in tasks.items():
    g = got[(source, run, model, pack)]
    g[2] += 1
    errors = sum(t["harness_error"] == "1" for t in tries)
    need = 1 if source == "proof" else 2
    if errors >= need:
        continue
    g[1] += 1
    passes = sum(t["passed"] == "1" and t["harness_error"] == "0" for t in tries)
    g[0] += passes >= need
    first = [t for t in tries if t["try"] == "1" and t["harness_error"] == "0"]
    g[3] += bool(first and first[0]["passed"] == "1")
differences = 0
with open(os.path.join(here, "scores.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        key = (r["source"], r["run"], r["model"], r["test_set"])
        want = [int(r["passed"]), int(r["scored_tasks"]), int(r["planned_tasks"]), int(r["first_try_passed"])]
        if got.get(key) != want:
            differences += 1
            print("DIFFERENT:", key, "recounted", got.get(key), "published", want)
print(f"{len(tasks)} tasks, {sum(len(v) for v in tasks.values())} tries recounted, {differences} differences")
