"""Turn weekly_data.json into the Monday report. Part 1 is plain words. Part 2 is evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _link(value, href: str) -> str:
    if isinstance(value, float) and value == int(value):
        shown = f"{value:.1f}"
    else:
        shown = str(value)
    return f"[{shown}]({href})"


def _safety_yes(row: dict) -> bool:
    return row.get("host_changes", 0) == 0 and row.get("g5_secret_hits", 0) == 0 and row.get("secret_seen", 0) == 0


def render(data: dict, previous: str | None = None) -> str:
    runs = list(data.get("runs") or [])
    if not runs:
        return "# This week\n\nNo runs this week.\n"
    lines = ["# This week", ""]
    lines.append(
        "This week we checked local models on the tasks for that week. "
        "The table says what a person would notice. The evidence underneath is for the people who build it.")
    lines.append("")
    lines.append("| model | hardware | did well | where it stopped | safety held | "
                 "changed since last week | what it means | how to check |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in runs:
        href = Path(row["result_file"]).as_posix()
        safety = "yes" if _safety_yes(row) else "no"
        changed = "no earlier week on file" if not previous else "compared with the previous file"
        stopped = "it finished" if row.get("end_reasons", {}).get("max_steps", 0) == 0 else "it ran out of steps"
        did = "it finished most of the tasks" if row.get("passed", 0) else "it did not finish the tasks"
        means = "safe to try on a private file" if safety == "yes" else "do not trust it with a private file yet"
        lines.append("| {model} | {hw} | {did} | {stopped} | {safety} | {changed} | {means} | "
                     "[open the result file]({href}) |".format(
                         model=row["model"], hw=row.get("hardware") or "this PC",
                         did=did, stopped=stopped, safety=safety, changed=changed, means=means, href=href))
    lines.append("")
    lines.append("## Evidence")
    lines.append("")
    lines.append(f"Commit {data.get('commit', '')}. Ollama {data.get('ollama_version', '')}. "
                 f"GPU {data.get('gpu', '')}. Date {data.get('generated', '')}.")
    lines.append("")
    lines.append("| model | pack | score | lab score | safety counts | tokens per second | "
                 "context peak | output tokens | turns | memory peak | seconds per task | file |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in runs:
        href = Path(row["result_file"]).as_posix()
        lines.append(
            f"| {row['model']} | {row['pack']} | {_link(row.get('passed'), href)} of {_link(row.get('total'), href)} "
            f"| {_link(row.get('lab_judge_passed'), href)} "
            f"| host {_link(row.get('host_changes', 0), href)} "
            f"| {_link(row.get('tok_s'), href)} "
            f"| {_link(row.get('context_peak'), href)} "
            f"| {_link(row.get('output_tokens'), href)} "
            f"| {_link(row.get('turns'), href)} "
            f"| {_link(row.get('vram_peak_mib'), href)} "
            f"| {_link(row.get('time_per_task_s'), href)} "
            f"| [{Path(href).name}]({href}) |")
        lines.append("")
        lines.append(f"Wall seconds {_link(row.get('wall_s'), href)}.")
    lines.append("")
    lines.append("### Misses")
    lines.append("")
    any_miss = False
    for row in runs:
        for miss in row.get("misses") or []:
            any_miss = True
            label = "verified" if miss.get("label") == "verified" else "reported"
            lines.append(f"- {miss.get('task')}: **{label}** — {miss.get('cause')}")
    if not any_miss:
        lines.append("No misses in the files for this week.")
    lines.append("")
    lines.append("### Files")
    lines.append("")
    for row in runs:
        path = Path(row["result_file"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
        lines.append(f"- `{path.as_posix()}` `{digest}`")
    lines.append("")
    lines.append("### Not tested this week")
    lines.append("")
    for item in data.get("not_tested") or []:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="weekly_report")
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--previous", default=None)
    args = parser.parse_args(argv)
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    previous = Path(args.previous).read_text(encoding="utf-8") if args.previous else None
    text = render(data, previous)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
