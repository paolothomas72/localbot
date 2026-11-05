# lab_ref — frozen lab code (do not edit)

Byte-identical copy of the files Stage 1 reuses, taken 2026-10-26 from
the owner's private `local-lab` repository (git commit `5668bd2`, plus its working tree on that day).

Why it exists: the September scores were produced by this exact code. Keeping it frozen means the baseline can be re-run at any time and compared fairly.

- `paths.py`, `lib/`, `scripts/` — the lab's own exam loop, guards, task packs and runners.
- `results_sept/` — the September 27–29 score files Local Bot must match or beat. These 41 files are held back from this public copy: they hold local paths from the owner's PC.
- `MANIFEST.sha256` — SHA-256 of the files here. In this public copy it lists only the files the copy holds (not the held-back `results_sept/` files). If a line stops matching, something edited the copy.

Re-run the baseline from inside this folder (`cd lab_ref`), commands in `docs/STAGE1_SPEC.md` §3. The scripts write their output into `lab_ref/data/raw/`, which git ignores. Copy the result JSON into `results/stage1/baseline/`.

New code never imports from here at runtime, except the disk exam runner, which reads the task definitions in `lib/*_tasks.py` read-only.
