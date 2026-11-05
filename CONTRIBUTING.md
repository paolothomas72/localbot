# Contributing

Thanks for looking. Local Bot is pre-alpha and changes fast: open an issue before a big change.
Security problems go through private reporting, never an issue (see `SECURITY.md`).

## Set up

Python 3.12. Optional: Ollama for models, Podman and/or Docker for the `bash` tool.

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt      # Windows
.venv/bin/python -m pip install -r requirements-dev.txt          # Linux
```

## Tests

- **Host tests** (no container, GPU or model): `python -m pytest -m "not container" -q -rs -p no:cacheprovider`.
  CI runs these on Windows and Ubuntu for every push and pull request.
- **Container tests**: `python -m pytest -m container -rs -p no:cacheprovider` with
  `LOCALBOT_TEST_RUNTIMES=podman`, then again with `=docker`. Run them one at a time and never
  next to an exam run: they share one container runtime, and some tests push it to its limits.
- **Model runs** need Ollama and a GPU, and are never part of CI.

## Rules for a change

1. **A failing test first**, then the fix, then the test passing. A safety fix comes with a test
   that fails on the old code.
2. **Evidence is real output.** Say what you ran and what it printed (the exact result line).
   Keep "verified" (re-run, dated) apart from "reported" (someone said so).
3. **Safety lives in code, not in the prompt.** Don't fix a safety bug by asking the model to behave.
4. **Frozen copies stay frozen.** Never edit `lab_ref/` or `third_party/`; `lab_ref/MANIFEST.sha256`
   holds their hashes. Code copied from there gets a malformed-input test before it is trusted.
5. **Standard library first.** A new runtime dependency needs a reason in the pull request.
6. **LF line endings**, Python 3.12.
7. **No real data.** Never put a real secret, key, personal path or private file in a test, an
   issue, a log or a result file. Use fake values.

## Commit messages

A short summary line, then a body that says what changed and what was tested, with the exact
result line (for example `412 passed in 61s`). If a test was skipped or a runtime wasn't running,
say so.
