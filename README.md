# Local Bot

A personal agent for **local and open-weight models**. The aim: measure the model you load, then fit
the harness to what that model can really do, so each model works as a safe agent on your own
hardware. No account, no cloud.

**Status: pre-alpha.** Stage 1, the exam and the safety foundation, is built. There is no
`localbot run` yet, so you can't hand it a task of your own. What exists today:

- an agent loop that reads native tool calls, and text tool calls in the Qwen, Hermes and older
  spellings, for models that need them;
- `read`, `write`, `edit` and `bash` tools, with `bash` in a Podman or Docker sandbox, a workspace
  path layer, a guard and a snapshot before every step;
- `localbot undo <bot> [steps]`, which puts a bot's workspace back as it was before a step;
- a disk exam that runs task packs (work, trust, privacy) against Ollama models and saves redacted
  results.

What was tested, with dates and what is still open: [`docs/EVIDENCE.md`](docs/EVIDENCE.md).
How to report a security problem: [`SECURITY.md`](SECURITY.md).

## Requirements

- Windows 11 (tested). Ubuntu 24.04: the host tests run in CI; containers and models aren't
  tested there yet. macOS is unsupported for now.
- Python 3.12.
- [Ollama](https://ollama.com) for models, and Podman or Docker for the `bash` tool.

## Install from source

```
git clone <this repository>
cd localbot
python -m venv .venv
.venv\Scripts\activate            # Windows (Linux: source .venv/bin/activate)
python -m pip install -r requirements-dev.txt
python -m pip install -e .
localbot --help
```

## Run the tests

```
python -m pytest -m "not container" -q -rs -p no:cacheprovider
```

That is the host suite: no container, GPU or model. CI runs it on Windows and Ubuntu. The
container tests and model runs are in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Run the exam

```
python -m localbot.exam.runner --pack privacy --profile core4 --n 3 <ollama-model-tag>
```

`--pack work|trust|privacy|all`; `--profile core4` is the default tool set with the container
shell and undo, and `parity_lab6` reproduces the lab loop the scores are compared with. Each try
runs in a fresh folder that is deleted after its result is saved; secrets in results are redacted.

## Where Local Bot keeps its state

Undo snapshots and bot workspaces live outside this folder: in `LOCALBOT_STATE_DIR` if you set it,
else `%LOCALAPPDATA%\localbot` on Windows and `$XDG_STATE_HOME/localbot` (default
`~/.local/state/localbot`) on Linux.

## Test evidence

How Local Bot was tested, every round from the first day to the final run, with the numbers and the
commands to check them yourself: [docs/EVIDENCE.md](docs/EVIDENCE.md).

Weekly test report #1 (9 Nov 2026): [docs/reports/2026-11-09/REPORT.md](docs/reports/2026-11-09/REPORT.md)

## Principles

- Evidence first: real command output, n=3 tries per task, "verified" kept apart from "reported".
- Safety lives in code, not in the prompt.
- Foundation before features; the interface comes last.

## License

MIT, see [`LICENSE`](LICENSE). The text tool-call parser is vendored from
[LocalHarness](https://github.com/ahwurm/localharness) 0.22.1 (MIT); see `localbot/NOTICE`.
