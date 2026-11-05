# Security policy

Local Bot runs commands written by a language model on your computer. It treats every model as
untrusted: the safety rules live in code, not in the prompt, and they are meant to hold whatever
the model or the prompt does. How well each model behaves is measured, never relied on.

## Status

Pre-alpha. Don't point it at a machine or a folder you can't afford to lose yet. Only the `main`
branch gets fixes.

## Reporting a vulnerability

Please don't open a public issue. Use GitHub's private vulnerability reporting instead: the
**Security** tab of this repository, then **Report a vulnerability**.

Useful to include: the model and its tag, the task or the files in the workspace, the tool call or
command, what crossed the line, and your OS and container runtime. Never send a real secret: put a
fake value in its place. There is no bug bounty. A fix gets a test that fails on the old code, and
the advisory says which version has it.

## In scope

- A model's command that changes anything outside its own workspace.
- A secret file's content ending up in another file, or leaving the workspace.
- A way out of the container sandbox, or past its limits (network, memory, processes, disk).
- Undo that loses or changes a file it should have restored.
- A tool call that the model only quoted (from a file or an earlier tool result) being run.
- A way to make Local Bot report success for something that didn't happen.

Out of scope: what a model says (its words aren't actions), a model that is simply bad at a task,
and attacks that need the attacker to already control your user account.

## What protects you

- **Container sandbox.** `bash` runs in Podman or Docker as a non-root user, with no network,
  CPU, memory and process limits, and only the bot's workspace mounted.
- **Workspace path layer.** The file tools stay inside the workspace (Windows path rules first:
  drive letters, UNC paths, short names, reserved names, links).
- **Guard.** Refuses copying secrets, destructive steps the task didn't ask for, and network use.
- **Undo.** A snapshot before each `bash`, `write` and `edit` step; `localbot undo <bot>` restores it.

## Known limits

- A secret file is recognised by its name, from a configurable list. A secret saved under a name
  the list doesn't know isn't treated as a secret.
- Tested on Windows 11 with Python 3.12, with models up to 20B on one 12 GB GPU. On Linux only
  the host tests run (CI). macOS is unsupported.
- `docs/EVIDENCE.md` lists what has been tested, with dates, and what hasn't.
