# Code files not covered by `docs/CODE_61fb534.sha256`

`docs/CODE_61fb534.sha256` fingerprints every file under `localbot/`, `scripts/` and `lab_ref/`
as it was stored at commit `61fb534`; check it from the folder this file sits in with
`sha256sum -c docs/CODE_61fb534.sha256`. The files below are in this export but are not in
that list.

## New since `61fb534` (the file did not exist then)

- `localbot/chat.py`: C3: ``localbot chat``: a terminal chat with a bot that works in its own folder.
- `scripts/chat_ladder.py`: C6 part C, C7 part C: find the limits step by step. Runs the REAL ``localbot chat`` (real Ollama, real container,
- `scripts/evidence_history.py`: History totals of the test evidence, recomputed from docs/runs.json and the result files.
- `scripts/install_command.ps1`: added after the base commit (no description in the file)
- `scripts/model_queue.py`: C9: overnight model queue — download a batch, quick-check, full Stage 1 packs + ladder, resume, summarize, delete.
- `scripts/queue_status.py`: queue_status: one plain line per model of a running (or finished) model queue. Read only.

## Changed since `61fb534` (the file existed, its bytes differ)

- `lab_ref/MANIFEST.sha256`: sha256 cf8ed26f6847d86827bb49e6edcecd6edca644240243ec93df63d0a2bacff6cf; written by the export: lists only the files in this copy
- `lab_ref/README.md`: sha256 9cb65382d3698189d1803f57bd5b747ebb0b724828ce17556b4765605399f566; written by the export: says `results_sept/` is held back from this copy
- `localbot/__main__.py`: sha256 e23c735a1d258895213ca0acf04c60223f003da8e6394888157805cb3e0988a0
- `localbot/exam/runner.py`: sha256 c46b1ca990462aeec98667f50aaac20bd037f3ab6cf66bc78b848c8066a69ab8
- `localbot/guard/delete.py`: sha256 0bb7120fca4dc87da6aa95b5b9c68089b1db1d1652b0f77ffa53b811ae6928e4
- `localbot/loop.py`: sha256 212c9ce17b5f92ba014974f37458490cc70fef6f9b75a750c17fe653c4f52fb2
- `localbot/provider/ollama.py`: sha256 d35d4312204a034a4404ae525cda09403598492a108af635cddc6dfaf2db9abe
- `localbot/watchdog.py`: sha256 ed3f4ad9166786e2258bf268ec6ae2269d0978e93fdc1100a86bbd205a0f0a90
- `scripts/gate_run.py`: sha256 0de166ba0fedcee67f787aa27250b535f6411f79d8361d4334a1faa84b947e93
