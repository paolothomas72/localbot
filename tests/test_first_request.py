"""S8f.2: the first /api/chat request of ``parity_lab6`` equals the lab loop's, byte for byte.

Both sides are captured by ``scripts/first_request.py`` along their real code paths, with
``urllib.request.urlopen`` replaced by a recorder that refuses every request (and every socket
connect refused), so no model is called and nothing leaves the process:

- Local Bot: ``localbot.exam.runner.run_pack`` → ``run_task`` → ``OllamaProvider`` (the runner's
  own wiring, as ``python -m localbot.exam.runner`` builds it).
- lab: ``lab_ref/scripts/run_lh_product.py`` (work, ``--slim``), ``run_lh_safety.py`` (trust),
  ``run_lh_bands.py --privacy`` (privacy): each script's ``run_trial`` → ``loop`` →
  ``ollama_chat.chat``. Imported read-only, bytecode off; nothing is written into lab_ref/.

Each task's first request is refused, so each task ends there: one captured request per task.
The only allowed difference is the URL host (``DOCUMENTED`` in the script, with its reason).
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from localbot.exam.runner import load_pack

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "first_request.py"
MODEL = "hf.co/deepreinforce-ai/Ornith-1.0-9B-GGUF:latest"
PACKS = ("work", "trust", "privacy")
CASES = [(p, t["id"]) for p in PACKS for t in load_pack(p)]


def _script():
    spec = importlib.util.spec_from_file_location("first_request_s8f2", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _capture(tmp: Path, name: str, *args: str) -> dict:
    out = tmp / f"{name}.json"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    r = subprocess.run([sys.executable, "-I", str(SCRIPT), "capture", *args, "--pack", "all",
                        "--model", MODEL, "--out", str(out)],
                       capture_output=True, text=True, encoding="utf-8", env=env, cwd=REPO, timeout=300)
    assert r.returncode == 0, r.stderr[-4000:]
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def caps(tmp_path_factory) -> dict:
    tmp = tmp_path_factory.mktemp("first_request")
    return {"localbot": _capture(tmp, "localbot", "--side", "localbot", "--profile", "parity_lab6"),
            "lab": _capture(tmp, "lab", "--side", "lab")}


@pytest.mark.parametrize("pack,task", CASES, ids=[f"{p}-{t}" for p, t in CASES])
def test_parity_first_request_equals_lab(caps, pack, task):
    fr = _script()
    ours, lab = fr.request_for(caps["localbot"], pack, task), fr.request_for(caps["lab"], pack, task)
    diffs = fr.diff_requests(ours, lab)
    assert fr.undocumented(diffs) == [], fr.format_diffs(diffs)
    assert ours["body_sha256"] == lab["body_sha256"]


def test_capture_is_one_refused_request_per_task(caps):
    for side, cap in caps.items():
        for pack in PACKS:
            rows = cap["packs"][pack]
            assert [r["task"] for r in rows] == [t["id"] for t in load_pack(pack)], (side, pack)
            for r in rows:
                assert r["url"].endswith("/api/chat") and r["method"] == "POST", (side, r["task"])
                assert r["user_is_prompt"] is True, (side, r["task"])
                assert r["end"] == cap["refused_end"], (side, r["task"], r["end"])
        # Every connect was refused; the only ones tried are loopback probes (Local Bot: :8090).
        assert all(any(h in c for h in ("127.0.0.1", "::1", "localhost")) for c in cap["connects_refused"]), side


def test_lab_side_ran_slim_and_localbot_side_ran_this_repo(caps):
    assert caps["lab"]["code"]["system_is_slim"] is True
    code = caps["localbot"]["code"]
    assert Path(code["localbot_file"]).resolve().is_relative_to(REPO.resolve())
    assert caps["localbot"]["profile"] == "parity_lab6"


def test_capture_files_hold_no_test_secret(caps):
    # s_no_secret's prompt carries the lab's fake canary: the file shows it redacted, never raw.
    for cap in caps.values():
        text = json.dumps(cap)
        assert "CANARY_pw_" not in text
    row = _script().request_for(caps["lab"], "trust", "s_no_secret")
    assert row["redactions"] > 0 and "[REDACTED]" in row["body_text"]


def test_body_text_is_the_wire_bytes_when_nothing_redacted(caps):
    row = _script().request_for(caps["lab"], "trust", "s_clean_list")
    assert row["redactions"] == 0
    assert hashlib.sha256(row["body_text"].encode("utf-8")).hexdigest() == row["body_sha256"]
    assert json.loads(row["body_text"]) == row["body"]


def _mutated(row: dict, fn) -> dict:
    body = copy.deepcopy(row["body"])
    body = fn(body) or body
    text = json.dumps(body)
    return {**row, "body": body, "body_text": text, "body_sha256": hashlib.sha256(text.encode()).hexdigest()}


def test_diff_names_each_kind_of_difference(caps):
    """Negative control: the comparator is not vacuous."""
    fr = _script()
    base = fr.request_for(caps["lab"], "trust", "s_clean_list")
    assert fr.diff_requests(base, base) == []

    def fields(other):
        return {d["field"] for d in fr.diff_requests(base, other)}

    def swap_tools(b):
        b["tools"][3], b["tools"][5] = b["tools"][5], b["tools"][3]

    def system(b):
        b["messages"][0]["content"] += " "

    def seed(b):
        b["options"]["seed"] = 1

    def think(b):
        b["think"] = False

    def describe(b):
        b["tools"][5]["function"]["description"] = "Run bash."

    def reorder(b):
        return {k: b[k] for k in reversed(list(b))}

    assert "tools.order" in fields(_mutated(base, swap_tools))
    assert "messages[0].content" in fields(_mutated(base, system))
    assert "options.seed" in fields(_mutated(base, seed))
    assert "body.think" in fields(_mutated(base, think))
    assert "tools[5].description" in fields(_mutated(base, describe))
    assert fields(_mutated(base, reorder)) == {"body.bytes"}
    host = {**base, "url": base["url"].replace("localhost", "127.0.0.1")}
    assert fr.undocumented(fr.diff_requests(base, host)) == []
    path = {**base, "url": base["url"].replace("/api/chat", "/api/generate")}
    assert [d["field"] for d in fr.undocumented(fr.diff_requests(base, path))] == ["url.path"]
