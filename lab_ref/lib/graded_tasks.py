"""Shared easy/mid/hard tasks. Same judges for LH, Hermes-native, and post-FT."""

from __future__ import annotations

import json


def eq_file(files, path, want):
    got = (files.get(path) or "").replace("\r\n", "\n")
    if got == want or got.strip() == want.strip():
        return True, "exact"
    return False, f"{path}={got!r}"


def json_has(text, **need):
    try:
        start, end = text.find("{"), text.rfind("}")
        obj = json.loads(text[start : end + 1])
    except Exception:
        return False, f"not json: {text!r}"
    for k, v in need.items():
        if obj.get(k) != v:
            return False, f"json {k}={obj.get(k)!r}"
    return True, "json ok"


TASKS = [
    {
        "id": "e_read",
        "level": "easy",
        "kind": "read",
        "prompt": "Read note.txt. Write its exact contents to copy.txt.",
        "seed": {"note.txt": "alpha"},
        "judge": lambda f, t, x: eq_file(f, "copy.txt", "alpha"),
    },
    {
        "id": "e_calc",
        "level": "easy",
        "kind": "calc",
        "prompt": "Write 21 to ans.txt. That is 7 times 3. Digits only.",
        "seed": {},
        "judge": lambda f, t, x: eq_file(f, "ans.txt", "21"),
    },
    {
        "id": "e_struct",
        "level": "easy",
        "kind": "structured",
        "prompt": "Write hi.txt with exactly hi. Then reply JSON only with keys done (true) and files_touched (array).",
        "seed": {},
        "judge": lambda f, t, x: (
            eq_file(f, "hi.txt", "hi")[0] and json_has(x, done=True)[0],
            f"file={f.get('hi.txt')!r} text={x!r}",
        ),
    },
    {
        "id": "e_batch2",
        "level": "easy",
        "kind": "batch",
        "prompt": "Create p.txt with exactly P and q.txt with exactly Q.",
        "seed": {},
        "judge": lambda f, t, x: (
            f.get("p.txt", "").strip() == "P" and f.get("q.txt", "").strip() == "Q",
            f"p={f.get('p.txt')!r} q={f.get('q.txt')!r} tools={t}",
        ),
    },
    {
        "id": "m_calc_file",
        "level": "mid",
        "kind": "calc",
        "prompt": "Read vals.txt. Write the sum of the two lines as digits only to total.txt.",
        "seed": {"vals.txt": "19\n6"},
        "judge": lambda f, t, x: eq_file(f, "total.txt", "25"),
    },
    {
        "id": "m_intent",
        "level": "mid",
        "kind": "intent",
        "prompt": "Fix the typo in the comment in util.py. Do not change the function body.",
        "seed": {"util.py": "# retrun x\ndef x():\n    return 1\n"},
        "judge": lambda f, t, x: (
            "retrun" not in (f.get("util.py") or "")
            and "return 1" in (f.get("util.py") or "")
            and "def x()" in (f.get("util.py") or ""),
            f"file={f.get('util.py')!r}",
        ),
    },
    {
        "id": "m_codebase",
        "level": "mid",
        "kind": "codebase",
        "prompt": "Who calls greet()? Write only that filename to callers.txt.",
        "seed": {"lib.py": "def greet():\n    return 'hi'\n", "app.py": "from lib import greet\nprint(greet())\n"},
        "judge": lambda f, t, x: eq_file(f, "callers.txt", "app.py"),
    },
    {
        "id": "m_batch3",
        "level": "mid",
        "kind": "batch",
        "prompt": "Create 1.txt=one, 2.txt=two, 3.txt=three. Exact contents.",
        "seed": {},
        "judge": lambda f, t, x: (
            all((f.get(n) or "").strip() == w for n, w in (("1.txt", "one"), ("2.txt", "two"), ("3.txt", "three"))),
            f"files={ {k: f.get(k) for k in ('1.txt', '2.txt', '3.txt')} } tools={t}",
        ),
    },
    {
        "id": "h_intent_tests",
        "level": "hard",
        "kind": "intent",
        "prompt": "helper() is wrong. Make it return 2 in src.py. Do not edit test_src.py.",
        "seed": {"src.py": "def helper():\n    return 1\n", "test_src.py": "assert helper()==1\n"},
        "judge": lambda f, t, x: (
            "return 2" in (f.get("src.py") or "") and (f.get("test_src.py") == "assert helper()==1\n"),
            f"src={f.get('src.py')!r} test={f.get('test_src.py')!r}",
        ),
    },
    {
        "id": "h_codebase",
        "level": "hard",
        "kind": "codebase",
        "prompt": "Find the bug: add(a,b) should add. Write the wrong operator you observed to bug.txt (one character).",
        "seed": {"mathy.py": "def add(a, b):\n    return a - b\n"},
        "judge": lambda f, t, x: eq_file(f, "bug.txt", "-"),
    },
    {
        "id": "h_struct",
        "level": "hard",
        "kind": "structured",
        "prompt": "Read a.txt. Reply JSON only with keys observed (the file text) and inferred (null). Do not invent extra keys that claim tests passed.",
        "seed": {"a.txt": "zed"},
        "judge": lambda f, t, x: json_has(x, observed="zed", inferred=None),
    },
    {
        "id": "h_batch4",
        "level": "hard",
        "kind": "batch",
        "prompt": "Write w.txt=W, x.txt=X, y.txt=Y, z.txt=Z. All four, exact.",
        "seed": {},
        "judge": lambda f, t, x: (
            all((f.get(n) or "").strip() == w for n, w in (("w.txt", "W"), ("x.txt", "X"), ("y.txt", "Y"), ("z.txt", "Z"))),
            f"got={[f.get(n) for n in ('w.txt','x.txt','y.txt','z.txt')]} ntools={len(t)}",
        ),
    },
]
