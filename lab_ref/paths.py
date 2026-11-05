"""Single root for the local-model lab. Import this from any script."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIB = ROOT / "lib"
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "config"
HARNESS = ROOT / "harness"
TICKETS = ROOT / "tickets"
FINETUNE = ROOT / "finetune"
DOCS = ROOT / "docs"
EVAL = ROOT / "eval"
DATA = ROOT / "data"
GOLD = DATA / "gold"
SEEDS = DATA / "seeds"
HOLDOUT = DATA / "holdout"
RAW = DATA / "raw"
TEMPLATES = DATA / "templates"
LH_HOME = Path.home() / ".localharness"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))
