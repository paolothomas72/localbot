"""Local stand-ins for the two LocalHarness types that fn_call.py imports.

Copied from LocalHarness 0.22.1 src/localharness/core/types.py (commit 42a58bc, MIT, see
localbot/NOTICE). Only ToolCall and ToolSchema are needed; the rest of that module
(provenance, messages) depends on LocalHarness internals and is not used here.
"""
from dataclasses import dataclass
from typing import Any

# Tool schema — JSON Schema format as expected by OpenAI API
ToolSchema = dict[str, Any]


# Parsed tool call from model response
@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = ""
