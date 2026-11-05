"""Unique 12GB-fit models. Skip spill, Q8, embeddings, and same-weight twins."""

from __future__ import annotations

# One tag per weight. Twins / spill stay out of interactive batteries.
FIT_MODELS: list[str] = [
    "granite4:tiny-h",
    "nemotron-3-nano:4b",
    "llama3.2:1b",
    "llama3.1:8b",
    "hermes3:8b",
    "hf.co/deepreinforce-ai/Ornith-1.0-9B-GGUF:latest",
    "qwen3-8b-ft-agentic-v1:latest",
    "qwen3:8b",
    "lfm2.5:8b",
    "qwen3-vl:8b",
    "gemma4:12b",
    "JetBrains/mellum2-thinking-q4_k_m:latest",
    "cogito:14b",
    "hermes4-14b-mvp64k:latest",
    "qwen2.5-mvp64k:latest",
    "gemma4:e4b",
    "apriel16-15b-mvp64k:latest",
]

HERMES_NATIVE: list[str] = [
    "hermes3:8b",
    "hermes4-14b-mvp64k:latest",
]

# Spill / Q8 / toy FT / embeddings / duplicate weights
SKIP: set[str] = {
    "gpt-oss:20b",
    "llama3.1:8b-instruct-q8_0",
    "llama31-8b-q8-test:latest",
    "llama32-1b-localharness-poc:latest",
    "nomic-embed-text:latest",
    "hermes3-8b-mvp64k:latest",
    "llama31-8b-mvp64k:latest",
    "mellum2-thinking-mvp64k:latest",
    "ServiceNow-AI/Apriel-1.6-15b-Thinker:Q4_K_M",
    "lfm25-8b-mvp64k:latest",
    "qwen3vl8b-mvp64k:latest",
    "qwen3-8b-mvp64k:latest",
    "cogito14b-mvp64k:latest",
    "cogito14b-syswrap:latest",
    "hf.co/bartowski/NousResearch_Hermes-4-14B-GGUF:Q4_K_M",
    "qwen3-8b-finetuned-test:latest",
}

# Advertised 64K is not the test budget. Probe these only.
CTX_CANDIDATES: list[int] = [8192, 16384, 32768]
DEFAULT_CTX = 8192
