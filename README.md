# Anamnesis (ἀνάμνησις) — Cognitive Working Memory & Selective Attention for AI Agents

> *"Learning is not the acquisition of new information from scratch, but the recollection (anamnesis) of what the mind already holds in its latent space."* — Adapted from Plato's *Meno* and *Phaedo*.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)

**Anamnesis** is an open-source, ultra-lightweight cognitive working memory controller for AI agents and LLMs. Instead of naively stacking conversation history into giant token windows or relying on expensive recursive LLM summarizers, Anamnesis models human-like working memory dynamics using:

1. **Dual-Head Attention Anchors ($A_t$):** Exponential Moving Average (EMA) vectors that naturally track topic shifts and prevent centroid washout.
2. **Dynamic Intensity & Ebbinghaus Decay:** Automatic forgotten-curve fading with reinforcement boosts on reactivation.
3. **Pinned Invariants:** Immune preservation of critical system constraints, credentials, and safety directives.
4. **Symbolic Antonym Safeguards:** Hybrid vector-boolean filtering to eliminate contradiction blindspots.
5. **Zero Token Overhead:** Deterministic vector computation using local ONNX embeddings (`FastEmbed`) running in under 5ms.

---

## Architecture Overview

```
                      [ Incoming User/Agent Event ]
                                    │
                                    ▼
                      [ Local Embedder (FastEmbed) ] ──► (384-dim vector e_t)
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
 [ Micro Dialogue Anchor ]                             [ Macro Task Anchor ]
(alpha = 0.35, turn drift)                            (alpha = 0.03, intent lock)
         │                                                     │
         └──────────────────────────┬──────────────────────────┘
                                    ▼
                        [ Dual-Head Attention Score ]
                       S_i = 0.6 * sim_d + 0.4 * sim_t
                                    │
                                    ▼
                     [ Dynamic Context Partitioning ]
  ┌─────────────────────────────────┼─────────────────────────────────┐
  ▼                                 ▼                                 ▼
Active Tier (Pinned + I > 0.7)     Warm Tier (0.3 <= I <= 0.7)       Cold Tier (I < 0.3)
Verbatim injection in prompt       Key-value / single-line stubs     Evicted to SQLite / FAISS
```

---

## Repository Structure

```
anamnesis/
├── core/                         # Core algorithmic engine (dependency-minimal)
│   ├── anchor.py                 # Dual-Head EMA Attention Anchors
│   ├── intensity.py              # Temporal decay, boost signals & pinned invariants
│   ├── embedder.py               # Local FastEmbed (ONNX) wrapper (384-dim)
│   ├── store.py                  # SQLite / SQLite-VSS vector persistence
│   └── controller.py             # Memory orchestrator & prompt assembler
│
├── integrations/                 # Turnkey deployment wrappers
│   ├── claude_code_hook.py       # Deterministic UserPromptSubmit hook
│   ├── mcp_server.py             # Standard Model Context Protocol server
│   ├── litellm_proxy.py          # Transparent HTTP interceptor middleware
│   └── browser_extension/        # Local-first content script for ChatGPT/Claude web
│
├── telemetry/                    # Local token savings & TTFT performance tracker
│   └── metrics.py
│
├── tests/                        # Full regression & mitigation verification suite
│   └── test_memory.py
│
├── requirements.txt
└── README.md
```

---

## Quickstart

### Installation
```bash
git clone https://github.com/momo590/anamnesis.git
cd anamnesis
pip install -r requirements.txt
```

### Basic Python Usage
```python
from core.controller import AnamnesisController

# Initialize controller with local 384-dim encoder
memory = AnamnesisController(dim=384)

# 1. Register a pinned critical rule (immune to decay)
memory.add_block(
    block_id="rule_01",
    content="MANDATORY: Verify customs certificate for EU-Africa freight.",
    vector=...,
    pinned=True
)

# 2. Simulate conversation turns
memory.step_interaction(event_text="Check customs in Dakar", event_vector=...)

# 3. Assemble pruned prompt within a strict token budget
context_prompt = memory.assemble_prompt_context(max_tokens=1500)
print(context_prompt)
```

---

## Verification & Tests

Run the test suite to verify centroid stability, rule survival, and adaptive alpha performance:

```bash
python -m unittest tests/test_memory.py
```

---

## Philosophy & Origin
The project is rooted in Platonic epistemology: the mind does not ingest knowledge by infinite accumulation, but surfaces latent truths through structured recollection (*Anamnesis*). In AI architectures, context management is not about remembering everything, but about mastering **the art of selective forgetting**.

---

## License
MIT License.
