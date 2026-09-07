# Anamnesis — Cognitive Working Memory & Selective Attention for AI Agents

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)

**Anamnesis** is an open-source, ultra-lightweight working memory controller for AI agents and LLMs. 

Unlike long-term memory databases or brute-force token buffers, Anamnesis models **cognitive RAM**: it provides real-time attention tracking, selective forgetting, and deterministic prompt pruning directly on the local execution path without consuming LLM inference tokens.

---

## The Problem: Storage (Hard Drive) vs Working Memory (RAM)

Most memory frameworks for agents (Mem0, Letta/MemGPT, Zep) tackle **Long-Term Memory**. They act like **hard drives**:
- At each conversation turn, an auxiliary LLM extracts factual statements (*"user prefers dark mode"*), storing them into a persistent graph or vector database.
- Every retrieval relies on static semantic search or keyword matching, incurring external latency (2–4 seconds) and accumulating recurring API costs.

Meanwhile, within active agent sessions, practitioners face the **limits of giant context windows**:
1. **The "Lost-in-the-Middle" Phenomenon:** As context windows swell to 100k+ tokens, model attention degrades and subtle instructions are overlooked.
2. **Cost and Latency:** Passing massive conversation histories at every turn explodes Time-To-First-Token (TTFT) and API bills.
3. **The Summarization Dilemma:** Recursive LLM summarization causes latency bottlenecks, doubles token overhead, and suffers from progressive information erosion.

---

## The Solution: Real-Time Attention Dynamics & Selective Forgetting

Anamnesis operates as **Cognitive RAM**. Instead of storing static text cards or dumping raw chat history, it mathematically tracks continuous attention drift and prunes obsolete context in sub-millisecond local execution.

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
Verbatim injection in prompt       Key-value / single-line stubs     Evicted from prompt budget
```

### Core Architecture

1. **Dual-Head Attention Anchors ($A_t$):**
   - **Micro Dialogue Anchor ($\alpha = 0.35$):** Glides rapidly to follow short-term conversational turns.
   - **Macro Task Anchor ($\alpha = 0.03$):** Anchors the long-term project boundaries, preventing centroid washout during digressions.
   $$A_{t+1} = \alpha \cdot \mathbf{e}(x_{t+1}) + (1 - \alpha) \cdot A_t$$

2. **Temporal Decay & Intensity Boosts:**
   Every memory item has an activation score $I \in [0.0, 1.0]$. Unused context fades smoothly turn-by-turn ($-\lambda \cdot \Delta t$), while active references receive immediate boosts.

3. **Pinned Invariants:**
   Critical instructions (safety rules, credentials, mandatory workflows) are flagged `pinned=True`. Their intensity remains locked at $1.0$, guaranteeing permanent prompt priority immune to decay.

4. **Hybrid Symbolic & Vector Filtering:**
   Antonym and negation blindspots are eliminated: deprecated or canceled orders are deterministically filtered by status, avoiding vector-similarity misfires.

5. **Deterministic Zero-Token Computation:**
   All vector updates and ranking run in under 2ms using local NumPy calculations. No LLM calls are needed to manage memory.

---

## Comparison: Where Anamnesis Fits

| Dimension | Long-Term Memory (Mem0, Letta) | Naive Context Stacking | Anamnesis Working Memory |
| :--- | :--- | :--- | :--- |
| **System Role** | Hard Drive (Cross-session facts) | Dumb Buffer (FIFO) | **Cognitive RAM (Active focus)** |
| **Memory Ingestion Cost** | 1 auxiliary LLM call / turn | 0 calls (token bloat) | **0 LLM calls (Local vector math)** |
| **Attention Management** | Static semantic retrieval | None (Lost-in-the-Middle) | **Dual-Head EMA drift tracking** |
| **Critical Constraints** | Prone to dilution | Prone to truncation | **Pinned slots (Guaranteed survival)** |
| **Turn Latency Overhead** | 1000ms – 3000ms | 0ms (high TTFT upstream) | **< 2ms local execution** |

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
import numpy as np
from core.controller import AnamnesisController

# Initialize controller with 384-dimensional vector space
memory = AnamnesisController(dim=384, decay_rate=0.04)

# 1. Register an immutable pinned constraint (immune to decay)
v_rule = np.zeros(384, dtype=np.float32)
v_rule[0] = 1.0
memory.add_block(
    block_id="rule_customs_01",
    content="MANDATORY: Verify EUR.1 origin certificate on all freight.",
    vector=v_rule,
    pinned=True
)

# 2. Track conversational turns
v_turn = np.zeros(384, dtype=np.float32)
v_turn[0] = 0.8
memory.step_interaction(event_text="Check customs rates in Dakar", event_vector=v_turn)

# 3. Assemble prompt context within token budget
prompt_context = memory.assemble_prompt_context(max_tokens=600)
print(prompt_context)
```

---

## Integrations

- **Claude Code CLI:** Seamless pre-prompt injection via `integrations/claude_code_hook.py` using `UserPromptSubmit`.
- **Claude Desktop & Cursor:** Zero-latency MCP tools (`get_working_memory`, `record_memory`) in `integrations/mcp_server.py`.
- **LiteLLM / Team Proxy:** Transparent reverse-proxy context pruning in `integrations/litellm_proxy.py`.
- **Local Telemetry:** Real-time token savings and latency analytics in `telemetry/metrics.py`.

---

## Verification Suite

Anamnesis includes an automated test suite verifying core stability guarantees:

```bash
python -m unittest tests/test_memory.py -v
```

1. `test_pinned_rules_survive_decay_cycles`: Verifies that critical constraints never fade, even across 25+ unrelated turns.
2. `test_dual_head_anchor_prevents_centroid_washout`: Proves that task boundaries stay intact during sudden topic switches.
3. `test_symbolic_filter_blocks_deprecated_and_antonyms`: Confirms that canceled directives are deterministically excluded.

---

## License
MIT License.
