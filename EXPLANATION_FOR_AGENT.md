# Comprehensive Agent Briefing & Architecture Blueprint: From Ada Memory to Anamnesis

## 1. Project Identity & Philosophy
- **Legacy Name:** Ada Memory Layer (`memory/`)
- **New Project Name:** **Anamnesis** (ἀνάμνησις)
- **Theoretical Grounding:** Rooted in Platonic epistemology (*Meno*, *Phaedo*): Knowledge in an AI agent is not accumulated via brute-force unbounded token history; it is the structured recollection and surfacing of relevant latent state. In modern agentic computing, Anamnesis serves as a **Local Cognitive Working Memory Controller (RAM)** with deterministic selective forgetting.

---

## 2. Legacy Codebase Analysis (`memory/` folder)
The legacy system designed in `memory/` contains solid mathematical foundations:
- `anchor.py` & `anchor_updater.py`: Implements an Exponential Moving Average (EMA) attention vector:
  anchor_{t+1} = alpha * embedding(event) + (1 - alpha) * anchor_t
  Operating on 384-dimensional dense vectors with calibrated alpha events (`ALPHA_FOCUS = 0.25`, `ALPHA_VIEW = 0.20`, `ALPHA_HOVER = 0.15`).
- `intensity.py`: Calculates activation score I in [0.0, 1.0] with daily decay (`DECAY_PER_DAY = 0.03`) and event-based boost signals (`BOOST_FOCUSED = 0.20`, `BOOST_RELINKED = 0.15`, etc.).
- `embedder.py`: Local, fast ONNX-based text encoder using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` via `fastembed` (384 dimensions, ~5ms per inference).
- `embeddings_store.py`: Dual-backend vector store backed by SQLite (using `sqlite-vss` FAISS virtual tables with pure-numpy brute-force fallback).
- `threads.py`, `exchanges.py`, `latent_search.py`: Persists turns as chronological traces and enables L2 nearest-neighbour retrieval.

---

## 3. Four Identified Architectural Limitations & Their Solutions in Anamnesis

### A. Centroid Washout (Topic Multi-tasking Dilution)
- **Problem:** Averaging disparate topics into a single EMA vector creates an uninformative mathematical centroid in latent space, degrading retrieval precision.
- **Solution (Dual-Head Anchor):** Split the attention vector into two parallel moving averages:
  - **Macro Task Anchor ($A^{\text{task}}$, $\alpha = 0.03$):** Locks onto project scope and enduring intent.
  - **Micro Dialogue Anchor ($A^{\text{dialogue}}$, $\alpha = 0.35$):** Rapidly tracks immediate conversational turns.
  - Combined similarity: S_i = 0.6 * sim(A^dialogue, e_i) + 0.4 * sim(A^task, e_i).

### B. Amnesia of Critical Invariants (Safety & Credentials)
- **Problem:** Passive mathematical decay causes vital, low-frequency rules (e.g., customs clearance protocols, safety guardrails, API keys) to fade and get evicted.
- **Solution (Pinned Constraints):** Introduce a boolean `pinned: bool = True` attribute. Pinned memory blocks are strictly immune to decay ($I \equiv 1.0$) and receive guaranteed pre-allocated budget in the prompt assembly stage.

### C. Embedding Distance Blindspots (Antonym & Negation Ambiguity)
- **Problem:** Sentences with opposite meanings (e.g., "dispatch freight" vs "cancel freight dispatch") have tiny Euclidean distances because they share identical domain terms.
- **Solution (Hybrid Symbolic Filtering):** Enforce stateful metadata flags (`status: 'active' | 'deprecated' | 'superseded'`, `polarity: bool`). Deprecated commands are deterministically pruned regardless of high vector proximity.

### D. Hyperparameter Sensitivity (Rigid Drift vs Premature Amnesia)
- **Problem:** Fixed alpha either forgets prior context too quickly or fails to pivot during abrupt user topic shifts.
- **Solution (Adaptive Alpha):** Measure Euclidean distance between the previous anchor and the incoming event vector:
  - If dist > 1.1 (abrupt topic rupture): adapt aggressively with alpha = 0.55.
  - If dist < 0.4 (incremental topic continuity): adapt smoothly with alpha = 0.15.

---

## 4. Verification Suite & Test Specifications to Execute
Any engineer or agent building this repository must ensure all 4 verification tests pass:
1. `test_pinned_rules_survive_decay_cycles`: Run 25 consecutive decay cycles; assert pinned block intensity remains 1.0, while regular chatter drops to 0.0.
2. `test_dual_head_anchor_prevents_centroid_washout`: Feed 5 consecutive turns of completely unrelated domain embeddings; verify task anchor maintains >0.70 similarity to the base project domain.
3. `test_symbolic_filter_blocks_deprecated_and_antonyms`: Place an active order and a cancelled order in the memory pool; verify only the active order is selected despite identical embedding coordinates.
4. `test_dynamic_alpha_responsiveness`: Ensure abrupt distance triggers alpha >= 0.50 while incremental distance keeps alpha <= 0.20.
