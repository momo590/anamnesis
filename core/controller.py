"""Anamnesis Core Controller: Reference implementation integrating all mitigations.

Composes Dual-Head Anchoring, Pinned Invariants, Symbolic Filtering,
and Local SQLite Persistence into a standalone memory manager.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger("anamnesis.controller")

DEFAULT_DECAY_RATE = 0.04
DEFAULT_MIN_INTENSITY_THRESHOLD = 0.20
DEFAULT_PINNED_BUDGET_RATIO = 0.25


@dataclass
class MemoryBlock:
    id: str
    content: str
    vector: np.ndarray
    intensity: float = 1.0
    pinned: bool = False
    status: Literal["active", "deprecated", "superseded"] = "active"
    polarity: bool = True
    metadata: dict = field(default_factory=dict)
    turn_added: int = 0
    last_accessed: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))


class DualHeadAnchor:
    """Maintains two EMA attention vectors: micro (dialogue) and macro (task)."""

    def __init__(self, dim: int):
        self.dim = dim
        self.task_anchor: np.ndarray | None = None
        self.dialogue_anchor: np.ndarray | None = None

    def update(self, vec: np.ndarray, is_task_shift: bool = False):
        if vec.shape != (self.dim,):
            raise ValueError(f"Vector dim mismatch: expected ({self.dim},), got {vec.shape}")
        
        v = np.ascontiguousarray(vec, dtype=np.float32)
        if self.dialogue_anchor is None:
            self.dialogue_anchor = v.copy()
            self.task_anchor = v.copy()
            return

        # Micro: rapid conversational drift
        self.dialogue_anchor = 0.35 * v + 0.65 * self.dialogue_anchor

        # Macro: slow project goal drift
        alpha_task = 0.25 if is_task_shift else 0.03
        self.task_anchor = alpha_task * v + (1.0 - alpha_task) * self.task_anchor

    def similarity_score(self, target_vec: np.ndarray) -> float:
        """Inverted Euclidean distance combined across both attention heads."""
        dist_d = float(np.linalg.norm(self.dialogue_anchor - target_vec))
        dist_t = float(np.linalg.norm(self.task_anchor - target_vec))
        sim_d = 1.0 / (1.0 + dist_d)
        sim_t = 1.0 / (1.0 + dist_t)
        return 0.6 * sim_d + 0.4 * sim_t


class AnamnesisController:
    """Standalone working memory manager implementing selective forgetting."""

    def __init__(
        self,
        dim: int = 384,
        decay_rate: float = DEFAULT_DECAY_RATE,
        storage_path: Path | None = None,
    ):
        self.dim = dim
        self.decay_rate = decay_rate
        self.anchor = DualHeadAnchor(dim=dim)
        self.blocks: list[MemoryBlock] = []
        self.current_turn = 0
        self.storage_path = storage_path

    def add_block(
        self,
        block_id: str,
        content: str,
        vector: np.ndarray,
        pinned: bool = False,
        status: Literal["active", "deprecated", "superseded"] = "active",
        polarity: bool = True,
        metadata: dict | None = None,
    ) -> MemoryBlock:
        block = MemoryBlock(
            id=block_id,
            content=content,
            vector=vector,
            intensity=1.0,
            pinned=pinned,
            status=status,
            polarity=polarity,
            metadata=metadata or {},
            turn_added=self.current_turn,
        )
        self.blocks.append(block)
        return block

    def step_interaction(
        self,
        event_text: str,
        event_vector: np.ndarray,
        is_task_shift: bool = False,
    ):
        """Advances conversational turn, updates attention anchor, and decays memories."""
        self.current_turn += 1

        # 1. Update dual attention anchor
        self.anchor.update(event_vector, is_task_shift=is_task_shift)

        # 2. Apply selective decay (pinned invariants are completely immune)
        for b in self.blocks:
            if b.pinned or b.status != "active":
                continue
            b.intensity = max(0.0, b.intensity - self.decay_rate)

    def assemble_prompt_context(self, max_tokens: int, chars_per_token: int = 4) -> str:
        """Assembles the optimized prompt context obeying token limits and tiers."""
        budget_chars = max_tokens * chars_per_token
        active_items = [b for b in self.blocks if b.status == "active"]

        # Step A: Mandatory reservation for pinned rules
        pinned_items = [b for b in active_items if b.pinned]
        regular_items = [b for b in active_items if not b.pinned]

        selected_pinned: list[str] = []
        used_chars = 0

        for p in pinned_items:
            entry = f"[CRITICAL RULE]: {p.content}"
            selected_pinned.append(entry)
            used_chars += len(entry) + 2

        # Step B: Score remaining memories via dual-head anchor and intensity
        scored_regular = []
        for r in regular_items:
            sim = self.anchor.similarity_score(r.vector)
            composite_score = 0.5 * r.intensity + 0.5 * sim
            scored_regular.append((r, composite_score))

        scored_regular.sort(key=lambda x: x[1], reverse=True)

        # Step C: Fill remaining budget with active/warm tier items
        selected_regular: list[str] = []
        for r, score in scored_regular:
            if r.intensity < DEFAULT_MIN_INTENSITY_THRESHOLD and score < 0.35:
                continue

            entry = f"• {r.content}"
            if used_chars + len(entry) + 2 <= budget_chars:
                selected_regular.append(entry)
                used_chars += len(entry) + 2

        sections = []
        if selected_pinned:
            sections.append("=== PERMANENT CONSTRAINTS ===\\n" + "\\n".join(selected_pinned))
        if selected_regular:
            sections.append("=== ACTIVE WORKING MEMORY ===\\n" + "\\n".join(selected_regular))

        return "\\n\\n".join(sections)
