"""Model Context Protocol (MCP) Server for Anamnesis.

Exposes tools for Claude Desktop, Cursor, and MCP clients to query and record
working memory with selective forgetting.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP
import numpy as np

from core.controller import AnamnesisController

logger = logging.getLogger("anamnesis.mcp")

# Initialize FastMCP application
mcp = FastMCP("anamnesis-memory")

# Shared global controller instance
# In production, this coordinates with persistent storage (SQLite/VSS)
controller = AnamnesisController(dim=384)


@mcp.tool()
def get_working_memory(max_tokens: int = 1500) -> str:
    """Retrieve the currently active working memory assembled by Anamnesis.
    
    Includes permanent constraints (pinned) and top-intensity context items
    aligned with the current attention anchor.
    """
    context = controller.assemble_prompt_context(max_tokens=max_tokens)
    return context if context else "No active working memory recorded."


@mcp.tool()
def record_memory(
    content: str,
    pinned: bool = False,
    is_task_shift: bool = False,
    status: str = "active",
) -> str:
    """Record a memory artifact or user directive into Anamnesis working memory.
    
    Args:
        content: The text instruction, fact, or event to store.
        pinned: True if this is an immutable critical rule (immune to decay).
        is_task_shift: True if the user switched macro objectives.
        status: 'active' or 'deprecated'.
    """
    # Deterministic fallback vector generation if embedder is offline
    synthetic_vec = np.zeros(384, dtype=np.float32)
    # Simple deterministic hash projection for zero-dep MCP bootstrap
    for i, char in enumerate(content[:384]):
        synthetic_vec[i % 384] += (ord(char) % 100) / 100.0
    norm = np.linalg.norm(synthetic_vec)
    if norm > 0:
        synthetic_vec /= norm

    block_id = f"mem_{controller.current_turn + 1}_{len(controller.blocks)}"
    controller.add_block(
        block_id=block_id,
        content=content,
        vector=synthetic_vec,
        pinned=pinned,
        status=status,
    )
    controller.step_interaction(content, synthetic_vec, is_task_shift=is_task_shift)

    status_str = "PINNED (Immune to decay)" if pinned else "DECAYABLE (Active Working Memory)"
    return f"Memory successfully recorded [{status_str}]: {content[:80]}..."


@mcp.tool()
def inspect_attention_anchor() -> str:
    """Inspect the current dual-head attention status and active block count."""
    return json.dumps({
        "current_turn": controller.current_turn,
        "total_blocks": len(controller.blocks),
        "pinned_count": sum(1 for b in controller.blocks if b.pinned),
        "active_unpinned_count": sum(1 for b in controller.blocks if not b.pinned and b.status == "active"),
        "decay_rate": controller.decay_rate,
    }, indent=2)


if __name__ == "__main__":
    mcp.run()
