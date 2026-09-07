"""LiteLLM Custom Handler / Interceptor for Transparent Context Pruning.

Inspects incoming completion requests, applies Anamnesis selective decay,
and prunes cold turns from the messages payload before forwarding to the upstream LLM.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np

from litellm.integrations.custom_logger import CustomLogger
from core.controller import AnamnesisController

logger = logging.getLogger("anamnesis.litellm")


class AnamnesisLiteLLMHandler(CustomLogger):
    """Custom interceptor hooked into the LiteLLM proxy pipeline."""

    def __init__(self, dim: int = 384, max_context_tokens: int = 2000):
        super().__init__()
        self.controller = AnamnesisController(dim=dim)
        self.max_context_tokens = max_context_tokens

    async def async_pre_call_hook(
        self,
        user_api_key_dict: dict[str, Any],
        data: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any]:
        """Runs immediately before the request is dispatched to OpenAI/Anthropic/Bedrock."""
        messages: list[dict[str, str]] = data.get("messages", [])
        if not messages:
            return data

        # Extract latest user message
        latest_user_turn = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                latest_user_turn = m.get("content", "")
                break

        if latest_user_turn:
            # Deterministic representation for fast inline routing
            vec = np.zeros(self.controller.dim, dtype=np.float32)
            for i, c in enumerate(latest_user_turn[:self.controller.dim]):
                vec[i % self.controller.dim] += (ord(c) % 50) / 50.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm

            # Step the attention anchor
            self.controller.step_interaction(latest_user_turn, vec)

        # Assemble optimized working memory
        pruned_memory = self.controller.assemble_prompt_context(
            max_tokens=self.max_context_tokens
        )

        if pruned_memory:
            system_msg = {
                "role": "system",
                "content": f"=== ANAMNESIS SELECTIVE CONTEXT ===\n{pruned_memory}",
            }
            # Inject system context at position 0 or after existing developer instructions
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] += f"\n\n{system_msg['content']}"
            else:
                messages.insert(0, system_msg)

        data["messages"] = messages
        return data

    def log_success_event(self, kwargs: Any, response_obj: Any, start_time: Any, end_time: Any):
        """Track completion tokens and latency metrics."""
        usage = getattr(response_obj, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            logger.info("Anamnesis Proxy: Upstream call processed with %d prompt tokens", prompt_tokens)
