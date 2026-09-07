"""Automated Verification Suite for Anamnesis Mitigations."""

import unittest
import numpy as np
from core.controller import AnamnesisController, DualHeadAnchor, MemoryBlock

class TestAnamnesisVerification(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.dim = 384
        # Synthetic unit vectors for deterministic testing
        self.v_domain_a = np.zeros(self.dim, dtype=np.float32)
        self.v_domain_a[0] = 1.0  # Axis 0: Domain A (Primary task)
        
        self.v_domain_b = np.zeros(self.dim, dtype=np.float32)
        self.v_domain_b[1] = 1.0  # Axis 1: Domain B (Disparate topic)
        
        self.v_negation = np.zeros(self.dim, dtype=np.float32)
        self.v_negation[0] = 0.95
        self.v_negation[2] = 0.31  # Close vector space, but opposite semantic status

    def test_pinned_rules_survive_decay_cycles(self):
        """Verifies that critical constraints never suffer from selective forgetting."""
        controller = AnamnesisController(dim=self.dim, decay_rate=0.05)
        
        b_pinned = controller.add_block(
            block_id="rule_01",
            content="CRITICAL INVARIANT: Ensure JWT signature matches tenant authority.",
            vector=self.v_domain_a,
            pinned=True
        )
        b_chatter = controller.add_block(
            block_id="chat_01",
            content="What was the latency profile on request batch #441?",
            vector=self.v_domain_a,
            pinned=False
        )

        # 25 conversation turns
        for _ in range(25):
            controller.step_interaction("turn", self.v_domain_a)

        self.assertEqual(b_pinned.intensity, 1.0, "Pinned rule intensity must remain locked at 1.0")
        self.assertEqual(b_chatter.intensity, 0.0, "Unpinned chatter must decay to 0.0")

    def test_dual_head_anchor_prevents_centroid_washout(self):
        """Validates that conversational drift does not overwrite long-term project intent."""
        anchor = DualHeadAnchor(dim=self.dim)
        anchor.update(self.v_domain_a, is_task_shift=True)

        for _ in range(5):
            anchor.update(self.v_domain_b, is_task_shift=False)

        sim_dialogue_b = 1.0 / (1.0 + np.linalg.norm(anchor.dialogue_anchor - self.v_domain_b))
        self.assertGreater(sim_dialogue_b, 0.70, "Dialogue anchor failed to track current topic")

        sim_task_a = 1.0 / (1.0 + np.linalg.norm(anchor.task_anchor - self.v_domain_a))
        self.assertGreater(sim_task_a, 0.70, "Task anchor washed out due to local dialogue drift")

    def test_symbolic_filter_blocks_deprecated_and_antonyms(self):
        """Validates that deprecated commands are pruned despite close vector distance."""
        controller = AnamnesisController(dim=self.dim)
        controller.step_interaction("init", self.v_domain_a)

        controller.add_block(
            block_id="action_active",
            content="EXECUTE: Deploy worker node to cluster us-east-1.",
            vector=self.v_domain_a,
            status="active"
        )
        controller.add_block(
            block_id="action_cancelled",
            content="ABORT: Do not deploy worker node to cluster us-east-1.",
            vector=self.v_negation,
            status="deprecated"
        )

        prompt = controller.assemble_prompt_context(max_tokens=500)
        self.assertIn("Deploy worker node", prompt)
        self.assertNotIn("Do not deploy worker node", prompt)

if __name__ == "__main__":
    unittest.main()
