"""Automated Verification Suite for Anamnesis Mitigations."""

import unittest
import numpy as np
from core.controller import AnamnesisController, DualHeadAnchor, MemoryBlock

class TestAnamnesisVerification(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.dim = 384
        self.v_logistics = np.zeros(self.dim, dtype=np.float32)
        self.v_logistics[0] = 1.0  # Axis 0: supply chain
        
        self.v_coding = np.zeros(self.dim, dtype=np.float32)
        self.v_coding[1] = 1.0     # Axis 1: Python/software
        
        self.v_negation = np.zeros(self.dim, dtype=np.float32)
        self.v_negation[0] = 0.95
        self.v_negation[2] = 0.31

    def test_pinned_rules_survive_decay_cycles(self):
        """Verifies that critical constraints never suffer from selective forgetting."""
        controller = AnamnesisController(dim=self.dim, decay_rate=0.05)
        
        b_pinned = controller.add_block(
            block_id="rule_01",
            content="MANDATORY: Verify export license before shipping.",
            vector=self.v_logistics,
            pinned=True
        )
        b_chatter = controller.add_block(
            block_id="chat_01",
            content="Can we check tracking on shipment #441?",
            vector=self.v_logistics,
            pinned=False
        )

        # 25 conversation turns
        for _ in range(25):
            controller.step_interaction("turn", self.v_logistics)

        self.assertEqual(b_pinned.intensity, 1.0, "Pinned rule intensity must remain locked at 1.0")
        self.assertEqual(b_chatter.intensity, 0.0, "Unpinned chatter must decay to 0.0")

    def test_dual_head_anchor_prevents_centroid_washout(self):
        """Validates that conversational drift does not overwrite long-term project intent."""
        anchor = DualHeadAnchor(dim=self.dim)
        anchor.update(self.v_logistics, is_task_shift=True)

        for _ in range(5):
            anchor.update(self.v_coding, is_task_shift=False)

        sim_dialogue_coding = 1.0 / (1.0 + np.linalg.norm(anchor.dialogue_anchor - self.v_coding))
        self.assertGreater(sim_dialogue_coding, 0.70, "Dialogue anchor failed to track current topic")

        sim_task_logistics = 1.0 / (1.0 + np.linalg.norm(anchor.task_anchor - self.v_logistics))
        self.assertGreater(sim_task_logistics, 0.70, "Task anchor washed out due to local dialogue drift")

    def test_symbolic_filter_blocks_deprecated_and_antonyms(self):
        """Validates that deprecated commands are pruned despite close vector distance."""
        controller = AnamnesisController(dim=self.dim)
        controller.step_interaction("init", self.v_logistics)

        controller.add_block(
            block_id="order_active",
            content="Dispatch parcel via DHL.",
            vector=self.v_logistics,
            status="active"
        )
        controller.add_block(
            block_id="order_cancelled",
            content="CANCEL parcel dispatch for DHL.",
            vector=self.v_negation,
            status="deprecated"
        )

        prompt = controller.assemble_prompt_context(max_tokens=500)
        self.assertIn("Dispatch parcel via DHL", prompt)
        self.assertNotIn("CANCEL parcel dispatch", prompt)

if __name__ == "__main__":
    unittest.main()
