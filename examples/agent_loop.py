"""End-to-End Example: Demonstrating Anamnesis in an Autonomous Agent Loop."""

import numpy as np
from core.controller import AnamnesisController
from telemetry.metrics import TelemetryTracker


def run_logistics_agent_simulation():
    print("=" * 60)
    print("ANAMNESIS: Cognitive Working Memory Simulation")
    print("=" * 60)

    # Initialize memory & telemetry
    dim = 384
    memory = AnamnesisController(dim=dim, decay_rate=0.08)
    telemetry = TelemetryTracker(db_path="simulation_telemetry.db")

    # 1. Register Pinned Invariant (e.g. Customs compliance rule)
    v_customs = np.zeros(dim, dtype=np.float32)
    v_customs[0] = 1.0  # Domain axis: customs & compliance

    memory.add_block(
        block_id="rule_customs_01",
        content="CRITICAL: All shipments over 1000 EUR require EUR.1 Certificate of Origin.",
        vector=v_customs,
        pinned=True,
    )
    print("\n[+] Registered Pinned Rule: EUR.1 certificate requirement.")

    # 2. Simulate 10 turns of active dialogue with topic drift
    v_freight = np.zeros(dim, dtype=np.float32)
    v_freight[0] = 0.8
    v_freight[1] = 0.2

    v_css = np.zeros(dim, dtype=np.float32)
    v_css[10] = 1.0  # Completely unrelated topic: web development styling

    turns = [
        ("Quote freight rate for 2 pallets to Dakar", v_freight, False),
        ("Include port handling and demurrage", v_freight, False),
        ("What about terminal handling charges?", v_freight, False),
        ("Can you change the dashboard button to dark blue?", v_css, True),  # Topic shift
        ("Add a hover effect on the tracking card", v_css, False),
        ("Also adjust the CSS grid layout", v_css, False),
    ]

    for turn_idx, (user_prompt, vec, shift) in enumerate(turns, 1):
        memory.step_interaction(user_prompt, vec, is_task_shift=shift)
        memory.add_block(
            block_id=f"turn_{turn_idx}",
            content=f"User asked: {user_prompt}",
            vector=vec,
            pinned=False,
        )

        # Assemble prompt budget
        prompt = memory.assemble_prompt_context(max_tokens=600)
        
        # Track telemetry
        raw_est = 400 * turn_idx  # Unbounded naive history growth
        pruned_est = len(prompt) // 4
        metric = telemetry.record_turn(
            raw_tokens_estimate=raw_est,
            pruned_tokens=pruned_est,
            pinned_count=1,
            active_memory_count=len(memory.blocks),
        )
        print(f"Turn {turn_idx} | Topic Shift: {shift:<5} | Token Savings: {metric.saved_percentage}%")

    print("\n" + "=" * 60)
    print("FINAL ASSEMBLED CONTEXT (Pruned & Focused):")
    print("=" * 60)
    print(memory.assemble_prompt_context(max_tokens=800))

    print("\n" + "=" * 60)
    print("TELEMETRY SUMMARY REPORT:")
    print("=" * 60)
    for k, v in telemetry.summary().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    run_logistics_agent_simulation()
