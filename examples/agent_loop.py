"""End-to-End Example: Demonstrating Anamnesis in an Autonomous Agent Loop."""

import numpy as np
from core.controller import AnamnesisController
from telemetry.metrics import TelemetryTracker


def run_cloud_ops_agent_simulation():
    print("=" * 60)
    print("ANAMNESIS: Cognitive Working Memory Simulation")
    print("=" * 60)

    # Initialize memory & telemetry
    dim = 384
    memory = AnamnesisController(dim=dim, decay_rate=0.08)
    telemetry = TelemetryTracker(db_path="simulation_telemetry.db")

    # 1. Register Pinned Invariant (e.g. Mandatory Infrastructure Security Policy)
    v_security = np.zeros(dim, dtype=np.float32)
    v_security[0] = 1.0  # Domain axis: Security & Compliance

    memory.add_block(
        block_id="rule_security_01",
        content="CRITICAL SECURITY POLICY: All database endpoints must require TLSv1.3 and mTLS.",
        vector=v_security,
        pinned=True,
    )
    print("\n[+] Registered Pinned Invariant: Database TLS enforcement.")

    # 2. Simulate dialogue with task execution and topic drift
    v_infra = np.zeros(dim, dtype=np.float32)
    v_infra[0] = 0.8
    v_infra[1] = 0.2

    v_frontend = np.zeros(dim, dtype=np.float32)
    v_frontend[10] = 1.0  # Disparate topic: UI / Frontend styling

    turns = [
        ("Scale Kubernetes worker pool to 10 nodes", v_infra, False),
        ("Verify ingress controller health check latency", v_infra, False),
        ("Check memory utilization across pod replica sets", v_infra, False),
        ("Can you change the dashboard button to dark blue?", v_frontend, True),  # Topic shift
        ("Add a hover animation on the telemetry card", v_frontend, False),
        ("Also adjust the CSS grid layout for mobile viewports", v_frontend, False),
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
    run_cloud_ops_agent_simulation()
