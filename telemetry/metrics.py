"""Telemetry & Performance Analytics for Anamnesis Working Memory.

Tracks token savings, TTFT (Time To First Token) acceleration,
and memory lifecycle dynamics in local SQLite storage.
"""

from __future__ import annotations

import sqlite3
import datetime as dt
from pathlib import Path
from dataclasses import dataclass


@dataclass
class TurnMetric:
    raw_tokens_estimate: int
    pruned_tokens: int
    saved_percentage: float
    pinned_count: int
    active_memory_count: int
    timestamp: str = ""


class TelemetryTracker:
    """Local-first telemetry recorder for individual and team environments."""

    def __init__(self, db_path: Path | str = "anamnesis_telemetry.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (\n                    id INTEGER PRIMARY KEY AUTOINCREMENT,\n                    raw_tokens INTEGER NOT NULL,\n                    pruned_tokens INTEGER NOT NULL,\n                    saved_percentage REAL NOT NULL,\n                    pinned_count INTEGER NOT NULL,\n                    active_memory_count INTEGER NOT NULL,\n                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP\n                )\n            """)
            conn.commit()

    def record_turn(
        self,
        raw_tokens_estimate: int,
        pruned_tokens: int,
        pinned_count: int,
        active_memory_count: int,
    ) -> TurnMetric:
        saved = max(0, raw_tokens_estimate - pruned_tokens)
        ratio = (saved / max(1, raw_tokens_estimate)) * 100.0

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO metrics (\n                    raw_tokens, pruned_tokens, saved_percentage, pinned_count, active_memory_count\n                ) VALUES (?, ?, ?, ?, ?)\n            """, (raw_tokens_estimate, pruned_tokens, ratio, pinned_count, active_memory_count))
            conn.commit()

        return TurnMetric(
            raw_tokens_estimate=raw_tokens_estimate,
            pruned_tokens=pruned_tokens,
            saved_percentage=round(ratio, 2),
            pinned_count=pinned_count,
            active_memory_count=active_memory_count,
            timestamp=dt.datetime.now(dt.UTC).isoformat(),
        )

    def summary(self) -> dict[str, float]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT \n                    COUNT(*),\n                    AVG(saved_percentage),\n                    SUM(raw_tokens - pruned_tokens),\n                    AVG(pinned_count)\n                FROM metrics\n            """).fetchone()

        if not row or row[0] == 0:
            return {"turns": 0, "avg_saved_pct": 0.0, "total_tokens_saved": 0}

        return {
            "total_turns": int(row[0]),
            "avg_token_savings_pct": round(float(row[1] or 0.0), 2),
            "total_tokens_saved": int(row[2] or 0),
            "avg_pinned_items": round(float(row[3] or 0.0), 1),
        }
