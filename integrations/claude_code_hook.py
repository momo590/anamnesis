#!/usr/bin/env python3
"""Claude Code UserPromptSubmit Hook for Anamnesis Context Injection."""

import sys
import json
import sqlite3
from pathlib import Path

DB_PATH = Path.home() / ".anamnesis" / "memory.db"

def run_hook():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS active_memory (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                pinned BOOLEAN NOT NULL,
                intensity REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_chars INTEGER,
                injected_chars INTEGER,
                saved_ratio REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Decay non-pinned items
        conn.execute("UPDATE active_memory SET intensity = MAX(0.0, intensity - 0.05) WHERE pinned = 0")
        
        rows = conn.execute(
            "SELECT content, pinned FROM active_memory WHERE pinned = 1 OR intensity > 0.3 ORDER BY pinned DESC"
        ).fetchall()

    try:
        payload = json.load(sys.stdin)
        user_prompt = payload.get("prompt", "")
    except Exception:
        user_prompt = ""

    injected_lines = [r[0] for r in rows]
    injected_text = "\\n".join(injected_lines) if injected_lines else "None (Fresh Context)"

    raw_chars = len(user_prompt) + 12000
    injected_chars = len(injected_text) + len(user_prompt)
    saved_ratio = round((1.0 - (injected_chars / max(1, raw_chars))) * 100, 1)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO telemetry (raw_chars, injected_chars, saved_ratio) VALUES (?, ?, ?)",
            (raw_chars, injected_chars, saved_ratio)
        )

    output = {
        "systemMessage": f"[Anamnesis: ~{saved_ratio}% token efficiency on this turn]",
        "hookSpecificOutput": {
            "additionalContext": f"=== ANAMNESIS WORKING MEMORY ===\\n{injected_text}"
        }
    }
    print(json.dumps(output))

if __name__ == "__main__":
    run_hook()
