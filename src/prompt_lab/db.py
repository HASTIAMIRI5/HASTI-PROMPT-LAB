from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path("data/prompt_lab.db")


class ExperimentStore:
    """Minimal SQLite store for prompt runs and experiment metadata."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    variant_name TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT,
                    dataset_name TEXT,
                    metrics TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def save_run(
        self,
        *,
        run_id: str,
        variant_name: str,
        provider: str,
        model: str | None,
        dataset_name: str | None,
        metrics: dict[str, Any],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO experiments (
                    run_id, variant_name, provider, model, dataset_name, metrics, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    run_id,
                    variant_name,
                    provider,
                    model,
                    dataset_name,
                    json.dumps(metrics, default=str),
                ),
            )

    def list_runs(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM experiments ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiments WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        payload = dict(row)
        payload["metrics"] = json.loads(payload["metrics"])
        return payload
