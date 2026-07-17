"""① PromptStore lưu bằng SQLite — registry bền, xem lại được sau khi tắt máy."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ...core.interfaces import BasePromptStore
from ...core.registry import register
from ...core.types import PromptVersion


@register("store", "sqlite")
class SQLitePromptStore(BasePromptStore):
    """:param db_path: file DB. :param run_name: gom các phiên bản theo 1 lần chạy."""

    def __init__(self, db_path: str = "prompt_versions.db", run_name: str = "default"):
        self.db_path = Path(db_path)
        self.run_name = run_name
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prompt_versions (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_name   TEXT NOT NULL,
                    version    INTEGER NOT NULL,
                    text       TEXT NOT NULL,
                    score      REAL,
                    created_at TEXT,
                    metadata   TEXT,
                    UNIQUE(run_name, version)
                )
                """
            )

    def save(self, prompt_text: str, metadata: Optional[dict] = None) -> PromptVersion:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(version) + 1, 0) AS v FROM prompt_versions WHERE run_name = ?",
                (self.run_name,),
            ).fetchone()
            version = int(row["v"])
            created = datetime.now().isoformat(timespec="seconds")
            conn.execute(
                """INSERT INTO prompt_versions (run_name, version, text, score, created_at, metadata)
                   VALUES (?, ?, ?, NULL, ?, ?)""",
                (self.run_name, version, prompt_text, created, json.dumps(metadata or {})),
            )
        return PromptVersion(version=version, text=prompt_text, created_at=created,
                             metadata=metadata or {})

    def record_score(self, version: int, score: float) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE prompt_versions SET score = ? WHERE run_name = ? AND version = ?",
                (score, self.run_name, version),
            )

    def _to_pv(self, r) -> PromptVersion:
        return PromptVersion(
            version=r["version"], text=r["text"], score=r["score"],
            created_at=r["created_at"], metadata=json.loads(r["metadata"] or "{}"),
        )

    def history(self) -> List[PromptVersion]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM prompt_versions WHERE run_name = ? ORDER BY version",
                (self.run_name,),
            ).fetchall()
        return [self._to_pv(r) for r in rows]

    def best(self) -> Optional[PromptVersion]:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM prompt_versions
                   WHERE run_name = ? AND score IS NOT NULL
                   ORDER BY score DESC, version ASC LIMIT 1""",
                (self.run_name,),
            ).fetchone()
        return self._to_pv(row) if row else None
