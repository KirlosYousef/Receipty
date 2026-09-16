from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Protocol

from app.domain.schemas import ReceiptExtract

EMBEDDING_DIMENSIONS = 1536


class ReceiptRepository(Protocol):
    def init_db(self) -> None: ...

    def save(self, row: ReceiptExtract) -> int: ...

    def list_all(self) -> list[dict]: ...

    def save_document(
        self, kind: str, source_id: str, content: str, embedding: list[float]
    ) -> int: ...

    def list_documents(self) -> list[dict]: ...

    def close(self) -> None: ...


def build_repository(
    *,
    db_path: Path | None = None,
    database_url: str | None = None,
) -> ReceiptRepository:
    if database_url:
        return PostgresReceiptRepository(database_url)
    return SqliteReceiptRepository(db_path or Path("receipts.db"))


class SqliteReceiptRepository:
    def __init__(self, db_path: Path):
        self._db_path = db_path

    def init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS receipts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    is_receipt INTEGER NOT NULL,
                    merchant TEXT,
                    total TEXT,
                    currency TEXT,
                    date TEXT,
                    tax TEXT,
                    outcome TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    source_id TEXT NOT NULL UNIQUE,
                    content TEXT NOT NULL,
                    embedding TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, row: ReceiptExtract) -> int:
        conn = self._connect()
        try:
            cur = conn.execute(
                """
                INSERT INTO receipts
                (is_receipt, merchant, total, currency, date, tax, outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(row.is_receipt),
                    row.merchant,
                    str(row.total) if row.total is not None else None,
                    row.currency,
                    str(row.date) if row.date is not None else None,
                    str(row.tax) if row.tax is not None else None,
                    row.outcome if row.outcome is not None else None,
                ),
            )
            conn.commit()
            if cur.lastrowid is None:
                raise RuntimeError("SQLite did not return an inserted receipt ID")
            return int(cur.lastrowid)
        finally:
            conn.close()

    def list_all(self) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM receipts ORDER BY id DESC").fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def close(self) -> None:
        return

    def save_document(
        self, kind: str, source_id: str, content: str, embedding: list[float]
    ) -> int:
        conn = self._connect()
        try:
            cur = conn.execute(
                """
                INSERT INTO documents (kind, source_id, content, embedding)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    kind=excluded.kind,
                    content=excluded.content,
                    embedding=excluded.embedding
                """,
                (kind, source_id, content, json.dumps(embedding)),
            )
            conn.commit()
            return int(cur.lastrowid or 0)
        finally:
            conn.close()

    def list_documents(self) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, kind, source_id, content FROM documents ORDER BY id"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


class PostgresReceiptRepository:  # pragma: no cover
    """Postgres-backed repository.

    Excluded from CI coverage: it requires a live Postgres instance.
    Run ``docker compose up`` and ``TEST_DATABASE_URL=... pytest`` to exercise it.
    """

    def __init__(self, database_url: str):
        import psycopg

        self._psycopg = psycopg
        self._database_url = database_url
        self._conn: Any = None

    def _connect(self) -> Any:
        if self._conn is None:
            from pgvector.psycopg import register_vector

            self._conn = self._psycopg.connect(self._database_url, autocommit=True)
            register_vector(self._conn)
        return self._conn

    def init_db(self) -> None:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS receipts (
                    id BIGSERIAL PRIMARY KEY,
                    is_receipt BOOLEAN NOT NULL,
                    merchant TEXT,
                    total TEXT,
                    currency TEXT,
                    date TEXT,
                    tax TEXT,
                    outcome TEXT,
                    embedding vector({EMBEDDING_DIMENSIONS})
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS documents (
                    id BIGSERIAL PRIMARY KEY,
                    kind TEXT NOT NULL,
                    source_id TEXT NOT NULL UNIQUE,
                    content TEXT NOT NULL,
                    embedding vector({EMBEDDING_DIMENSIONS})
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS documents_embedding_hnsw
                ON documents USING hnsw (embedding vector_cosine_ops)
                """
            )

    def save(self, row: ReceiptExtract) -> int:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO receipts
                (is_receipt, merchant, total, currency, date, tax, outcome)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    row.is_receipt,
                    row.merchant,
                    str(row.total) if row.total is not None else None,
                    row.currency,
                    str(row.date) if row.date is not None else None,
                    str(row.tax) if row.tax is not None else None,
                    row.outcome if row.outcome is not None else None,
                ),
            )
            result = cur.fetchone()
            if result is None:
                raise RuntimeError("Postgres did not return an inserted receipt ID")
            return int(result[0])

    def list_all(self) -> list[dict]:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, is_receipt, merchant, total, currency, date, tax, outcome "
                "FROM receipts ORDER BY id DESC"
            )
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def save_document(
        self, kind: str, source_id: str, content: str, embedding: list[float]
    ) -> int:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (kind, source_id, content, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (source_id) DO UPDATE SET
                    kind = EXCLUDED.kind,
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding
                RETURNING id
                """,
                (kind, source_id, content, embedding),
            )
            result = cur.fetchone()
            if result is None:
                raise RuntimeError("Postgres did not return an inserted document ID")
            return int(result[0])

    def list_documents(self) -> list[dict]:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, kind, source_id, content FROM documents ORDER BY id"
            )
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]
