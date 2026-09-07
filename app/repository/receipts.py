import sqlite3
from pathlib import Path

from app.domain.schemas import ReceiptExtract


class ReceiptRepository:
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
                    needs_review INTEGER NOT NULL
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
                (is_receipt, merchant, total, currency, date, tax, needs_review)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(row.is_receipt),
                    row.merchant,
                    str(row.total) if row.total is not None else None,
                    row.currency,
                    str(row.date) if row.date is not None else None,
                    str(row.tax) if row.tax is not None else None,
                    int(row.needs_review),
                ),
            )
            conn.commit()
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
