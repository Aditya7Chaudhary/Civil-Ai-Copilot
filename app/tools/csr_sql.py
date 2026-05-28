import csv
import os
import sqlite3
from typing import Any, Dict, List, Tuple


CSR_CSV_PATH = os.path.join("data", "csr", "csr_rates.csv")
CSR_SQLITE_PATH = os.path.join("data", "csr", "csr_rates.sqlite3")


def ensure_csr_sqlite() -> str:
    """
    Ensure a local SQLite DB exists for CSR rates.

    This allows BOQ/COST to be executed via SQL (SELECT-only) rather than scanning CSV rows.
    """
    os.makedirs(os.path.dirname(CSR_SQLITE_PATH), exist_ok=True)
    if os.path.exists(CSR_SQLITE_PATH):
        return CSR_SQLITE_PATH

    if not os.path.exists(CSR_CSV_PATH):
        raise FileNotFoundError("CSR registry database file not found.")

    conn = sqlite3.connect(CSR_SQLITE_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS csr_rates (
                item_code TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                unit TEXT NOT NULL,
                rate_inr REAL NOT NULL
            )
            """
        )
        conn.execute("DELETE FROM csr_rates")

        with open(CSR_CSV_PATH, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows: List[Tuple[str, str, str, float]] = []
            for r in reader:
                rows.append(
                    (
                        (r.get("item_code") or "").strip().upper(),
                        (r.get("description") or "").strip(),
                        (r.get("unit") or "").strip(),
                        float(r.get("rate_inr") or 0.0),
                    )
                )

        conn.executemany(
            "INSERT OR REPLACE INTO csr_rates(item_code, description, unit, rate_inr) VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()

    return CSR_SQLITE_PATH


def execute_select(sql: str) -> List[Dict[str, Any]]:
    """
    Execute a SELECT-only query against the CSR SQLite DB and return rows as dicts.
    """
    db_path = ensure_csr_sqlite()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(sql)
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()

