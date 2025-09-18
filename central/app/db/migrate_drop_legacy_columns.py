"""One-off migration script to drop legacy Job columns.

Usage (from repo root):
    python -m central.app.db.migrate_drop_legacy_columns

This rebuilds the jobs table without the removed columns using a SQLite
schema copy approach (since ALTER TABLE DROP COLUMN is limited pre-3.35).
Safe to run multiple times (idempotent check on existing columns).
"""
from __future__ import annotations
import sqlite3
from pathlib import Path

# central/app/db/migrate_drop_legacy_columns.py -> repo_root/data/central.db
DB_PATH = Path(__file__).resolve().parents[2] / ".." / "data" / "central.db"
DB_PATH = DB_PATH.resolve()

REMOVED_COLS = {
    "missing_spec",
    "iteration_before_first_imputation",
    "iteration_between_imputations",
    "imputation_trials",
}

def get_columns(cur, table: str) -> list[str]:
    cur.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]

def main():
    if not DB_PATH.exists():
        print(f"DB not found at {DB_PATH}")
        return
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cols = get_columns(cur, "jobs")
    present = [c for c in REMOVED_COLS if c in cols]
    if not present:
        print("No legacy columns present; nothing to do.")
        return
    keep_cols = [c for c in cols if c not in REMOVED_COLS]
    col_list = ", ".join(keep_cols)
    cur.execute("BEGIN TRANSACTION")
    try:
        cur.execute("ALTER TABLE jobs RENAME TO jobs_legacy_backup")
        # Obtain schema of old table
        cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='jobs_legacy_backup'")
        create_sql = cur.fetchone()[0]
        # Strip removed columns from CREATE statement (simple heuristic)
        for rc in present:
            create_sql = create_sql.replace(f", \"{rc}\" INTEGER", "")
            create_sql = create_sql.replace(f", \"{rc}\" JSON", "")
            create_sql = create_sql.replace(f", \"{rc}\" TEXT", "")
        # Also handle potential trailing commas before )
        create_sql = create_sql.replace(",)" , ")")
        create_sql = create_sql.replace("jobs_legacy_backup", "jobs")
        cur.execute(create_sql)
        cur.execute(f"INSERT INTO jobs ({col_list}) SELECT {col_list} FROM jobs_legacy_backup")
        con.commit()
        print(f"Removed columns: {present}. Backup table: jobs_legacy_backup (retain until manually dropped).")
    except Exception as e:
        con.rollback()
        print(f"Migration failed: {e}")
    finally:
        con.close()

if __name__ == "__main__":
    main()
