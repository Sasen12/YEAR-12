"""Simple and idempotent migration runner for SQLite SQL files."""
from pathlib import Path
import sqlite3

BASE = Path(__file__).parent
DB_PATH = BASE / "app.db"
MIGRATIONS = sorted((BASE / "migrations").glob("*.sql"))


def _ensure_migration_table(cur: sqlite3.Cursor) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            name TEXT PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def _is_duplicate_column_error(exc: sqlite3.OperationalError) -> bool:
    return "duplicate column name" in str(exc).lower()


def run():
    """Execute unapplied SQL migrations in lexical order.

    The runner tracks applied files in `schema_migrations` and tolerates
    duplicate-column alter statements so it is safe to re-run.
    """
    print("Using database:", DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    _ensure_migration_table(cur)
    conn.commit()

    applied = {
        row[0]
        for row in cur.execute("SELECT name FROM schema_migrations").fetchall()
    }

    for migration_path in MIGRATIONS:
        name = migration_path.name
        if name in applied:
            print("Skipping:", name)
            continue

        print("Applying:", name)
        sql = migration_path.read_text(encoding="utf-8")
        try:
            cur.executescript(sql)
        except sqlite3.OperationalError as exc:
            if _is_duplicate_column_error(exc):
                print(f"Warning: {name} attempted an existing column; continuing.")
            else:
                conn.rollback()
                conn.close()
                raise

        cur.execute("INSERT INTO schema_migrations(name) VALUES (?)", (name,))
        conn.commit()

    conn.close()
    print("Migrations applied.")


if __name__ == "__main__":
    run()
