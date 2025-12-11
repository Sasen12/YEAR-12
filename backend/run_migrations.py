"""Simple migration runner for SQLite using provided SQL files in migrations/"""
from pathlib import Path
import sqlite3

BASE = Path(__file__).parent
DB_PATH = BASE / "app.db"
MIGRATIONS = sorted((BASE / "migrations").glob("*.sql"))

def run():
    """Execute SQL migration files against the local SQLite database.

    The function applies every `migrations/*.sql` file in lexical
    order. It is intended for local development and quick bootstrapping
    of the example database.
    """
    print("Using database:", DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for m in MIGRATIONS:
        print("Applying:", m.name)
        sql = m.read_text(encoding="utf-8")
        cur.executescript(sql)
    conn.commit()
    conn.close()
    print("Migrations applied.")

if __name__ == '__main__':
    run()
