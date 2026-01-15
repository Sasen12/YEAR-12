from pathlib import Path
import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def reset_db():
    """Ensure a fresh SQLite database for tests."""
    db_path = Path(__file__).resolve().parents[1] / "app.db"
    if db_path.exists():
        try:
            db_path.unlink()
        except Exception:
            pass
    # Let the app import recreate tables on demand
    yield
