"""Database engine and helpers.

This module configures the SQLModel/SQLAlchemy engine for a local
SQLite database and provides small helpers used by the application and
tests. The database file is located at the repository root as `app.db`.
"""

from sqlmodel import SQLModel, create_engine, Session
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DB_URL = f"sqlite:///{BASE / 'app.db'}"
engine = create_engine(DB_URL, echo=False, connect_args={"check_same_thread": False})


def create_db_and_tables():
    """Create database tables using SQLModel metadata.

    This function is intended for local development and lightweight
    scripts; production deployments should rely on a proper migration
    tool (alembic) instead.
    """
    SQLModel.metadata.create_all(engine)
    _ensure_explanation_column()
    _ensure_exam_metadata_columns()


def _ensure_explanation_column():
    """Ensure the `explanation` column exists on questions for older DB files.

    This is a lightweight, idempotent ALTER to keep demo databases in sync
    when new columns are added without a full migration run.
    """
    with engine.connect() as conn:
        try:
            conn.exec_driver_sql("ALTER TABLE questions ADD COLUMN explanation TEXT")
        except Exception:
            # ignore if column already exists or table missing
            pass


def _ensure_exam_metadata_columns():
    """Ensure `exam_year` and `question_number` columns exist for older DB files."""
    with engine.connect() as conn:
        for col in ("exam_year INTEGER", "question_number INTEGER"):
            try:
                conn.exec_driver_sql(f"ALTER TABLE questions ADD COLUMN {col}")
            except Exception:
                pass


def get_session():
    """Yield a database `Session` for FastAPI dependency injection.

    The generator yields a session and ensures it is closed when the
    request scope finishes.
    """
    with Session(engine) as session:
        yield session
