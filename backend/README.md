# Software Development Study Backend (FastAPI)

This repository contains a FastAPI backend for a Software Development study app.

Features:
- User auth (register/login) with hashed passwords and JWT tokens
- Import questions from CSV, TXT, JSON, PDF, DOCX
- Store questions, answers, quiz results, weekly goals in SQLite
- Endpoints for fetching questions, random quizzes, grading, goals
- Modular architecture: controllers, services, repositories, models
- Tests (pytest)

Quick start (Windows PowerShell):
```powershell
cd "c:\Users\sasen\OneDrive - Crest Education\Documents\School\2026\Software Dev\Code\Yr12 SAT\backend"
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_migrations.py
uvicorn app.main:app --reload --port 8000
```

Run tests:
```powershell
pytest -q
```

Importing local `Exams/` folders
--------------------------------
If you have an `Exams/` folder next to the `backend/` folder (for example the repository root contains `backend/` and `Exams/`), you can import exam files directly into the database.

- To import via the API (authenticated):
	- POST `/softwaredev/import_from_exams?year=2019` (optional `year` parameter). This endpoint scans the `Exams/` folder and imports supported file types (`.csv`, `.txt`, `.json`, `.pdf`, `.docx`).

- To import from the command line (server machine):
```powershell
cd "c:\Users\sasen\OneDrive - Crest Education\Documents\School\2026\Software Dev\Code\Yr12 SAT\backend"
.venv\Scripts\Activate.ps1
python scripts/import_exams.py --year 2019
# or omit --year to import all years
python scripts/import_exams.py
```

Notes:
- The import attempts to parse common formats. For PDF/DOCX/TXT that don't include explicit correct-answer markers, the first answer found is treated as the correct answer.
- The import reports per-file created question counts and parsing errors in the API response and CLI output.
