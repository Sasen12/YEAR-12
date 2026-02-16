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
cd "c:\Users\sasen\OneDrive - Crest Education\Documents\School\2026\Software Dev\Code\YEAR-12\YEAR-12\backend"
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_migrations.py
uvicorn app.main:app --reload --port 8000
```

Alternatively run the included helper script which creates a venv,
installs deps, runs migrations and tests:

```powershell
.
\scripts\dev_setup_and_test.ps1
```

Fastest OCR lab start (one command):
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_ocr_lab.ps1
```
This opens `http://127.0.0.1:8000/ocr` automatically.

Notes:
- The app is tested with modern Python 3.x and dependency ranges in `requirements.txt`.
- Docker remains a good default for reproducible local runs (`python:3.11-slim` in `Dockerfile`).
- Set `JWT_SECRET` to a long random value for any non-dev deployment.
- OCR defaults to `rapidocr` and supports preprocessing + segmentation variants for better handwriting extraction.
- Optional extra engines are available via `requirements-ocr-extras.txt`.
- OCR lab now supports background jobs (`POST /ocr/jobs` + `GET /ocr/jobs/{job_id}`).
- OCR observability stats are available at `GET /ocr/stats`.
- Tune OCR with env vars in `.env.example` (`OCR_REQUEST_TIMEOUT_SECONDS`, `OCR_MAX_IMAGE_SIDE`, `OCR_RATE_LIMIT_PER_MIN`, etc.).

OCR benchmarking:
```powershell
python scripts/eval_ocr.py
# optional filters
python scripts/eval_ocr.py --split val --limit 100
```
The script writes a JSON report to `data/ocr_training/reports/` by default.

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
cd "c:\Users\sasen\OneDrive - Crest Education\Documents\School\2026\Software Dev\Code\YEAR-12\YEAR-12\backend"
.venv\Scripts\Activate.ps1
python scripts/import_exams.py --year 2019
# or omit --year to import all years
python scripts/import_exams.py
```

Notes:
- The import attempts to parse common formats. For PDF/DOCX/TXT that don't include explicit correct-answer markers, the first answer found is treated as the correct answer.
- The import reports per-file created question counts and parsing errors in the API response and CLI output.
