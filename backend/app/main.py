"""FastAPI application entrypoint and HTTP controllers.

This module defines the HTTP endpoints used by the Software Development
study backend. Controllers are intentionally thin: they accept requests,
delegate to services, and return JSON responses.

Endpoints implemented:
- POST /auth/register
- POST /auth/login
- GET /softwaredev/questions
- POST /softwaredev/import
- POST /softwaredev/import_from_exams
- GET /softwaredev/quiz 
- POST /softwaredev/grade
- POST /goals/set
- GET /goals/progress
"""

from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session
import os
import io
import json
import logging
import queue
import threading
import time
import uuid
from .database import engine, create_db_and_tables, get_session
from . import services, repositories, models
from .auth import get_current_user
from .schemas import RegisterIn, QuizSubmission
import jwt
from datetime import date
from pathlib import Path
from PIL import Image
from .utils.exam_loader import find_exam_files
from .utils.ocr import ocr_file_to_lines
from .utils.ocr_training import save_labeled_sample
from .utils.ocr_jobs import OCRJobStore
from .utils.rate_limit import InMemoryRateLimiter
from .utils.ocr_observability import get_ocr_stats
from .config import settings

app = FastAPI(title="Software Development Study API")
logger = logging.getLogger("app.api")
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
_ocr_rate_limiter = InMemoryRateLimiter()
_ocr_jobs = OCRJobStore(
    max_jobs=int(os.getenv("OCR_JOB_MAX_JOBS", "500")),
    ttl_seconds=int(os.getenv("OCR_JOB_TTL_SECONDS", "86400")),
)

# Allow simple browser testing from file:// or localhost frontends
# Wide-open CORS keeps local HTML testers (e.g., static/simple.html) working without extra config in dev.
if settings.ALLOW_DEV_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Serve static files (simple test page lives here)
# This lets you open /static/simple.html while the API is running.
static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

create_db_and_tables()


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
    request.state.request_id = req_id
    started = time.perf_counter()
    response: Response
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        if request.url.path.startswith("/ocr"):
            logger.exception(
                "request_failed %s",
                json.dumps(
                    {
                        "request_id": req_id,
                        "path": request.url.path,
                        "method": request.method,
                        "duration_ms": elapsed_ms,
                        "client": request.client.host if request.client else "unknown",
                    },
                    ensure_ascii=True,
                ),
            )
        raise
    response.headers["X-Request-ID"] = req_id
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
    if request.url.path.startswith("/ocr"):
        logger.info(
            "request_done %s",
            json.dumps(
                {
                    "request_id": req_id,
                    "path": request.url.path,
                    "method": request.method,
                    "status_code": response.status_code,
                    "duration_ms": elapsed_ms,
                    "client": request.client.host if request.client else "unknown",
                },
                ensure_ascii=True,
            ),
        )
    return response


def _run_ocr_with_timeout(payload: bytes, filename: str, timeout_s: float):
    """Run OCR in a daemon thread and return promptly on timeout."""
    out: queue.Queue = queue.Queue(maxsize=1)

    def _work():
        try:
            result = ocr_file_to_lines(payload, filename, return_meta=True)
            out.put((True, result))
        except Exception as exc:
            out.put((False, exc))

    t = threading.Thread(target=_work, daemon=True)
    t.start()
    try:
        ok, value = out.get(timeout=timeout_s)
    except queue.Empty as exc:
        raise TimeoutError(f"OCR timed out after {timeout_s:.0f}s") from exc
    if ok:
        return value
    raise value


def _validate_upload_filename(filename: str) -> None:
    if not filename or len(filename) > 200:
        raise HTTPException(status_code=400, detail="invalid filename")
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="invalid filename path")


def _max_ocr_upload_bytes() -> int:
    default_bytes = min(settings.MAX_UPLOAD_BYTES, 5 * 1024 * 1024)
    try:
        val = int(os.getenv("OCR_MAX_UPLOAD_BYTES", str(default_bytes)))
        return max(128 * 1024, val)
    except Exception:
        return default_bytes


def _sniff_upload_kind(payload: bytes, filename: str, content_type: str | None) -> str:
    lower = filename.lower()
    if payload[:4] == b"%PDF" or lower.endswith(".pdf") or content_type == "application/pdf":
        return "pdf"
    try:
        Image.open(io.BytesIO(payload)).verify()
        return "image"
    except Exception:
        raise HTTPException(status_code=415, detail="unsupported file content; expected image or PDF")


def _enforce_ocr_rate_limit(request: Request) -> None:
    max_per_min = int(os.getenv("OCR_RATE_LIMIT_PER_MIN", "60"))
    window = int(os.getenv("OCR_RATE_LIMIT_WINDOW_SECONDS", "60"))
    key = f"{request.client.host if request.client else 'unknown'}:{request.url.path}"
    allowed, retry_after = _ocr_rate_limiter.allow(key, max_per_min, window)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"rate limit exceeded; retry after {retry_after}s",
            headers={"Retry-After": str(retry_after)},
        )


def _ocr_job_worker(payload: bytes, filename: str, expected_text: str | None, note: str | None) -> dict:
    try:
        timeout_s = float(os.getenv("OCR_JOB_TIMEOUT_SECONDS", "120"))
    except Exception:
        timeout_s = 120.0
    lines, engine_used, warnings = _run_ocr_with_timeout(payload, filename, timeout_s)
    out = {
        "engine_used": engine_used,
        "ocr_lines": lines,
        "ocr_text": "\n".join(lines),
        "ocr_warnings": warnings,
        "saved_for_training": False,
    }
    if expected_text is not None and expected_text.strip():
        saved = save_labeled_sample(
            file_bytes=payload,
            filename=filename,
            expected_text=expected_text.strip(),
            predicted_lines=lines,
            engine_used=engine_used,
            warnings=warnings,
            note=note,
        )
        out["saved_for_training"] = True
        out["training_sample"] = saved
    return out

@app.post('/auth/register')
def register(payload: RegisterIn, db: Session = Depends(get_session)):
    """Register a new user (idempotent).

    Returns existing user if the username already exists to make the
    operation idempotent (useful for automation/tests).
    """
    auth = services.AuthService(db)
    existing = repositories.UserRepository(db).get_by_username(payload.username)
    if existing:
        # return existing user (idempotent register)
        return {'id': existing.id, 'username': existing.username}
    user = auth.register(payload.username, payload.password)
    return {'id': user.id, 'username': user.username}

@app.post('/auth/login')
def login(payload: RegisterIn, db: Session = Depends(get_session)):
    """Authenticate a user and return a short-lived JWT token.

    The returned token contains `user_id` and `username` and is signed
    using the configured JWT secret.
    """
    auth = services.AuthService(db)
    token = auth.authenticate(payload.username, payload.password)
    if not token:
        raise HTTPException(status_code=401, detail='invalid credentials')
    return {'access_token': token}

@app.get('/softwaredev/questions')
def list_questions(db: Session = Depends(get_session)):
    """List all questions for the subject `Software Development`.

    The response contains question metadata and the available answers
    for each question.
    """
    qrepo = repositories.QuestionRepository(db)
    qs = qrepo.list_by_subject('Software Development')
    out = []
    for q in qs:
        answers = repositories.AnswerRepository(db).list_for_question(q.id)
        out.append({
            'id': q.id,
            'question_text': q.question_text,
            'difficulty': q.difficulty,
            'question_type': q.question_type,
            'exam_year': q.exam_year,
            'question_number': q.question_number,
            'answers': [{'id': a.id, 'answer_text': a.answer_text} for a in answers]
        })
    return out

@app.post('/softwaredev/import')
def import_questions(file: UploadFile = File(...), db: Session = Depends(get_session), user: models.User = Depends(get_current_user)):
    """Upload a single file and import any questions found.

    The uploaded file may be CSV, TXT, JSON, PDF or DOCX. The endpoint is
    protected and requires authentication (Bearer token).
    Returns a JSON summary with created count and any parsing errors.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail='no file')
    if file.content_type not in ("text/csv", "text/plain", "application/json", "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
        raise HTTPException(status_code=400, detail='unsupported content type')
    content = file.file.read(settings.MAX_UPLOAD_BYTES + 1)
    if len(content) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail='file too large')
    svc = services.ImportService(db)
    try:
        res = svc.import_file(content, file.filename, subject='Software Development', deduplicate=False)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # res: {created, errors, warnings}
    return JSONResponse(status_code=200, content=res)

@app.get('/softwaredev/quiz')
def random_quiz(limit: int = 10, db: Session = Depends(get_session), user: models.User = Depends(get_current_user)):
    """Return a random quiz of `limit` questions for the authenticated user.

    The endpoint is protected and returns the question text and answers.
    """
    qrepo = repositories.QuestionRepository(db)
    qs = qrepo.get_random('Software Development', limit=limit)
    out = []
    for q in qs:
        answers = repositories.AnswerRepository(db).list_for_question(q.id)
        out.append({
            'id': q.id,
            'question_text': q.question_text,
            'exam_year': q.exam_year,
            'question_number': q.question_number,
            'answers': [{'id': a.id, 'answer_text': a.answer_text} for a in answers]
        })
    return out

@app.post('/softwaredev/grade')
def grade(submission: QuizSubmission, db: Session = Depends(get_session), user: models.User = Depends(get_current_user)):
    """Grade a submitted quiz.

    The request body contains a list of {question_id, given_answer} items.
    The service stores the quiz result and per-question outcomes in the
    database and returns a full result summary.
    """
    svc = services.GradingService(db)
    # convert to list of dicts; allow answer_id for stricter grading when clients send it
    answers = [{'question_id': a.question_id, 'given_answer': a.given_answer, 'answer_id': a.answer_id} for a in submission.answers]
    # use authenticated user id
    try:
        result = svc.grade(user.id, answers)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result

@app.post('/goals/set')
def set_goal(week_start: date, goal_type: str, goal_value: int, db: Session = Depends(get_session), user: models.User = Depends(get_current_user)):
    """Set or update a weekly goal for the authenticated user.

    `goal_type` should be a short string like `quizzes` or `questions` and
    `goal_value` is an integer >= 0.
    """
    svc = services.GoalService(db)
    try:
        g = svc.set_weekly_goal(user.id, week_start, goal_type, goal_value)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {'status': 'ok', 'goal_id': g.id}

@app.get('/goals/progress')
def get_progress(week_start: date, db: Session = Depends(get_session), user: models.User = Depends(get_current_user)):
    """Return the current goal progress for the given week start date.

    Progress is computed from stored quiz history for the authenticated user.
    """
    svc = services.GoalService(db)
    p = svc.get_progress(user.id, week_start)
    return p


@app.post('/softwaredev/import_from_exams')
def import_from_exams(year: int = None, db: Session = Depends(get_session), user: models.User = Depends(get_current_user)):
    """Scan the local `Exams/` folder (project root sibling) and import supported files.
    Optional `year` param restricts import to a specific year folder (e.g. 2019).
    """
    exams_root = Path(__file__).resolve().parents[2] / 'Exams'
    if not exams_root.exists():
        raise HTTPException(status_code=400, detail=f'Exams folder not found at {exams_root}')
    files = find_exam_files(exams_root, year=year)
    if not files:
        return {'imported_files': 0, 'details': []}
    svc = services.ImportService(db)
    summary = []
    total_created = 0
    total_skipped = 0
    for f in files:
        try:
            b = f.read_bytes()
            inferred_year = year
            if inferred_year is None:
                try:
                    inferred_year = int(f.parent.name)
                except Exception:
                    inferred_year = None
            result = svc.import_file(b, f.name, subject='Software Development', default_exam_year=inferred_year)
            if isinstance(result, dict):
                total_created += result.get('created', 0)
                total_skipped += result.get('skipped', 0)
            summary.append({'file': str(f), 'result': result})
        except Exception as e:
            summary.append({'file': str(f), 'error': str(e)})
    return {'imported_files': len(files), 'created_questions': total_created, 'skipped_questions': total_skipped, 'details': summary}


@app.get("/", response_class=HTMLResponse)
def home():
    """Minimal homepage for quick manual testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="UTF-8" />
      <title>Study API</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 32px; }
        a { color: #0a6; }
        .card { max-width: 640px; padding: 16px; border: 1px solid #ddd; border-radius: 8px; }
      </style>
    </head>
    <body>
      <div class="card">
        <h1>Software Dev Study API</h1>
        <p>Quick links for local testing:</p>
        <ul>
          <li><a href="/docs">Swagger UI</a></li>
          <li><a href="/static/simple.html">Simple frontend tester</a></li>
          <li><a href="/static/ocr_lab.html">OCR training lab</a></li>
        </ul>
        <p>Use <code>/auth/register</code> + <code>/auth/login</code> to get a token, then try <code>/softwaredev/quiz</code> or <code>/softwaredev/import_from_exams</code>.</p>
      </div>
    </body>
    </html>
    """



@app.get("/ocr")
def ocr_lab_shortcut():
    """Shortcut route to open the OCR training lab page."""
    return RedirectResponse(url="/static/ocr_lab.html")


@app.get("/health")
def health():
    """Lightweight health check for uptime monitoring."""
    return {"status": "ok"}


@app.post('/softwaredev/ocr_grade')
def ocr_grade(
    request: Request,
    question_ids: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_session),
    user: models.User = Depends(get_current_user),
):
    """Grade handwritten answers by OCR-ing an uploaded image/PDF.

    `question_ids` should be a comma-separated list matching the order of answers
    in the uploaded file (one answer per non-empty line after OCR).
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail='no file')
    _enforce_ocr_rate_limit(request)
    _validate_upload_filename(file.filename)
    max_ocr_bytes = _max_ocr_upload_bytes()
    payload = file.file.read(max_ocr_bytes + 1)
    if len(payload) > max_ocr_bytes:
        raise HTTPException(status_code=400, detail='file too large')
    _sniff_upload_kind(payload, file.filename, file.content_type)
    try:
        lines, engine_used, warnings = ocr_file_to_lines(payload, file.filename, return_meta=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")
    q_ids = [int(x) for x in question_ids.split(',') if x.strip()]
    if not q_ids:
        raise HTTPException(status_code=400, detail='question_ids required')
    if len(lines) < len(q_ids):
        raise HTTPException(status_code=400, detail='Not enough answers detected in OCR')
    answers = []
    for idx, qid in enumerate(q_ids):
        answers.append({'question_id': qid, 'given_answer': lines[idx]})
    svc = services.GradingService(db)
    try:
        result = svc.grade(user.id, answers)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    result['ocr_engine_used'] = engine_used
    result['ocr_warnings'] = warnings
    result['ocr_lines'] = lines
    return result


@app.post("/ocr/read")
def ocr_read(
    request: Request,
    file: UploadFile = File(...),
    expected_text: str | None = Form(default=None),
    note: str | None = Form(default=None),
):
    """Direct OCR test endpoint with optional labeled-sample persistence.

    Upload an image/PDF and get extracted lines immediately. If `expected_text`
    is included, the sample is saved to the OCR training dataset folder.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="no file")
    _enforce_ocr_rate_limit(request)
    _validate_upload_filename(file.filename)
    max_ocr_bytes = _max_ocr_upload_bytes()
    payload = file.file.read(max_ocr_bytes + 1)
    if len(payload) > max_ocr_bytes:
        raise HTTPException(status_code=400, detail="file too large")
    _sniff_upload_kind(payload, file.filename, file.content_type)

    try:
        timeout_s = float(os.getenv("OCR_REQUEST_TIMEOUT_SECONDS", "30"))
    except Exception:
        timeout_s = 30.0
    try:
        lines, engine_used, warnings = _run_ocr_with_timeout(payload, file.filename, timeout_s)
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"OCR timed out after {timeout_s:.0f}s. Try a tighter crop or smaller image.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

    out = {
        "engine_used": engine_used,
        "ocr_lines": lines,
        "ocr_text": "\n".join(lines),
        "ocr_warnings": warnings,
        "saved_for_training": False,
    }
    if expected_text is not None and expected_text.strip():
        saved = save_labeled_sample(
            file_bytes=payload,
            filename=file.filename,
            expected_text=expected_text.strip(),
            predicted_lines=lines,
            engine_used=engine_used,
            warnings=warnings,
            note=note,
        )
        out["saved_for_training"] = True
        out["training_sample"] = saved
    return out


@app.post("/ocr/jobs", status_code=202)
def create_ocr_job(
    request: Request,
    file: UploadFile = File(...),
    expected_text: str | None = Form(default=None),
    note: str | None = Form(default=None),
):
    """Queue OCR processing and return a job id for polling."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="no file")
    _enforce_ocr_rate_limit(request)
    _validate_upload_filename(file.filename)
    max_ocr_bytes = _max_ocr_upload_bytes()
    payload = file.file.read(max_ocr_bytes + 1)
    if len(payload) > max_ocr_bytes:
        raise HTTPException(status_code=400, detail="file too large")
    _sniff_upload_kind(payload, file.filename, file.content_type)
    created = _ocr_jobs.submit(
        filename=file.filename,
        payload=payload,
        expected_text=expected_text,
        note=note,
        request_id=getattr(request.state, "request_id", ""),
        worker=_ocr_job_worker,
    )
    return {
        **created,
        "status_url": f"/ocr/jobs/{created['job_id']}",
    }


@app.get("/ocr/jobs/{job_id}")
def get_ocr_job(job_id: str):
    """Poll background OCR job status."""
    job = _ocr_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/ocr/stats")
def ocr_stats():
    """Return OCR aggregate runtime stats from observability logs."""
    return get_ocr_stats()
