"""In-memory background job store for OCR tasks."""

from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional


class OCRJobStore:
    def __init__(self, max_jobs: int = 500, ttl_seconds: int = 24 * 3600):
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._max_jobs = max_jobs
        self._ttl_seconds = ttl_seconds

    def submit(
        self,
        *,
        filename: str,
        payload: bytes,
        expected_text: Optional[str],
        note: Optional[str],
        request_id: str,
        worker: Callable[[bytes, str, Optional[str], Optional[str]], dict],
    ) -> dict:
        self._cleanup()
        job_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        job = {
            "job_id": job_id,
            "status": "queued",
            "filename": filename,
            "created_at": now,
            "started_at": None,
            "finished_at": None,
            "request_id": request_id,
            "result": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job
            if len(self._jobs) > self._max_jobs:
                # remove oldest finished first
                finished = sorted(
                    (j for j in self._jobs.values() if j.get("finished_at")),
                    key=lambda x: x.get("finished_at") or "",
                )
                for old in finished[: max(0, len(self._jobs) - self._max_jobs)]:
                    self._jobs.pop(old["job_id"], None)

        thread = threading.Thread(
            target=self._run_job,
            kwargs={
                "job_id": job_id,
                "payload": payload,
                "filename": filename,
                "expected_text": expected_text,
                "note": note,
                "worker": worker,
            },
            daemon=True,
        )
        thread.start()
        return {"job_id": job_id, "status": "queued"}

    def get(self, job_id: str) -> Optional[dict]:
        self._cleanup()
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def _run_job(
        self,
        *,
        job_id: str,
        payload: bytes,
        filename: str,
        expected_text: Optional[str],
        note: Optional[str],
        worker: Callable[[bytes, str, Optional[str], Optional[str]], dict],
    ) -> None:
        with self._lock:
            if job_id not in self._jobs:
                return
            self._jobs[job_id]["status"] = "running"
            self._jobs[job_id]["started_at"] = datetime.now(timezone.utc).isoformat()
        try:
            result = worker(payload, filename, expected_text, note)
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id]["status"] = "succeeded"
                    self._jobs[job_id]["result"] = result
                    self._jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()
        except Exception as exc:
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id]["status"] = "failed"
                    self._jobs[job_id]["error"] = str(exc)
                    self._jobs[job_id]["finished_at"] = datetime.now(timezone.utc).isoformat()

    def _cleanup(self) -> None:
        cutoff = time.time() - self._ttl_seconds
        with self._lock:
            to_delete = []
            for job_id, job in self._jobs.items():
                finished = job.get("finished_at")
                if not finished:
                    continue
                try:
                    ts = datetime.fromisoformat(finished).timestamp()
                except Exception:
                    ts = 0
                if ts < cutoff:
                    to_delete.append(job_id)
            for job_id in to_delete:
                self._jobs.pop(job_id, None)
