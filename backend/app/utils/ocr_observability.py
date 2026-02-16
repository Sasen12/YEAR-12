"""Structured OCR observability helpers."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


_WRITE_LOCK = Lock()
_LOGGER = logging.getLogger("app.ocr")


def _observability_root() -> Path:
    raw = os.getenv("OCR_OBSERVABILITY_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "data" / "ocr_observability"


def _events_path() -> Path:
    root = _observability_root()
    root.mkdir(parents=True, exist_ok=True)
    return root / "ocr_events.jsonl"


def _stats_path() -> Path:
    root = _observability_root()
    root.mkdir(parents=True, exist_ok=True)
    return root / "ocr_stats.json"


def get_ocr_stats() -> dict:
    """Return current OCR aggregate stats snapshot."""
    stats_file = _stats_path()
    if not stats_file.exists():
        return {
            "total_runs": 0,
            "failed_runs": 0,
            "slow_runs": 0,
            "avg_duration_ms": 0.0,
            "max_duration_ms": 0.0,
            "last_error": "",
        }
    try:
        return json.loads(stats_file.read_text(encoding="utf-8"))
    except Exception:
        return {
            "total_runs": 0,
            "failed_runs": 0,
            "slow_runs": 0,
            "avg_duration_ms": 0.0,
            "max_duration_ms": 0.0,
            "last_error": "",
        }


def record_ocr_event(event: dict) -> None:
    """Write event to jsonl and maintain a tiny aggregate stats file."""
    payload = dict(event)
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    with _WRITE_LOCK:
        with _events_path().open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
        _update_stats(payload)
    _LOGGER.info("ocr_event %s", json.dumps(payload, ensure_ascii=True))


def _update_stats(event: dict) -> None:
    stats_file = _stats_path()
    if stats_file.exists():
        try:
            stats = json.loads(stats_file.read_text(encoding="utf-8"))
        except Exception:
            stats = {}
    else:
        stats = {}
    stats.setdefault("total_runs", 0)
    stats.setdefault("failed_runs", 0)
    stats.setdefault("slow_runs", 0)
    stats.setdefault("total_duration_ms", 0.0)
    stats.setdefault("max_duration_ms", 0.0)
    stats.setdefault("last_error", "")

    duration_ms = float(event.get("duration_ms", 0.0))
    is_failed = bool(event.get("failed", False))
    slow_threshold = float(os.getenv("OCR_SLOW_MS", "8000"))
    is_slow = duration_ms >= slow_threshold

    stats["total_runs"] += 1
    stats["total_duration_ms"] += duration_ms
    stats["max_duration_ms"] = max(float(stats["max_duration_ms"]), duration_ms)
    if is_failed:
        stats["failed_runs"] += 1
        stats["last_error"] = str(event.get("error", ""))
    if is_slow:
        stats["slow_runs"] += 1
    stats["avg_duration_ms"] = (stats["total_duration_ms"] / stats["total_runs"]) if stats["total_runs"] else 0.0
    stats["updated_at"] = datetime.now(timezone.utc).isoformat()
    stats_file.write_text(json.dumps(stats, ensure_ascii=True, indent=2), encoding="utf-8")
