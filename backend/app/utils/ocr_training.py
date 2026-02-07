"""Helpers for storing OCR training samples collected from manual uploads."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def get_training_root() -> Path:
    """Return the base directory for OCR training samples."""
    raw = os.getenv("OCR_TRAINING_DIR", "")
    if raw.strip():
        return Path(raw).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / "data" / "ocr_training").resolve()


def save_labeled_sample(
    *,
    file_bytes: bytes,
    filename: str,
    expected_text: str,
    predicted_lines: list[str],
    engine_used: str,
    warnings: list[str],
    note: str | None = None,
) -> dict:
    """Persist an OCR sample image and metadata for later model training."""
    root = get_training_root()
    images_dir = root / "images"
    samples_dir = root / "samples"
    images_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(filename).suffix or ".bin"
    sample_id = uuid4().hex
    image_name = f"{sample_id}{suffix.lower()}"
    image_path = images_dir / image_name
    image_path.write_bytes(file_bytes)

    sample = {
        "id": sample_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "image_path": str(image_path),
        "expected_text": expected_text,
        "predicted_lines": predicted_lines,
        "predicted_text": "\n".join(predicted_lines),
        "engine_used": engine_used,
        "warnings": warnings,
        "note": note or "",
    }

    sample_file = samples_dir / f"{sample_id}.json"
    sample_file.write_text(json.dumps(sample, ensure_ascii=True, indent=2), encoding="utf-8")

    labels_path = root / "labels.jsonl"
    with labels_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(sample, ensure_ascii=True) + "\n")

    return {
        "sample_id": sample_id,
        "training_root": str(root),
        "image_path": str(image_path),
        "sample_file": str(sample_file),
        "labels_file": str(labels_path),
    }
