"""Helpers for storing OCR training samples collected from manual uploads."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from PIL import Image


def get_training_root() -> Path:
    """Return the base directory for OCR training samples."""
    raw = os.getenv("OCR_TRAINING_DIR", "")
    if raw.strip():
        return Path(raw).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / "data" / "ocr_training").resolve()


def _normalize_expected_text(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if len(cleaned) < 2:
        raise ValueError("expected_text too short")
    if len(cleaned) > 1000:
        raise ValueError("expected_text too long")
    if not re.search(r"[A-Za-z0-9]", cleaned):
        raise ValueError("expected_text must contain alphanumeric characters")
    return cleaned


def _validate_payload_kind(file_bytes: bytes, filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf") or file_bytes[:4] == b"%PDF":
        return "pdf"
    try:
        Image.open(io.BytesIO(file_bytes)).verify()
        return "image"
    except Exception:
        raise ValueError("training sample must be a valid image or PDF")


def _dedupe_key(file_bytes: bytes, expected_text: str) -> str:
    h = hashlib.sha256()
    h.update(file_bytes)
    h.update(b"\n")
    h.update(expected_text.encode("utf-8"))
    return h.hexdigest()


def _load_dedupe_index(root: Path) -> dict:
    idx_path = root / "dedupe_index.json"
    if not idx_path.exists():
        return {}
    try:
        return json.loads(idx_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_dedupe_index(root: Path, data: dict) -> None:
    idx_path = root / "dedupe_index.json"
    idx_path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")


def _assign_split(sample_id: str) -> str:
    bucket = int(sample_id[:8], 16) % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "val"
    return "train"


def _append_split_manifest(root: Path, split: str, sample_id: str) -> None:
    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    target = splits_dir / f"{split}.txt"
    existing = set()
    if target.exists():
        existing = {ln.strip() for ln in target.read_text(encoding="utf-8").splitlines() if ln.strip()}
    if sample_id not in existing:
        with target.open("a", encoding="utf-8") as fh:
            fh.write(sample_id + "\n")


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

    expected_text = _normalize_expected_text(expected_text)
    _validate_payload_kind(file_bytes, filename)

    dedupe = _load_dedupe_index(root)
    key = _dedupe_key(file_bytes, expected_text)
    existing_id = dedupe.get(key)
    if existing_id:
        existing_sample = samples_dir / f"{existing_id}.json"
        if existing_sample.exists():
            existing_data = json.loads(existing_sample.read_text(encoding="utf-8"))
            return {
                "sample_id": existing_id,
                "training_root": str(root),
                "image_path": existing_data.get("image_path"),
                "sample_file": str(existing_sample),
                "labels_file": str(root / "labels.jsonl"),
                "duplicate_of_existing": True,
            }

    suffix = Path(filename).suffix or ".bin"
    sample_id = uuid4().hex
    image_name = f"{sample_id}{suffix.lower()}"
    image_path = images_dir / image_name
    image_path.write_bytes(file_bytes)
    split = _assign_split(sample_id)

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
        "split": split,
        "dedupe_key": key,
    }

    sample_file = samples_dir / f"{sample_id}.json"
    sample_file.write_text(json.dumps(sample, ensure_ascii=True, indent=2), encoding="utf-8")

    labels_path = root / "labels.jsonl"
    with labels_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(sample, ensure_ascii=True) + "\n")
    dedupe[key] = sample_id
    _save_dedupe_index(root, dedupe)
    _append_split_manifest(root, split, sample_id)

    return {
        "sample_id": sample_id,
        "training_root": str(root),
        "image_path": str(image_path),
        "sample_file": str(sample_file),
        "labels_file": str(labels_path),
        "split": split,
        "duplicate_of_existing": False,
    }
