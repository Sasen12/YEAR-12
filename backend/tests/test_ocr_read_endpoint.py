import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


client = TestClient(app)


def _make_png() -> bytes:
    img = Image.new("RGB", (60, 40), "white")
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


def test_ocr_read_returns_lines(monkeypatch):
    monkeypatch.setattr("app.main.ocr_file_to_lines", lambda *_args, **_kwargs: (["hello"], "rapidocr", []))
    files = {"file": ("note.png", _make_png(), "image/png")}
    r = client.post("/ocr/read", files=files)
    assert r.status_code == 200
    data = r.json()
    assert data["engine_used"] == "rapidocr"
    assert data["ocr_lines"] == ["hello"]
    assert data["saved_for_training"] is False


def test_ocr_read_saves_training_sample(monkeypatch, tmp_path):
    monkeypatch.setattr("app.main.ocr_file_to_lines", lambda *_args, **_kwargs: (["hello world"], "rapidocr", ["ok"]))
    monkeypatch.setenv("OCR_TRAINING_DIR", str(tmp_path))
    files = {"file": ("note.png", _make_png(), "image/png")}
    data = {"expected_text": "hello world", "note": "clean writing"}
    r = client.post("/ocr/read", files=files, data=data)
    assert r.status_code == 200
    body = r.json()
    assert body["saved_for_training"] is True
    sample = body["training_sample"]

    labels_path = Path(sample["labels_file"])
    assert labels_path.exists()
    lines = labels_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    parsed = json.loads(lines[-1])
    assert parsed["expected_text"] == "hello world"
