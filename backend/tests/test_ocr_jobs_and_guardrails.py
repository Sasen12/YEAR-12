import io
import time
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.utils.rate_limit import InMemoryRateLimiter
from app.utils.ocr_training import save_labeled_sample


client = TestClient(app)


def _make_png() -> bytes:
    img = Image.new("RGB", (64, 32), "white")
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


def test_ocr_jobs_flow(monkeypatch, tmp_path):
    monkeypatch.setenv("OCR_TRAINING_DIR", str(tmp_path))
    monkeypatch.setattr("app.main.ocr_file_to_lines", lambda *_args, **_kwargs: (["nominal"], "rapidocr", []))
    files = {"file": ("note.png", _make_png(), "image/png")}
    data = {"expected_text": "nominal"}
    created = client.post("/ocr/jobs", files=files, data=data)
    assert created.status_code == 202
    job_id = created.json()["job_id"]

    deadline = time.time() + 5
    status = None
    while time.time() < deadline:
        poll = client.get(f"/ocr/jobs/{job_id}")
        assert poll.status_code == 200
        status = poll.json()["status"]
        if status in ("succeeded", "failed"):
            break
        time.sleep(0.05)

    assert status == "succeeded"
    final = client.get(f"/ocr/jobs/{job_id}").json()
    assert final["result"]["ocr_lines"] == ["nominal"]


def test_training_guardrails_dedupe_and_split(tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_TRAINING_DIR", str(tmp_path))
    payload = _make_png()
    sample1 = save_labeled_sample(
        file_bytes=payload,
        filename="a.png",
        expected_text="ordinal",
        predicted_lines=["ordinal"],
        engine_used="rapidocr",
        warnings=[],
        note="n1",
    )
    sample2 = save_labeled_sample(
        file_bytes=payload,
        filename="a.png",
        expected_text="ordinal",
        predicted_lines=["ordinal"],
        engine_used="rapidocr",
        warnings=[],
        note="n2",
    )
    assert sample1["sample_id"] == sample2["sample_id"]
    assert sample2["duplicate_of_existing"] is True
    split_file = Path(sample1["training_root"]) / "splits" / f"{sample1['split']}.txt"
    assert split_file.exists()
    assert sample1["sample_id"] in split_file.read_text(encoding="utf-8")


def test_ocr_read_rejects_invalid_upload():
    files = {"file": ("bad.txt", b"not an image", "text/plain")}
    r = client.post("/ocr/read", files=files)
    assert r.status_code == 415


def test_request_id_header_exists():
    r = client.get("/health")
    assert r.status_code == 200
    assert "X-Request-ID" in r.headers


def test_ocr_rate_limit(monkeypatch):
    monkeypatch.setattr("app.main._ocr_rate_limiter", InMemoryRateLimiter())
    monkeypatch.setenv("OCR_RATE_LIMIT_PER_MIN", "1")
    monkeypatch.setenv("OCR_RATE_LIMIT_WINDOW_SECONDS", "60")
    monkeypatch.setattr("app.main.ocr_file_to_lines", lambda *_args, **_kwargs: (["x"], "rapidocr", []))
    files = {"file": ("note.png", _make_png(), "image/png")}
    first = client.post("/ocr/read", files=files)
    assert first.status_code == 200
    second = client.post("/ocr/read", files=files)
    assert second.status_code == 429
