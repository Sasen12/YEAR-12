import io

import numpy as np
from PIL import Image

from app.utils import ocr


def _png_bytes() -> bytes:
    img = Image.new("RGB", (80, 30), "white")
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


def test_parse_engines_default_includes_rapidocr(monkeypatch):
    monkeypatch.delenv("OCR_ENGINES", raising=False)
    engines = ocr._parse_engines()
    assert engines
    assert engines[0] == "rapidocr"


def test_run_engine_best_variant_prefers_higher_quality(monkeypatch):
    variants = [("v1", np.zeros((10, 10), dtype=np.uint8)), ("v2", np.zeros((20, 20), dtype=np.uint8))]

    def fake_run_engine(_engine, img_arr):
        if img_arr.shape[0] == 10 and img_arr.shape[1] == 10:
            return (["a", "b"], [0.95, 0.95])
        return (["x"], [0.20])

    monkeypatch.setattr(ocr, "_run_engine_on_array", fake_run_engine)
    result = ocr._run_engine_best_variant("rapidocr", variants)
    assert result is not None
    assert result.lines == ["a", "b"]


def test_ocr_file_to_lines_adds_low_confidence_warning(monkeypatch):
    payload = _png_bytes()
    monkeypatch.setenv("OCR_MIN_CONFIDENCE", "0.90")
    monkeypatch.setattr(
        ocr,
        "_run_engine_best_variant",
        lambda _engine, _variants: ocr.OCRResult(lines=["detected text"], confidences=[0.40], variant="grayscale"),
    )

    lines, engine, warnings = ocr.ocr_file_to_lines(payload, "note.png", engines=["rapidocr"], return_meta=True)
    assert lines == ["detected text"]
    assert engine == "rapidocr"
    assert any("manual review recommended" in w for w in warnings)
