"""OCR helpers for extracting text from uploaded answer sheets.

We support multiple engines in priority order. Configure via env `OCR_ENGINES`
comma-separated (e.g., "tesseract,paddle,easyocr,troc"). Defaults to
["tesseract"] to keep dependencies light. Engines are attempted until one
returns non-empty lines. Heavy engines are lazy-imported and optional.
"""

import io
import os
from typing import List, Tuple

from PIL import Image
import pdfplumber
import numpy as np

# Optional engines: pytesseract (default), easyocr, paddleocr, trOCR
try:
    import pytesseract
except Exception as e:  # pragma: no cover
    pytesseract = None  # type: ignore
    _pytesseract_err = e
else:
    _pytesseract_err = None


def _parse_engines() -> List[str]:
    raw = os.getenv("OCR_ENGINES", "tesseract")
    return [e.strip().lower() for e in raw.split(",") if e.strip()]


def ocr_file_to_lines(file_bytes: bytes, filename: str, engines: List[str] | None = None, return_meta: bool = False):
    """Extract text lines from an uploaded file using the first working engine.

    Returns lines (and optionally engine_used, warnings list when return_meta=True).
    """
    name = filename.lower()
    # PDF: try text layer first; if empty, convert pages to images and OCR
    if name.endswith('.pdf'):
        lines = _pdf_to_lines(file_bytes)
        if lines:
            return (lines, 'pdf_text', []) if return_meta else lines
        # fall back to image OCR on rendered pages if needed
        # (not implemented here to keep runtime light)
    engines = engines or _parse_engines()
    last_err = None
    warnings = []
    for eng in engines:
        try:
            if eng == "tesseract":
                lines = _image_to_lines_tesseract(file_bytes)
            elif eng == "easyocr":
                lines = _image_to_lines_easyocr(file_bytes)
            elif eng == "paddle":
                lines = _image_to_lines_paddle(file_bytes)
            elif eng == "troc":
                lines = _image_to_lines_troc(file_bytes)
            else:
                continue
            else:
                continue
            if lines:
                return (lines, eng, warnings) if return_meta else lines
            else:
                warnings.append(f"{eng} returned no text")
        except Exception as e:  # pragma: no cover - best effort
            last_err = e
            warnings.append(f"{eng} failed: {e}")
            continue
    if last_err:
        raise RuntimeError(f"OCR failed with engines {engines}: {last_err}")
    raise RuntimeError(f"OCR failed; no engines available ({engines})")


def _pdf_to_lines(b: bytes) -> List[str]:
    out = []
    with pdfplumber.open(io.BytesIO(b)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ''
            out.extend(text.splitlines())
    return [ln.strip() for ln in out if ln.strip()]


def _image_to_lines_tesseract(b: bytes) -> List[str]:
    if pytesseract is None:
        raise RuntimeError(f"pytesseract not available: {_pytesseract_err}")
    img = Image.open(io.BytesIO(b)).convert("RGB")
    text = pytesseract.image_to_string(img)
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _image_to_lines_easyocr(b: bytes) -> List[str]:
    import easyocr  # type: ignore
    img = Image.open(io.BytesIO(b)).convert("RGB")
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(np.array(img), detail=0)  # type: ignore
    return [ln.strip() for ln in result if str(ln).strip()]


def _image_to_lines_paddle(b: bytes) -> List[str]:
    from paddleocr import PaddleOCR  # type: ignore
    img = Image.open(io.BytesIO(b)).convert("RGB")
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
    result = ocr.ocr(np.array(img), cls=True)  # type: ignore
    lines = []
    for res in result:
        for line in res:
            txt = line[1][0]
            if txt:
                lines.append(txt.strip())
    return [ln for ln in lines if ln]


def _image_to_lines_troc(b: bytes) -> List[str]:
    # TrOCR via transformers pipeline; requires torch + transformers installed.
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # type: ignore
    import torch  # type: ignore
    img = Image.open(io.BytesIO(b)).convert("RGB")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    pixel_values = processor(images=img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return [ln.strip() for ln in text.splitlines() if ln.strip()]
