"""OCR helpers for extracting text from uploaded answer sheets.

Pipeline:
- PDF text layer extraction first.
- Image OCR with multiple preprocess variants.
- Engine fallback order (default): rapidocr -> tesseract.
- Confidence-based warnings for low-trust results.
"""

import io
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pdfplumber
from PIL import Image, ImageOps

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import pytesseract
except Exception as e:  # pragma: no cover
    pytesseract = None  # type: ignore
    _pytesseract_err = e
else:
    _pytesseract_err = None

_rapidocr_engine = None


@dataclass
class OCRResult:
    lines: List[str]
    confidences: List[float]
    variant: str


def _parse_engines() -> List[str]:
    raw = os.getenv("OCR_ENGINES", "rapidocr,tesseract")
    return [e.strip().lower() for e in raw.split(",") if e.strip()]


def _parse_min_confidence() -> float:
    try:
        val = float(os.getenv("OCR_MIN_CONFIDENCE", "0.72"))
        return max(0.0, min(1.0, val))
    except Exception:
        return 0.72


def ocr_file_to_lines(file_bytes: bytes, filename: str, engines: List[str] | None = None, return_meta: bool = False):
    """Extract text lines from an uploaded file using the first good OCR engine.

    Returns lines; if `return_meta=True`, returns `(lines, engine_used, warnings)`.
    """
    name = filename.lower()
    warnings: List[str] = []

    if name.endswith(".pdf"):
        lines = _pdf_to_lines(file_bytes)
        if lines:
            return (lines, "pdf_text", warnings) if return_meta else lines
        warnings.append("pdf_text returned no text; falling back to OCR")

    image = _load_image(file_bytes)
    variants = _build_variants(image)
    engines = engines or _parse_engines()
    last_err: Optional[Exception] = None
    attempts = 0
    min_conf = _parse_min_confidence()

    for eng in engines:
        try:
            attempts += 1
            result = _run_engine_best_variant(eng, variants)
            if result and result.lines:
                avg_conf = (sum(result.confidences) / len(result.confidences)) if result.confidences else 0.0
                if avg_conf < min_conf:
                    warnings.append(
                        f"{eng} low confidence avg={avg_conf:.2f} below threshold={min_conf:.2f}; manual review recommended"
                    )
                warnings.append(f"{eng} best preprocess variant: {result.variant} (avg_conf={avg_conf:.2f})")
                return (result.lines, eng, warnings) if return_meta else result.lines
            warnings.append(f"{eng} returned no text")
        except Exception as e:  # pragma: no cover
            last_err = e
            warnings.append(f"{eng} failed: {e}")

    if attempts > 0:
        if return_meta:
            return ([], "none", warnings)
        return []
    if last_err:
        raise RuntimeError(f"OCR failed with engines {engines}: {last_err}")
    raise RuntimeError(f"OCR failed; no OCR engine returned text ({engines})")


def _load_image(b: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(b)).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Unsupported or invalid image input: {exc}") from exc


def _pdf_to_lines(b: bytes) -> List[str]:
    out = []
    with pdfplumber.open(io.BytesIO(b)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            out.extend(text.splitlines())
    return [ln.strip() for ln in out if ln.strip()]


def _build_variants(image: Image.Image) -> List[tuple[str, np.ndarray]]:
    """Build preprocessing variants to improve handwriting OCR robustness."""
    variants: List[tuple[str, np.ndarray]] = []
    rgb = np.array(image)
    variants.append(("original", rgb))

    gray_img = ImageOps.grayscale(image)
    gray = np.array(gray_img)
    variants.append(("grayscale", gray))

    boosted = ImageOps.autocontrast(gray_img)
    boosted_np = np.array(boosted)
    variants.append(("autocontrast", boosted_np))

    if cv2 is not None:
        blur = cv2.GaussianBlur(boosted_np, (3, 3), 0)
        th = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        variants.append(("adaptive_threshold", th))
    else:
        simple_th = boosted.point(lambda p: 255 if p > 145 else 0)
        variants.append(("threshold", np.array(simple_th)))

    return variants


def _run_engine_best_variant(engine_name: str, variants: List[tuple[str, np.ndarray]]) -> Optional[OCRResult]:
    best: Optional[OCRResult] = None
    best_score = -1.0

    for variant_name, arr in variants:
        lines, confidences = _run_engine_on_array(engine_name, arr)
        if not lines:
            continue
        avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0
        score = len(lines) * (0.5 + avg_conf)
        if score > best_score:
            best_score = score
            best = OCRResult(lines=lines, confidences=confidences, variant=variant_name)

    return best


def _run_engine_on_array(engine_name: str, img_arr: np.ndarray) -> tuple[List[str], List[float]]:
    if engine_name == "rapidocr":
        return _image_to_lines_rapidocr(img_arr)
    if engine_name == "tesseract":
        return _image_to_lines_tesseract(img_arr)
    if engine_name == "paddle":
        return _image_to_lines_paddle(img_arr)
    if engine_name == "easyocr":
        return _image_to_lines_easyocr(img_arr)
    if engine_name == "troc":
        return _image_to_lines_troc(img_arr)
    raise RuntimeError(f"unknown OCR engine: {engine_name}")


def _get_rapidocr():
    global _rapidocr_engine
    if _rapidocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore

        _rapidocr_engine = RapidOCR()
    return _rapidocr_engine


def _normalize_lines(lines: List[str]) -> List[str]:
    return [ln.strip() for ln in lines if isinstance(ln, str) and ln.strip()]


def _image_to_lines_rapidocr(img_arr: np.ndarray) -> tuple[List[str], List[float]]:
    engine = _get_rapidocr()
    result, _ = engine(img_arr)
    lines: List[str] = []
    confs: List[float] = []
    for item in result or []:
        if len(item) < 3:
            continue
        text = str(item[1]).strip()
        if not text:
            continue
        lines.append(text)
        try:
            confs.append(float(item[2]))
        except Exception:
            confs.append(0.0)
    return _normalize_lines(lines), confs


def _image_to_lines_tesseract(img_arr: np.ndarray) -> tuple[List[str], List[float]]:
    if pytesseract is None:
        raise RuntimeError(f"pytesseract not available: {_pytesseract_err}")
    pil = Image.fromarray(img_arr if img_arr.ndim == 2 else img_arr.astype(np.uint8)).convert("RGB")
    data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT)
    lines: List[str] = []
    confs: List[float] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        conf = data.get("conf", ["-1"])[i]
        try:
            conf_float = max(0.0, min(1.0, float(conf) / 100.0))
        except Exception:
            conf_float = 0.0
        lines.append(text)
        confs.append(conf_float)
    if lines:
        return [" ".join(lines)], [sum(confs) / len(confs)]
    return [], []


def _image_to_lines_easyocr(img_arr: np.ndarray) -> tuple[List[str], List[float]]:
    import easyocr  # type: ignore

    reader = easyocr.Reader(["en"], gpu=False)
    result = reader.readtext(img_arr, detail=1)  # type: ignore
    lines = []
    confs = []
    for item in result:
        if len(item) < 3:
            continue
        text = str(item[1]).strip()
        if not text:
            continue
        lines.append(text)
        try:
            confs.append(float(item[2]))
        except Exception:
            confs.append(0.0)
    return _normalize_lines(lines), confs


def _image_to_lines_paddle(img_arr: np.ndarray) -> tuple[List[str], List[float]]:
    from paddleocr import PaddleOCR  # type: ignore

    ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)
    result = ocr.ocr(img_arr, cls=True)  # type: ignore
    lines = []
    confs = []
    for res in result or []:
        for line in res:
            txt = (line[1][0] or "").strip()
            if not txt:
                continue
            lines.append(txt)
            try:
                confs.append(float(line[1][1]))
            except Exception:
                confs.append(0.0)
    return _normalize_lines(lines), confs


def _image_to_lines_troc(img_arr: np.ndarray) -> tuple[List[str], List[float]]:
    # TrOCR via transformers; confidence not surfaced by this path.
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # type: ignore

    img = Image.fromarray(img_arr if img_arr.ndim == 3 else np.stack([img_arr] * 3, axis=-1)).convert("RGB")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    pixel_values = processor(images=img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    confs = [0.7 for _ in lines]
    return lines, confs
