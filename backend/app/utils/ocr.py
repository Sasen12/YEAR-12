"""OCR helpers for extracting text from uploaded answer sheets.

Accuracy strategy:
- Extract PDF text layer first when available.
- Build multiple image variants (cleanup + contrast + threshold + rotations).
- Run multiple OCR engines and pick the best global candidate by score.
- Apply optional lexicon correction using built-ins + collected training labels.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
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


@dataclass
class OCRCandidate:
    engine: str
    variant: str
    lines: List[str]
    confidences: List[float]
    score: float


_BUILTIN_LEXICON = {
    "nominal",
    "ordinal",
    "interval",
    "ratio",
    "categorical",
    "quantitative",
    "qualitative",
    "discrete",
    "continuous",
    "mean",
    "median",
    "mode",
    "probability",
    "software",
    "development",
    "algorithm",
    "database",
}


def _parse_engines() -> List[str]:
    raw = os.getenv("OCR_ENGINES", "rapidocr")
    return [e.strip().lower() for e in raw.split(",") if e.strip()]


def _parse_min_confidence() -> float:
    try:
        val = float(os.getenv("OCR_MIN_CONFIDENCE", "0.68"))
        return max(0.0, min(1.0, val))
    except Exception:
        return 0.68


def _parse_use_lexicon_correction() -> bool:
    return os.getenv("OCR_LEXICON_CORRECTION", "true").strip().lower() == "true"


def _parse_lexicon_similarity_threshold() -> float:
    try:
        val = float(os.getenv("OCR_LEXICON_MIN_SIMILARITY", "0.70"))
        return max(0.0, min(1.0, val))
    except Exception:
        return 0.70


def _parse_max_image_side() -> int:
    try:
        val = int(os.getenv("OCR_MAX_IMAGE_SIDE", "1400"))
        return max(600, min(4000, val))
    except Exception:
        return 1400


def ocr_file_to_lines(file_bytes: bytes, filename: str, engines: List[str] | None = None, return_meta: bool = False):
    """Extract text lines from an uploaded file.

    Returns `lines` or `(lines, engine_used, warnings)` when `return_meta=True`.
    """
    name = filename.lower()
    warnings: List[str] = []

    if name.endswith(".pdf"):
        lines = _pdf_to_lines(file_bytes)
        if lines:
            return (lines, "pdf_text", warnings) if return_meta else lines
        warnings.append("pdf_text returned no text; falling back to OCR")

    image = _load_image(file_bytes)
    image = _downscale_if_needed(image, max_side=_parse_max_image_side())
    engines = engines or _parse_engines()
    min_conf = _parse_min_confidence()

    # Fast path first: lightweight RapidOCR sweep on core rotations.
    quick = _quick_rapidocr_pass(image)
    if quick:
        q_lines, q_variant, q_avg_conf = quick
        warnings.append(f"fast-path: rapidocr {q_variant} avg_conf={q_avg_conf:.2f}")
        corrected_lines = q_lines
        if _parse_use_lexicon_correction():
            corrected_lines, corrections = _lexicon_correct_lines(q_lines, q_avg_conf)
            if corrections:
                warnings.append("lexicon corrections: " + ", ".join(corrections))
        if q_avg_conf >= min_conf:
            if return_meta:
                return (corrected_lines, "rapidocr", warnings)
            return corrected_lines

    variants = _build_variants(image)

    candidates: List[OCRCandidate] = []
    for eng in engines:
        try:
            engine_variants = _select_variants_for_engine(eng, variants)
            result = _run_engine_best_variant(eng, engine_variants)
            if not result or not result.lines:
                warnings.append(f"{eng} returned no text")
                continue
            score = _score_candidate(result.lines, result.confidences)
            candidates.append(
                OCRCandidate(
                    engine=eng,
                    variant=result.variant,
                    lines=result.lines,
                    confidences=result.confidences,
                    score=score,
                )
            )
        except Exception as e:  # pragma: no cover
            warnings.append(f"{eng} failed: {e}")

    if not candidates:
        if return_meta:
            return ([], "none", warnings)
        return []

    best = max(candidates, key=lambda c: c.score)
    avg_conf = (sum(best.confidences) / len(best.confidences)) if best.confidences else 0.0
    warnings.append(f"best candidate: engine={best.engine}, variant={best.variant}, score={best.score:.2f}, avg_conf={avg_conf:.2f}")
    if avg_conf < min_conf:
        warnings.append(f"low confidence avg={avg_conf:.2f} below threshold={min_conf:.2f}; manual review recommended")

    corrected_lines = best.lines
    if _parse_use_lexicon_correction():
        corrected_lines, corrections = _lexicon_correct_lines(best.lines, avg_conf)
        if corrections:
            warnings.append("lexicon corrections: " + ", ".join(corrections))

    if return_meta:
        return (corrected_lines, best.engine, warnings)
    return corrected_lines


def _load_image(b: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(b)).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Unsupported or invalid image input: {exc}") from exc


def _downscale_if_needed(image: Image.Image, max_side: int) -> Image.Image:
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _pdf_to_lines(b: bytes) -> List[str]:
    out = []
    with pdfplumber.open(io.BytesIO(b)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            out.extend(text.splitlines())
    return [ln.strip() for ln in out if ln.strip()]


def _build_variants(image: Image.Image) -> List[tuple[str, np.ndarray]]:
    """Build preprocessing variants including rotations for orientation robustness."""
    base_variants: List[tuple[str, np.ndarray]] = []
    rgb = np.array(image)
    rgb = _remove_blue_grid_lines(rgb)
    base_variants.append(("cleaned", rgb))

    gray_img = ImageOps.grayscale(Image.fromarray(rgb))
    gray = np.array(gray_img)
    base_variants.append(("grayscale", gray))

    boosted = ImageOps.autocontrast(gray_img)
    boosted_np = np.array(boosted)
    base_variants.append(("autocontrast", boosted_np))

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
        base_variants.append(("adaptive_threshold", th))
    else:
        simple_th = boosted.point(lambda p: 255 if p > 145 else 0)
        base_variants.append(("threshold", np.array(simple_th)))

    variants: List[tuple[str, np.ndarray]] = []
    for name, arr in base_variants:
        variants.append((f"{name}_r0", arr))
        variants.append((f"{name}_r90", np.rot90(arr, 1)))
        variants.append((f"{name}_r180", np.rot90(arr, 2)))
        variants.append((f"{name}_r270", np.rot90(arr, 3)))
    return variants


def _select_variants_for_engine(engine_name: str, variants: List[tuple[str, np.ndarray]]) -> List[tuple[str, np.ndarray]]:
    if engine_name == "rapidocr":
        return variants
    if engine_name == "tesseract":
        preferred = []
        for name, arr in variants:
            if name.startswith("autocontrast_") or name.startswith("adaptive_threshold_"):
                preferred.append((name, arr))
        # Keep a bounded set for speed.
        bounded = [v for v in preferred if v[0].endswith("_r0") or v[0].endswith("_r90") or v[0].endswith("_r270")]
        return bounded[:6] if bounded else variants[:6]
    return variants


def _quick_rapidocr_pass(image: Image.Image) -> Optional[tuple[List[str], str, float]]:
    base = ImageOps.autocontrast(ImageOps.grayscale(image))
    best_lines: List[str] = []
    best_variant = ""
    best_conf = 0.0
    best_score = -1.0
    for angle in (0, 90, 270, 180):
        arr = np.array(base.rotate(angle, expand=True))
        lines, confs = _image_to_lines_rapidocr(arr)
        if not lines:
            continue
        score = _score_candidate(lines, confs)
        avg_conf = (sum(confs) / len(confs)) if confs else 0.0
        if score > best_score:
            best_score = score
            best_lines = lines
            best_variant = f"autocontrast_r{angle}"
            best_conf = avg_conf
    if not best_lines:
        return None
    return best_lines, best_variant, best_conf


def _remove_blue_grid_lines(arr: np.ndarray) -> np.ndarray:
    """Reduce notebook/grid blue line noise that often distorts handwriting OCR."""
    if cv2 is None or arr.ndim != 3:
        return arr
    try:
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        lower_blue = np.array([75, 35, 35], dtype=np.uint8)
        upper_blue = np.array([140, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        mask = cv2.dilate(mask, np.ones((2, 2), dtype=np.uint8), iterations=1)
        cleaned = cv2.inpaint(arr, mask, 3, cv2.INPAINT_TELEA)
        return cleaned
    except Exception:
        return arr


def _run_engine_best_variant(engine_name: str, variants: List[tuple[str, np.ndarray]]) -> Optional[OCRResult]:
    best: Optional[OCRResult] = None
    best_score = -1.0

    for variant_name, arr in variants:
        lines, confidences = _run_engine_on_array(engine_name, arr)
        if not lines:
            continue
        score = _score_candidate(lines, confidences)
        if score > best_score:
            best_score = score
            best = OCRResult(lines=lines, confidences=confidences, variant=variant_name)

    return best


def _score_candidate(lines: List[str], confidences: List[float]) -> float:
    text = " ".join(lines).strip()
    if not text:
        return -1.0
    avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0
    if len(text) < 3:
        return (avg_conf * 10.0) - 20.0

    words = re.findall(r"[A-Za-z0-9]+", text)
    word_count = len(words)
    longest_word = max((len(w) for w in words), default=0)
    alpha_chars = sum(1 for ch in text if ch.isalpha())
    digit_chars = sum(1 for ch in text if ch.isdigit())
    penalty = sum(1 for ch in text if ch in "|~_") * 0.5
    short_word_penalty = 18.0 if longest_word < 3 else 0.0
    return (avg_conf * 40.0) + (alpha_chars * 0.8) + (digit_chars * 0.25) + (word_count * 8.0) - penalty - short_word_penalty


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
    _ensure_tesseract_cmd()
    pil = Image.fromarray(img_arr if img_arr.ndim == 2 else img_arr.astype(np.uint8)).convert("RGB")
    best_lines: List[str] = []
    best_conf: List[float] = []
    for psm in ("6", "7", "11"):
        cfg = f"--oem 3 --psm {psm}"
        data = pytesseract.image_to_data(pil, config=cfg, output_type=pytesseract.Output.DICT)
        words: List[str] = []
        confs: List[float] = []
        n = len(data.get("text", []))
        for i in range(n):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            conf_raw = data.get("conf", ["-1"])[i]
            try:
                conf_float = max(0.0, min(1.0, float(conf_raw) / 100.0))
            except Exception:
                conf_float = 0.0
            words.append(text)
            confs.append(conf_float)
        if words:
            merged = [" ".join(words)]
            if _score_candidate(merged, confs) > _score_candidate(best_lines, best_conf):
                best_lines = merged
                best_conf = [sum(confs) / len(confs)] if confs else [0.0]
    return best_lines, best_conf


def _ensure_tesseract_cmd() -> None:
    """Configure pytesseract binary path when PATH is not refreshed yet."""
    if pytesseract is None:
        return
    detected = shutil.which("tesseract")
    if detected:
        pytesseract.pytesseract.tesseract_cmd = detected
        return
    candidates = [
        os.getenv("TESSERACT_CMD", "").strip(),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            pytesseract.pytesseract.tesseract_cmd = candidate
            return


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


def _lexicon_correct_lines(lines: List[str], avg_conf: float) -> tuple[List[str], List[str]]:
    """Apply token-level correction against a dynamic lexicon."""
    if not lines:
        return lines, []
    # Only correct aggressively when confidence is not high.
    if avg_conf >= 0.92:
        return lines, []

    lexicon = _load_lexicon()
    if not lexicon:
        return lines, []

    sim_threshold = _parse_lexicon_similarity_threshold()
    corrections: List[str] = []
    corrected_lines: List[str] = []
    token_pattern = re.compile(r"[A-Za-z]{4,}")
    lex_set = set(lexicon)

    for line in lines:
        corrected = line
        for match in token_pattern.finditer(line):
            token = match.group(0)
            low = token.lower()
            if low in lex_set:
                continue
            replacement, sim = _closest_lexicon_word(low, lexicon)
            if replacement and sim >= sim_threshold:
                corrected = re.sub(rf"\b{re.escape(token)}\b", replacement, corrected)
                corrections.append(f"{token}->{replacement}({sim:.2f})")
        corrected_lines.append(corrected)
    return corrected_lines, corrections


def _closest_lexicon_word(token: str, lexicon: List[str]) -> tuple[Optional[str], float]:
    best_word: Optional[str] = None
    best_sim = 0.0
    for word in lexicon:
        sim = SequenceMatcher(a=token, b=word).ratio()
        if sim > best_sim:
            best_sim = sim
            best_word = word
    return best_word, best_sim


def _load_lexicon() -> List[str]:
    words = set(_BUILTIN_LEXICON)
    labels = _labels_path()
    if labels.exists():
        try:
            for line in labels.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                expected = str(entry.get("expected_text") or "")
                for token in re.findall(r"[A-Za-z]{3,}", expected.lower()):
                    words.add(token)
        except Exception:
            pass
    return sorted(words)


def _labels_path() -> Path:
    raw = os.getenv("OCR_TRAINING_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve() / "labels.jsonl"
    return Path(__file__).resolve().parents[2] / "data" / "ocr_training" / "labels.jsonl"
