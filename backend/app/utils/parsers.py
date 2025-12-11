"""File parsing utilities that convert supported file formats into a
normalized question list.

Supported input types: JSON, CSV, TXT, PDF and DOCX. Parsers return a
list of dictionaries with keys: `question_text`, `possible_answers`,
`difficulty` and `question_type`.
"""

import io
import json
import csv
from typing import List, Dict
import pdfplumber
import docx
from typing import Tuple


def parse_file_to_questions(file_bytes: bytes, filename: str) -> List[Dict]:
    """Dispatch to the appropriate parser based on file extension."""
    name = filename.lower()
    if name.endswith('.json'):
        return parse_json(file_bytes)
    if name.endswith('.csv'):
        return parse_csv(file_bytes)
    if name.endswith('.txt'):
        return parse_txt(file_bytes)
    if name.endswith('.pdf'):
        return parse_pdf(file_bytes)
    if name.endswith('.docx'):
        return parse_docx(file_bytes)
    raise ValueError('Unsupported file type')


def parse_json(b: bytes):
    """Parse a JSON array of question objects and normalize them."""
    data = json.loads(b.decode('utf-8'))
    out = []
    for item in data:
        out.append(normalize_question(item))
    return out


def parse_csv(b: bytes):
    """Parse a CSV where a single column contains pipe-separated answers.

    Expected columns: `question` or `question_text`, optional `answers`
    (pipe separated) and optional `correct` to flag the correct answer.
    Optional metadata columns are passed through if present:
    `explanation`, `exam_year`/`year`, `question_number`/`qnum`.
    """
    out = []
    sio = io.StringIO(b.decode('utf-8'))
    reader = csv.DictReader(sio)
    for row in reader:
        item = {
            'question_text': str(row.get('question') or row.get('question_text') or ''),
            'possible_answers': [],
            'explanation': row.get('explanation'),
            'exam_year': _coerce_int(row.get('exam_year') or row.get('year')),
            'question_number': _coerce_int(row.get('question_number') or row.get('qnum')),
            'difficulty': row.get('difficulty'),
            'question_type': row.get('question_type')
        }
        # Accept pipe-delimited answers so teachers can author simple CSVs quickly.
        answers_raw = row.get('answers') or row.get('possible_answers') or ''
        parts = [p for p in answers_raw.split('|') if p.strip()]
        correct = row.get('correct')
        for p in parts:
            is_correct = False
            if correct is not None:
                is_correct = p.strip() == str(correct).strip()
            item['possible_answers'].append({'answer_text': p.strip(), 'is_correct': is_correct})
        out.append(item)
    return out


def parse_txt(b: bytes):
    """Parse a simple plaintext format where questions are separated by
    blank lines and the first answer line is treated as correct.
    """
    s = b.decode('utf-8')
    sections = [sec.strip() for sec in s.split('\n\n') if sec.strip()]
    out = []
    for sec in sections:
        lines = sec.splitlines()
        q = lines[0]
        answers = []
        for i, l in enumerate(lines[1:]):
            if l.strip():
                text, is_correct = _parse_answer_line(l.strip())
                answers.append({'answer_text': text, 'is_correct': is_correct or (i == 0)})
        out.append({'question_text': q, 'possible_answers': answers, 'difficulty': None, 'question_type': 'multiple_choice'})
    return out


def parse_pdf(b: bytes):
    """Extract text from PDF pages, split into blocks and build
    questions. For simple exam pages the first answer is assumed
    correct if no explicit marker exists.
    """
    # Extract text per page and split into question blocks by blank lines
    text_parts = []
    with pdfplumber.open(io.BytesIO(b)) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text() or ''
            text_parts.append(extracted)
    text = '\n'.join(text_parts)
    # split into blocks separated by two or more newlines
    blocks = [blk.strip() for blk in text.split('\n\n') if blk.strip()]
    out = []
    for blk in blocks:
            if '|' in blk:
                parts = [x.strip() for x in blk.split('|') if x.strip()]
                q = parts[0]
                answers = []
                for i, x in enumerate(parts[1:]):
                    text, is_correct = _parse_answer_line(x)
                    answers.append({'answer_text': text, 'is_correct': is_correct or (i == 0)})
                out.append({'question_text': q, 'possible_answers': answers, 'difficulty': None, 'question_type': 'multiple_choice'})
            else:
                lines = [l.strip() for l in blk.splitlines() if l.strip()]
                if not lines:
                    continue
                q = lines[0]
                answers = []
                for i, l in enumerate(lines[1:]):
                    text, is_correct = _parse_answer_line(l)
                    answers.append({'answer_text': text, 'is_correct': is_correct or (i == 0)})
                out.append({'question_text': q, 'possible_answers': answers, 'difficulty': None, 'question_type': 'multiple_choice'})
    return out


def parse_docx(b: bytes):
    """Parse a DOCX document into question blocks.

    Paragraph groups separated by empty paragraphs are treated as a
    question block. If a block contains `|` it is parsed as
    `question|answer1|answer2...` otherwise the first line is the
    question and subsequent lines are answers.
    """
    out = []
    doc = docx.Document(io.BytesIO(b))
    # Collect contiguous paragraphs into blocks separated by empty paragraphs
    blocks = []
    current = []
    for p in doc.paragraphs:
        text = (p.text or '').strip()
        if not text:
            if current:
                blocks.append('\n'.join(current))
                current = []
            continue
        current.append(text)
    if current:
        blocks.append('\n'.join(current))

    for blk in blocks:
        # If block contains '|' treat as question|ans1|ans2
        if '|' in blk:
            parts = [x.strip() for x in blk.split('|') if x.strip()]
            q = parts[0]
            answers = []
            for i, x in enumerate(parts[1:]):
                text, is_correct = _parse_answer_line(x)
                answers.append({'answer_text': text, 'is_correct': is_correct or (i == 0)})
            out.append({'question_text': q, 'possible_answers': answers, 'difficulty': None, 'question_type': 'multiple_choice'})
        else:
            lines = [l.strip() for l in blk.splitlines() if l.strip()]
            if not lines:
                continue
            q = lines[0]
            answers = []
            for i, l in enumerate(lines[1:]):
                text, is_correct = _parse_answer_line(l)
                answers.append({'answer_text': text, 'is_correct': is_correct or (i == 0)})
            out.append({'question_text': q, 'possible_answers': answers, 'difficulty': None, 'question_type': 'multiple_choice'})
    return out


def normalize_question(item: dict) -> dict:
    """Normalize a parsed question object (maps alternative keys to the
    canonical output shape).
    """
    # Ensure keys exist
    return {
        'question_text': item.get('question_text') or item.get('question') or '',
        'possible_answers': item.get('possible_answers') or item.get('answers') or [],
        'explanation': item.get('explanation') or item.get('solution') or None,
        'exam_year': _coerce_int(item.get('exam_year') or item.get('year')),
        'question_number': _coerce_int(item.get('question_number') or item.get('qnum')),
        'difficulty': item.get('difficulty'),
        'question_type': item.get('question_type')
    }


def _parse_answer_line(text: str) -> Tuple[str, bool]:
    """Detect simple correctness markers in an answer line.

    Supports leading '*' or trailing markers like '(correct)'; falls back to False.
    """
    is_correct = False
    cleaned = text.strip()
    lower = cleaned.lower()
    # trailing markers
    for marker in ('(correct)', '[correct]', '{correct}'):
        if lower.endswith(marker):
            is_correct = True
            cleaned = cleaned[: -len(marker)].strip()
            break
    # leading marker like "* answer"
    if cleaned.startswith('*'):
        is_correct = True
        cleaned = cleaned.lstrip('*').strip()
    return cleaned, is_correct


def _coerce_int(val):
    try:
        return int(val) if val is not None and str(val).strip() != '' else None
    except Exception:
        return None
