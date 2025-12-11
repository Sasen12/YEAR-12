"""Answer key loader for multiple choice (Section A) mapping."""

import csv
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Optional


@lru_cache(maxsize=16)
def load_answer_key(exams_root: Path, year: int) -> Dict[int, List[str]]:
    """Load an answer key CSV/JSON from Exams/<year>/answer_key.* if present.

    Expected CSV columns: question_number, correct (pipe-separated if multiple).
    """
    year_dir = exams_root / str(year)
    if not year_dir.exists():
        return {}
    candidates = list(year_dir.glob('answer_key.csv'))
    if not candidates:
        return {}
    key_file = candidates[0]
    mapping: Dict[int, List[str]] = {}
    with key_file.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                qnum = int(row.get('question_number') or row.get('qnum') or '')
            except Exception:
                continue
            correct = row.get('correct') or ''
            parts = [p.strip() for p in correct.split('|') if p.strip()]
            if parts:
                mapping[qnum] = parts
    return mapping


def get_correct_answers_from_key(exams_root: Path, year: Optional[int], qnum: Optional[int]) -> Optional[List[str]]:
    if year is None or qnum is None:
        return None
    mapping = load_answer_key(exams_root, year)
    return mapping.get(qnum)
