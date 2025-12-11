"""Utilities to extract per-question explanations from examiner reports.

The report PDFs often have headings like "Question 1", "Question 2", etc.
We split on those headings to build a mapping: question_number -> snippet.
"""

from pathlib import Path
from functools import lru_cache
import re
import pdfplumber
from typing import Optional, Dict


@lru_cache(maxsize=16)
def _load_report_sections(report_path: Path) -> Dict[int, str]:
    """Return a mapping of question number to text snippet for a report PDF."""
    text_parts = []
    with pdfplumber.open(report_path) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or '')
    full = "\n".join(text_parts)
    # Normalize spacing
    full = re.sub(r'\r', '\n', full)
    # Split on "Question <number>" headings
    matches = list(re.finditer(r'Question\\s+(\\d+)', full, flags=re.IGNORECASE))
    sections = {}
    for i, m in enumerate(matches):
        qnum = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full)
        snippet = full[start:end].strip()
        if snippet:
            sections[qnum] = snippet
    return sections


def find_report_explanation(exams_root: Path, year: int, question_number: int) -> Optional[str]:
    """Find an explanation snippet for a given year/question_number from report PDFs.

    This looks for a PDF in Exams/<year>/ with keywords 'examrep' or 'report'.
    Returns the text snippet if found, else None.
    """
    year_dir = exams_root / str(year)
    if not year_dir.exists():
        return None
    candidates = [p for p in year_dir.glob('*.pdf') if 'examrep' in p.name.lower() or 'report' in p.name.lower()]
    if not candidates:
        return None
    report = sorted(candidates)[0]
    sections = _load_report_sections(report)
    return sections.get(question_number)
