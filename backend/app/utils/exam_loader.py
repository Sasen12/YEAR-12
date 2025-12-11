"""Helpers to discover exam files under a local `Exams/` folder.

The loader scans supported extensions recursively and returns a list
of `Path` objects suitable for feeding to the import service.
"""

from pathlib import Path
from typing import List, Iterable, Optional
from .parsers import parse_file_to_questions

SUPPORTED_EXT = {'.csv', '.txt', '.json', '.pdf', '.docx'}
DEFAULT_SKIP_KEYWORDS = ('examrep', 'report', 'study design', 'studydesign')


def _should_skip(file_path: Path, skip_keywords: Iterable[str]) -> bool:
    """Return True if filename contains any skip keyword (case-insensitive)."""
    name = file_path.name.lower()
    for kw in skip_keywords:
        if kw and kw.lower() in name:
            return True
    return False


def find_exam_files(root: Path, year: int = None, skip_keywords: Optional[Iterable[str]] = None) -> List[Path]:
    """Return file paths for supported exam files.

    If `year` is provided the function will only search `root/<year>`;
    otherwise it scans all subdirectories at the top-level of `root`.
    Files whose names contain any of `skip_keywords` are ignored to avoid
    pulling examiner reports (e.g., names containing "examrep" or "report").
    """
    skip_keywords = list(skip_keywords) if skip_keywords is not None else list(DEFAULT_SKIP_KEYWORDS)
    files = []
    if year:
        year_dir = root / str(year)
        if not year_dir.exists():
            return []
        search_paths = [year_dir]
    else:
        # scan root recursively (covers all years) and pick supported files
        search_paths = [root]

    seen = set()
    for p in search_paths:
        for f in p.rglob('*'):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXT:
                # Avoid importing examiner reports by name; also dedupe identical Path objects.
                if _should_skip(f, skip_keywords):
                    continue
                if f not in seen:
                    files.append(f)
                    seen.add(f)
    # sort for deterministic order
    return sorted(files)
