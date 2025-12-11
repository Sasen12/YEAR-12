"""CLI script to import exams from local `Exams/` folder into the backend DB.
Usage: python scripts/import_exams.py [--year YEAR]
"""
import sys
import argparse
import pathlib
from typing import Optional
# Ensure `backend/` is on sys.path so `app` package imports work when running this script directly
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sqlmodel import Session
from app.database import engine
from app import services

def main(year: Optional[int] = None):
    """Scan the local `Exams/` folder and import found files.

    The optional `year` parameter restricts discovery to `Exams/<year>`.
    Results are printed to stdout for a quick CLI feedback loop.
    """
    exams_root = pathlib.Path(__file__).resolve().parents[2] / 'Exams'
    if not exams_root.exists():
        print(f'Exams folder not found at {exams_root}')
        return
    from app.utils.exam_loader import find_exam_files
    # use default skip keywords (exam reports, etc.) to avoid polluting questions
    files = find_exam_files(exams_root, year=year)
    if not files:
        print('No files found to import')
        return
    with Session(engine) as session:
        svc = services.ImportService(session)
        total_created = 0
        total_skipped = 0
        for f in files:
            try:
                b = f.read_bytes()
                inferred_year = year
                if inferred_year is None:
                    try:
                        inferred_year = int(f.parent.name)
                    except Exception:
                        inferred_year = None
                result = svc.import_file(b, f.name, subject='Software Development', default_exam_year=inferred_year)
                created = result.get('created', 0) if isinstance(result, dict) else 0
                skipped = result.get('skipped', 0) if isinstance(result, dict) else 0
                total_created += created
                total_skipped += skipped
                errors = len(result.get("errors", [])) if isinstance(result, dict) else "?"
                print(f'Imported {f}: created {created}, skipped {skipped}, errors {errors}')
            except Exception as e:
                print(f'Error importing {f}: {e}')
        print(f'Total created questions: {total_created}, skipped {total_skipped}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, help='Import only from this year folder')
    args = parser.parse_args()
    main(year=args.year)
