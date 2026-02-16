"""Evaluate OCR quality against saved labeled training samples.

Usage:
    python scripts/eval_ocr.py
    python scripts/eval_ocr.py --limit 100 --split val
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils.ocr import ocr_file_to_lines  # noqa: E402
from app.utils.ocr_training import get_training_root  # noqa: E402


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _char_error_rate(gt: str, pred: str) -> float:
    gt_n = _norm(gt)
    pred_n = _norm(pred)
    if not gt_n:
        return 0.0 if not pred_n else 1.0
    return _levenshtein(gt_n, pred_n) / max(1, len(gt_n))


def _load_records(labels_file: Path) -> list[dict]:
    if not labels_file.exists():
        return []
    out = []
    for line in labels_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=str(get_training_root()))
    parser.add_argument("--limit", type=int, default=0, help="Max records to evaluate (0 = all)")
    parser.add_argument("--split", default="", choices=["", "train", "val", "test"], help="Optional split filter")
    parser.add_argument("--output", default="", help="Optional report JSON path")
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    labels_file = root / "labels.jsonl"
    records = _load_records(labels_file)
    if args.split:
        records = [r for r in records if r.get("split") == args.split]
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        print("No records found for evaluation.")
        return 1

    results = []
    for rec in records:
        image_path = Path(rec.get("image_path", ""))
        if not image_path.exists():
            continue
        gt = str(rec.get("expected_text", ""))
        started = time.perf_counter()
        try:
            lines, engine, warnings = ocr_file_to_lines(image_path.read_bytes(), image_path.name, return_meta=True)
            pred = "\n".join(lines)
            ok = True
            err = ""
        except Exception as exc:
            lines, engine, warnings = [], "none", []
            pred = ""
            ok = False
            err = str(exc)
        duration_ms = (time.perf_counter() - started) * 1000.0
        cer = _char_error_rate(gt, pred) if ok else 1.0
        exact = _norm(gt) == _norm(pred) and ok
        results.append(
            {
                "id": rec.get("id"),
                "split": rec.get("split", ""),
                "engine": engine,
                "expected_text": gt,
                "predicted_text": pred,
                "char_error_rate": cer,
                "exact_match": exact,
                "duration_ms": round(duration_ms, 2),
                "ok": ok,
                "error": err,
                "warnings": warnings,
            }
        )

    if not results:
        print("No valid records could be evaluated.")
        return 2

    cer_values = [r["char_error_rate"] for r in results]
    durations = [r["duration_ms"] for r in results]
    exact_count = sum(1 for r in results if r["exact_match"])
    ok_count = sum(1 for r in results if r["ok"])
    summary = {
        "dataset_root": str(root),
        "evaluated": len(results),
        "successful_runs": ok_count,
        "failed_runs": len(results) - ok_count,
        "exact_match_rate": round(exact_count / len(results), 4),
        "avg_cer": round(sum(cer_values) / len(cer_values), 4),
        "median_cer": round(statistics.median(cer_values), 4),
        "avg_duration_ms": round(sum(durations) / len(durations), 2),
        "median_duration_ms": round(statistics.median(durations), 2),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    print(json.dumps(summary, indent=2))

    if args.output:
        output_path = Path(args.output)
    else:
        reports_dir = root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = reports_dir / f"ocr_eval_{stamp}.json"
    output_path.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(f"Report written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
