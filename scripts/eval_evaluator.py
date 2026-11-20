"""Compare persisted evaluator levels with reviewed per-skill labels."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.db import fetch
from app.ingest.metadata import canonical


def metrics(pairs: list[tuple[int, int]]) -> dict:
    return {"compared": len(pairs), "agreement_rate": sum(a == b for a, b in pairs) / len(pairs) if pairs else None,
        "mean_absolute_error": sum(abs(a - b) for a, b in pairs) / len(pairs) if pairs else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jd_id")
    parser.add_argument("labels", type=Path, nargs="?", default=settings.data_dir / "labeled" / "labels.json")
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    if not labels.get("human_reviewed") and not args.allow_synthetic:
        parser.error("Human review required; use --allow-synthetic only for a provisional smoke test")
    expected = {item["file"]: {canonical(key): value for key, value in item["expected_levels"].items()} for item in labels["resumes"]}
    rows = fetch("""SELECT r.file_name,s.skill_report FROM screening_results s JOIN resumes r ON r.id=s.resume_id
        WHERE s.jd_id=%s AND s.status='done'""", (args.jd_id,))
    pairs = []
    for row in rows:
        for skill in row["skill_report"] or []:
            target = expected.get(row["file_name"], {}).get(canonical(skill["skill"]))
            if target is not None:
                pairs.append((target, skill["level"]))
    print(json.dumps({"human_reviewed": labels.get("human_reviewed", False), **metrics(pairs)}, indent=2))
