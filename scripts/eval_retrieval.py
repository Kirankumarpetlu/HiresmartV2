"""Measure recall@10/25 using reviewed labels or an explicitly provisional smoke set."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.ingest.pipeline import process_upload
from app.ingest.chunk import chunk
from app.ingest.parse import parse
from app.ingest.preprocess import preprocess
from app.services.embed_service import encode
from app.db import get_pool
from app.models import ParsedJD
from app.screening.retrieval import retrieve


def recall(retrieved: list[str], relevant: list[str], k: int) -> float:
    return len(set(retrieved[:k]) & set(relevant)) / len(set(relevant)) if relevant else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path, nargs="?", default=settings.data_dir / "labeled" / "labels.json")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--rechunk", action="store_true", help="Rebuild only labeled resumes' derived chunks with the current configuration")
    args = parser.parse_args()
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    if not labels.get("human_reviewed") and not args.allow_synthetic:
        parser.error("Labels need human review; use --allow-synthetic only for a provisional smoke test")
    ids = {}
    for item in labels["resumes"]:
        file = (args.labels.parent / item["file"]).resolve()
        if not file.is_relative_to(args.labels.parent.resolve()):
            parser.error("Label file paths must stay within the labeled directory")
        ids[item["file"]] = process_upload(file.read_bytes(), file.name)["id"]
        if args.rechunk:
            text, _ = parse(file.read_bytes(), file.name)
            chunks = chunk(preprocess(text))
            vectors = encode([item["content"] for item in chunks])
            with get_pool().connection() as conn:
                conn.execute("UPDATE resumes SET embed_model=%s WHERE id=%s", (settings.embed_model, ids[item["file"]]))
                conn.execute("DELETE FROM resume_chunks WHERE resume_id=%s", (ids[item["file"]],))
                for item, vector in zip(chunks, vectors):
                    conn.execute("""INSERT INTO resume_chunks (resume_id,section,label,content,token_count,embedding)
                        VALUES (%s,%s,%s,%s,%s,%s)""", (ids[file.name], item["section"], item["label"], item["content"], item["token_count"], vector))
    reports = []
    for job in labels["jobs"]:
        results = retrieve(ParsedJD.model_validate(job["parsed"]), 25)
        retrieved = [str(row["id"]) for row in results]
        relevant = [ids[name] for name in job["relevant"]]
        reports.append({"job": job["name"], "recall@10": recall(retrieved, relevant, 10), "recall@25": recall(retrieved, relevant, 25)})
    result = {"human_reviewed": labels.get("human_reviewed", False), "chunk_target": settings.chunk_target,
        "query_prefix": settings.use_query_prefix, "aggregation": settings.agg_mode, "resumes": ids, "jobs": reports}
    output = args.labels.parent / f"retrieval_{settings.chunk_target}_{settings.use_query_prefix}_{settings.agg_mode}.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
