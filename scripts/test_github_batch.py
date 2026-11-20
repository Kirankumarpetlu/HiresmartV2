"""Run GitHub-only evidence collection on local resumes and a JD."""
import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cache import digest, get_jd
from app.config import settings
from app.db import execute, init_database
from app.ingest.parse import parse
from app.ingest.pipeline import process_upload
from app.ingest.preprocess import preprocess
from app.integrations.github_client import GithubClient, RateLimited, warn_if_unauthenticated
from app.integrations.github_governor import ConfigBlocked
from app.models import ParsedJD
from app.screening.github_agent import GithubEvidence, collect, github_status, match_projects, score_profile
from app.screening.jd_parser import PROMPT_VERSION, parse_jd
from app.screening.retrieval import retrieve
from app.screening.stage1_filter import should_filter


def quota() -> dict:
    try:
        with GithubClient("octocat") as client:
            return client.rate_limit()
    except Exception as exc:
        return {"error": type(exc).__name__}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resumes", type=Path)
    parser.add_argument("jd", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    warn_if_unauthenticated()
    init_database()
    paths = sorted(args.resumes.iterdir()) if args.resumes.is_dir() else [args.resumes]
    files = [file for file in paths if file.is_file() and file.suffix.lower() in (".pdf", ".docx")][:args.limit]
    if not files:
        parser.error("Provide a resume file or directory containing PDFs or DOCX files")
    ids = list(dict.fromkeys(process_upload(file.read_bytes(), file.name)["id"] for file in files))
    text = args.jd.read_text(encoding="utf-8-sig") if args.jd.suffix.lower() == ".txt" else parse(args.jd.read_bytes(), args.jd.name)[0]
    cached = get_jd(digest(preprocess(text)), PROMPT_VERSION)
    if cached:
        jd = ParsedJD.model_validate(cached["parsed"])
        vectors = np.asarray(cached["query_vectors"], dtype=np.float32)
    else:
        temporary_id, jd, vectors = parse_jd(text)
        execute("DELETE FROM job_descriptions WHERE id=%s", (temporary_id,))

    start = quota()
    print("Quota at start:", start, flush=True)
    rows = []
    for resume in retrieve(jd, resume_ids=ids, vectors=vectors, all_candidates=True):
        resume_id = str(resume["id"])
        if should_filter(jd, resume):
            rows.append({"resume_id": resume_id, "status": "filtered_out",
                         "score_explanation": "The resume did not pass the initial screening filter, so GitHub scoring was skipped."})
            continue
        try:
            evidence = collect(resume_id, resume, jd)
            result = score_profile(evidence, match_projects(evidence, resume), jd) if isinstance(evidence, GithubEvidence) else evidence
        except RateLimited as exc:
            result = github_status("rate_limited", limited=exc)
        except ConfigBlocked as exc:
            result = github_status("blocked_config", blocked=exc)
        except Exception as exc:
            result = github_status("failed").model_copy(update={"error_category": type(exc).__name__})
        rows.append({"resume_id": resume_id, **result.model_dump(mode="json", exclude={"project_matches", "red_flags", "strengths"})})
    end = quota()
    print("Quota at end:", end, flush=True)
    scored = [row for row in rows if row.get("github_score") is not None]
    report = {"indexed_resumes": len(ids), "retrieved": len(rows), "scored": len(scored),
              "authenticated": bool(settings.github_token.get_secret_value()),
              "quota_start": start, "quota_end": end, "status_counts": dict(Counter(row["status"] for row in rows)),
              "network_calls": sum(row.get("network_calls", 0) for row in rows),
              "cache_hits": sum(row.get("cache_hits", 0) for row in rows),
              "secondary_limits": sum(row.get("secondary_limits", 0) for row in rows),
              "secondary_limit_hits": sum(row.get("secondary_limits", 0) for row in rows),
              "primary_limit_hits": sum(row.get("primary_limit_hits", 0) for row in rows),
              "reserve_deferrals": sum(row.get("error_category") == "reserve" for row in rows),
              "config_blocks": sum(row["status"] == "blocked_config" for row in rows),
              "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Outcomes:", report["status_counts"], flush=True)
    print("Network calls:", report["network_calls"], "cache hits:", report["cache_hits"], flush=True)
    print("Quota counters:", {key: report[key] for key in
          ("reserve_deferrals", "config_blocks", "primary_limit_hits", "secondary_limit_hits")}, flush=True)


if __name__ == "__main__":
    main()
