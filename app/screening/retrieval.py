import numpy as np
from concurrent.futures import ThreadPoolExecutor

from app.config import settings
from app.db import fetch, get_pool
from app.models import ParsedJD
from app.screening.jd_parser import jd_vectors

SEARCH_SQL = """
    WITH nearest AS MATERIALIZED (
        SELECT c.resume_id, c.embedding <=> %(q)s AS distance
        FROM resume_chunks c
        ORDER BY c.embedding <=> %(q)s
        LIMIT %(chunk_limit)s
    ), numbered AS (
        SELECT nearest.*, row_number() OVER (PARTITION BY resume_id ORDER BY distance) AS position
        FROM nearest JOIN resumes r ON r.id=nearest.resume_id
        WHERE (r.years_exp IS NULL OR r.years_exp >= %(min_years)s)
          AND r.embed_model=%(embed_model)s
          AND (%(resume_ids)s::uuid[] IS NULL OR r.id=ANY(%(resume_ids)s::uuid[]))
    )
    SELECT resume_id, AVG(1-distance) AS similarity FROM numbered
    WHERE position <= %(top_chunks)s GROUP BY resume_id
    ORDER BY similarity DESC LIMIT %(n)s
"""

ALL_CANDIDATES_SQL = """
    WITH evidence AS (
        SELECT c.resume_id,c.embedding <=> %(q)s AS distance,
            row_number() OVER (PARTITION BY c.resume_id ORDER BY c.embedding <=> %(q)s) AS position
        FROM resume_chunks c JOIN resumes r ON r.id=c.resume_id
        WHERE (r.years_exp IS NULL OR r.years_exp >= %(min_years)s)
          AND r.embed_model=%(embed_model)s
          AND (%(resume_ids)s::uuid[] IS NULL OR r.id=ANY(%(resume_ids)s::uuid[]))
    )
    SELECT resume_id,AVG(1-distance) AS similarity FROM evidence
    WHERE position<=%(top_chunks)s GROUP BY resume_id
"""


def search_chunks(params: dict) -> list[dict]:
    with get_pool().connection() as conn:
        conn.execute("SET LOCAL hnsw.ef_search = 100")
        return conn.execute(SEARCH_SQL, params).fetchall()


def retrieve(jd: ParsedJD, n: int | None = None, vectors: np.ndarray | None = None,
             resume_ids: list[str] | None = None, *, all_candidates: bool = False) -> list[dict]:
    vectors = jd_vectors(jd) if vectors is None else vectors
    scores = {}
    n = n or settings.shortlist_size
    def search(vector):
        # shortcut: shortlist scans a bounded nearest-chunk pool; increase the limit for dense or heavily filtered corpora.
        params = {"q": vector, "min_years": jd.min_years or 0, "embed_model": settings.embed_model,
            "chunk_limit": max(settings.retrieval_chunk_limit, n * 3), "top_chunks": 1 if settings.agg_mode == "best" else 3, "n": n, "resume_ids": resume_ids}
        return fetch(ALL_CANDIDATES_SQL, params) if all_candidates else search_chunks(params)
    with ThreadPoolExecutor(max_workers=min(4, len(vectors))) as executor:
        searches = executor.map(search, vectors)
    for rows in searches:
        for row in rows:
            key = str(row["resume_id"])
            scores[key] = max(scores.get(key, -1), float(row["similarity"]))
    ids = sorted(scores, key=lambda value: (-scores[value], value))
    if not all_candidates:
        ids = ids[:n]
    if not ids:
        return []
    resumes = {str(row["id"]): row for row in fetch("""SELECT id,file_path,clean_text,skills,years_exp,
        github_username,github_repo_links FROM resumes WHERE id=ANY(%s::uuid[])""", (ids,))}
    return [dict(resumes[key], similarity=scores[key]) for key in ids if key in resumes]


def best_chunks(resume_id: str, vectors: np.ndarray, limit: int = 5) -> list[dict]:
    found = {}
    for vector in vectors:
        for row in fetch("""SELECT id,section,label,content,token_count,1-(embedding <=> %s) AS similarity
            FROM resume_chunks WHERE resume_id=%s ORDER BY embedding <=> %s LIMIT %s""", (vector, resume_id, vector, limit)):
            key = str(row["id"])
            if key not in found or row["similarity"] > found[key]["similarity"]:
                found[key] = row
    return sorted(found.values(), key=lambda row: row["similarity"], reverse=True)[:limit]


def best_chunks_for_resumes(resume_ids: list[str], vectors: np.ndarray, limit: int = 5) -> dict[str, list[dict]]:
    found = {str(resume_id): {} for resume_id in resume_ids}
    for vector in vectors:
        rows = fetch("""SELECT id,resume_id,section,label,content,token_count,similarity FROM (
            SELECT c.*,1-(embedding <=> %s) AS similarity,
                row_number() OVER (PARTITION BY resume_id ORDER BY embedding <=> %s) AS position
            FROM resume_chunks c WHERE resume_id=ANY(%s::uuid[])
        ) ranked WHERE position<=%s""", (vector, vector, resume_ids, limit))
        for row in rows:
            resume_id, chunk_id = str(row["resume_id"]), str(row["id"])
            if chunk_id not in found[resume_id] or row["similarity"] > found[resume_id][chunk_id]["similarity"]:
                found[resume_id][chunk_id] = row
    return {resume_id: sorted(chunks.values(), key=lambda row: row["similarity"], reverse=True)[:limit]
        for resume_id, chunks in found.items()}
