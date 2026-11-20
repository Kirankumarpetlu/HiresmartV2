"""Verify cosine retrieval and the HNSW access path on the actual database."""
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.db import fetch, get_pool
from app.screening.retrieval import SEARCH_SQL, search_chunks

if __name__ == "__main__":
    vector = fetch("SELECT embedding FROM resume_chunks LIMIT 1")[0]["embedding"]
    params = {"q": vector, "min_years": 0, "embed_model": settings.embed_model, "chunk_limit": settings.retrieval_chunk_limit, "top_chunks": 3, "n": 10, "resume_ids": None}
    rows = search_chunks(params)
    assert rows and all(-1.001 <= row["similarity"] <= 1.001 for row in rows)
    with get_pool().connection() as conn:
        # Tiny calibration tables favor a sequential scan; demonstrate the available index path.
        conn.execute("SET LOCAL enable_seqscan=off")
        conn.execute("SET LOCAL hnsw.ef_search = 100")
        plan = conn.execute("EXPLAIN (ANALYZE, COSTS OFF) " + SEARCH_SQL, params).fetchall()
    formatted = re.sub(r"'\[[^\]]+\]'::vector", "'[query vector]'::vector", "\n".join(row["QUERY PLAN"] for row in plan))
    assert "idx_chunks_embedding" in formatted, formatted
    print(formatted)
    print("Cosine retrieval and HNSW acceptance passed.")
