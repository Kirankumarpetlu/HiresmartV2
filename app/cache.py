import hashlib
import json
from collections.abc import Callable

import numpy as np
from pgvector import Vector
from psycopg.types.json import Jsonb

from app.config import settings
from app.db import execute, fetch, get_pool


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def stable_digest(value) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))


def model_id() -> str:
    return f"{settings.llm_repo}:{settings.llm_file}"


def get_jd(text_hash: str, prompt_version: str):
    if not settings.cache_enabled:
        return None
    rows = fetch("""SELECT parsed,query_vectors FROM jd_parse_cache
        WHERE text_hash=%s AND model_id=%s AND prompt_version=%s""",
        (text_hash, model_id(), prompt_version))
    return rows[0] if rows else None


def put_jd(text_hash: str, prompt_version: str, parsed: dict, vectors: np.ndarray) -> None:
    if settings.cache_enabled:
        execute("""INSERT INTO jd_parse_cache (text_hash,model_id,prompt_version,parsed,query_vectors)
            VALUES (%s,%s,%s,%s,%s) ON CONFLICT (text_hash,model_id,prompt_version)
            DO UPDATE SET parsed=EXCLUDED.parsed,query_vectors=EXCLUDED.query_vectors,created_at=now()""",
            (text_hash, model_id(), prompt_version, Jsonb(parsed), Jsonb(vectors.tolist())))


def cached_embeddings(texts: list[str], encode: Callable[[list[str]], np.ndarray]) -> np.ndarray:
    if not texts or not settings.cache_enabled:
        return encode(texts)
    hashes = [digest(text) for text in texts]
    rows = fetch("""SELECT text_hash,embedding FROM embedding_cache
        WHERE model_id=%s AND text_hash=ANY(%s)""", (settings.embed_model, hashes))
    found = {row["text_hash"]: row["embedding"] for row in rows}
    missing_hashes = list(dict.fromkeys(value for value in hashes if value not in found))
    if missing_hashes:
        by_hash = {digest(text): text for text in texts}
        vectors = encode([by_hash[value] for value in missing_hashes])
        with get_pool().connection() as conn:
            with conn.cursor() as cursor:
                cursor.executemany("""INSERT INTO embedding_cache (text_hash,model_id,embedding) VALUES (%s,%s,%s)
                    ON CONFLICT (text_hash,model_id) DO NOTHING""",
                    [(value, settings.embed_model, vector) for value, vector in zip(missing_hashes, vectors)])
        for value, vector in zip(missing_hashes, vectors):
            found[value] = vector
    vectors = [found[value] for value in hashes]
    return np.asarray([vector.to_numpy() if isinstance(vector, Vector) else vector for vector in vectors], dtype=np.float32)


def evaluation_key(resume: dict, jd: dict, chunks: list[dict]) -> tuple:
    return (str(resume["id"]), digest(resume["clean_text"]), stable_digest({
        "must_have": jd["must_have"], "nice_to_have": jd["nice_to_have"],
        "min_years": jd.get("min_years"), "responsibilities": jd["responsibilities"],
    }), stable_digest([str(item["id"]) for item in chunks]))


def get_evaluation(key: tuple, prompt_version: str):
    if not settings.cache_enabled:
        return None
    rows = fetch("""SELECT verified FROM evaluation_cache WHERE resume_id=%s AND content_hash=%s
        AND requirements_hash=%s AND chunks_hash=%s AND model_id=%s AND prompt_version=%s""",
        key + (model_id(), prompt_version))
    return rows[0]["verified"] if rows else None


def put_evaluation(key: tuple, prompt_version: str, verified: dict) -> None:
    if settings.cache_enabled:
        execute("""INSERT INTO evaluation_cache
            (resume_id,content_hash,requirements_hash,chunks_hash,model_id,prompt_version,verified)
            VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT
            (resume_id,content_hash,requirements_hash,chunks_hash,model_id,prompt_version)
            DO UPDATE SET verified=EXCLUDED.verified,created_at=now()""",
            key + (model_id(), prompt_version, Jsonb(verified)))


def get_report(results_hash: str, prompt_version: str) -> str | None:
    if not settings.cache_enabled:
        return None
    rows = fetch("SELECT summary FROM report_cache WHERE results_hash=%s AND model_id=%s AND prompt_version=%s",
        (results_hash, model_id(), prompt_version))
    return rows[0]["summary"] if rows else None


def put_report(results_hash: str, prompt_version: str, summary: str) -> None:
    if settings.cache_enabled:
        execute("""INSERT INTO report_cache (results_hash,model_id,prompt_version,summary)
            VALUES (%s,%s,%s,%s) ON CONFLICT (results_hash,model_id,prompt_version)
            DO UPDATE SET summary=EXCLUDED.summary,created_at=now()""",
            (results_hash, model_id(), prompt_version, summary))


def clear() -> None:
    execute("TRUNCATE jd_parse_cache,embedding_cache,evaluation_cache,report_cache,github_profiles")
