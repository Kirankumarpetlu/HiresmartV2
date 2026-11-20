from collections.abc import Sequence
from uuid import UUID

from psycopg.errors import UniqueViolation
from psycopg.types.json import Jsonb

from app.config import settings
from app.db import fetch, get_pool
from app.models import GithubResult, Weights
from app.integrations.github_governor import effective_reserve, quota_state
from app.ingest.pipeline import stored_github_metadata
from app.screening.github_agent import github_status, now
from app.screening.ranking import rank_candidates, rerank_transaction

LEASE = "2 minutes"


def enqueue(jd_id: str, count: int, *, ranked: Sequence[dict] | None = None) -> int:
    if not settings.github_enabled:
        return 0
    with get_pool().connection() as conn:
        job = conn.execute("SELECT ranking_weights FROM job_descriptions WHERE id=%s FOR UPDATE", (jd_id,)).fetchone()
        if not job:
            return 0
        rows = conn.execute("""SELECT s.*,r.years_exp,r.github_username,r.github_repo_links FROM screening_results s
            JOIN resumes r ON r.id=s.resume_id WHERE s.jd_id=%s AND s.status='done' AND s.github_json IS NULL""",
            (jd_id,)).fetchall()
        weights = Weights.model_validate(job["ranking_weights"] or {}).model_copy(update={"github": 0})
        if ranked is None:
            ordered = rank_candidates(rows, weights)
        else:
            by_id = {row["resume_id"]: row for row in rows}
            ordered = [by_id[row["resume_id"]] | {"final_score": row["final_score"]}
                       for row in ranked if row["resume_id"] in by_id]
        added = 0
        for row in ordered[:count]:
            username = row["github_username"]
            if row["github_repo_links"] is None and not username:
                resume = conn.execute("SELECT file_path,clean_text FROM resumes WHERE id=%s",
                                      (row["resume_id"],)).fetchone()
                if resume:
                    metadata = stored_github_metadata(resume)
                    username = metadata["github_username"]
                    conn.execute("""UPDATE resumes SET github_username=%s,github_repo_links=%s
                        WHERE id=%s AND github_repo_links IS NULL""",
                        (username, metadata["github_repo_links"], row["resume_id"]))
            if username:
                result = github_status("pending")
                conn.execute("""INSERT INTO github_jobs (candidate_id,username,run_id,priority)
                    VALUES (%s,lower(%s),%s,%s) ON CONFLICT (run_id,candidate_id) DO NOTHING""",
                    (row["resume_id"], username, jd_id, row["final_score"]))
                added += 1
            else:
                result = github_status("not_provided")
            conn.execute("UPDATE screening_results SET github_json=%s WHERE id=%s",
                         (Jsonb(result.model_dump(mode="json")), row["id"]))
        return added


def claim() -> dict | None:
    try:
        rows = fetch("""WITH picked AS (
                SELECT j.id FROM github_jobs j
                WHERE ((j.status='pending' AND (j.retry_after IS NULL OR j.retry_after <= now()))
                       OR (j.status='running' AND j.lease_until <= now()))
                AND NOT EXISTS (SELECT 1 FROM github_jobs active WHERE active.username=j.username
                    AND active.status='running' AND active.id<>j.id)
                ORDER BY j.priority DESC,j.created_at,j.id FOR UPDATE OF j SKIP LOCKED LIMIT 1
            )
            UPDATE github_jobs j SET status='running',lease_id=gen_random_uuid(),
                lease_until=now()+%s::interval,updated_at=now()
            FROM picked WHERE j.id=picked.id RETURNING j.*""", (LEASE,))
    except UniqueViolation:
        return None
    return rows[0] if rows else None


def renew(job: dict) -> bool:
    return bool(fetch("""UPDATE github_jobs SET lease_until=now()+%s::interval
        WHERE id=%s AND lease_id=%s AND status='running' RETURNING id""", (LEASE, job["id"], job["lease_id"])))


def finish(job: dict, result: GithubResult, *, status: str, attempts: int) -> bool:
    with get_pool().connection() as conn:
        locked = conn.execute("SELECT id FROM job_descriptions WHERE id=%s FOR UPDATE", (job["run_id"],)).fetchone()
        if not locked:
            return False
        current = conn.execute("""SELECT github_json,github_score FROM screening_results
            WHERE jd_id=%s AND resume_id=%s AND status='done'""", (job["run_id"], job["candidate_id"])).fetchone()
        metrics = (result.network_calls, result.cache_hits, result.secondary_limits, result.primary_limit_hits,
                   int(result.error_category == "reserve"), int(result.status == "blocked_config"))
        if current and current["github_score"] is not None and result.github_score is None and status != "blocked_config":
            previous = GithubResult.model_validate(current["github_json"])
            if previous.evidence_coverage in ("partial", "full"):
                result = previous.model_copy(update={"retry_after": result.retry_after, "error_category": result.error_category})
        row = conn.execute("""UPDATE github_jobs
            SET status=%s,attempts=%s,retry_after=%s,last_error_category=%s,evidence_coverage=%s,
                network_calls=network_calls+%s,cache_hits=cache_hits+%s,secondary_limits=secondary_limits+%s,
                primary_limit_hits=primary_limit_hits+%s,reserve_deferrals=reserve_deferrals+%s,config_blocks=config_blocks+%s,
                lease_until=NULL,lease_id=NULL,updated_at=now()
            WHERE id=%s AND lease_id=%s AND status='running' RETURNING id""",
            (status, attempts, result.retry_after, result.error_category, result.evidence_coverage,
             *metrics,
             job["id"], job["lease_id"])).fetchone()
        if not row:
            return False
        changed = conn.execute("""UPDATE screening_results SET github_json=%s,github_score=%s
            WHERE jd_id=%s AND resume_id=%s AND status='done' RETURNING id""",
            (Jsonb(result.model_dump(mode="json")), result.github_score, job["run_id"], job["candidate_id"])).fetchone()
        if changed and (result.github_score is not None or job["evidence_coverage"] != "none"):
            rerank_transaction(conn, job["run_id"])
        return True


def retry(jd_id: str, resume_ids: list[UUID]) -> int:
    with get_pool().connection() as conn:
        conn.execute("SELECT id FROM job_descriptions WHERE id=%s FOR UPDATE", (jd_id,))
        jobs = conn.execute("""UPDATE github_jobs SET status='pending',attempts=0,retry_after=NULL,updated_at=now()
            WHERE run_id=%s AND candidate_id=ANY(%s) AND (status IN ('failed','rate_limited') OR
                (status='pending' AND retry_after IS NOT NULL) OR
                (status='blocked_config' AND (last_error_category='reserve_exceeds_limit' OR %s)))
            RETURNING candidate_id""", (jd_id, resume_ids, bool(settings.github_token.get_secret_value()))).fetchall()
        for row in jobs:
            conn.execute("""UPDATE screening_results SET github_json=%s,github_score=NULL
                WHERE jd_id=%s AND resume_id=%s AND github_score IS NULL""",
                (Jsonb(github_status("pending").model_dump(mode="json")), jd_id, row["candidate_id"]))
        return len(jobs)


def awaiting(jd_id: str) -> bool:
    return bool(fetch("SELECT 1 FROM github_jobs WHERE run_id=%s AND status IN ('pending','running') LIMIT 1", (jd_id,)))


def ready_screenings() -> list[dict]:
    return fetch("""SELECT id FROM job_descriptions j WHERE github_waiting
        AND NOT EXISTS (SELECT 1 FROM github_jobs WHERE run_id=j.id AND status IN ('pending','running'))""")


def progress(jd_id: str) -> dict:
    counts = {row["status"]: row["n"] for row in fetch("""SELECT COALESCE(github_json->>'status','not_queued') AS status,
        count(*) AS n FROM screening_results WHERE jd_id=%s AND status='done' GROUP BY 1""", (jd_id,))}
    queue = fetch("""SELECT count(*) FILTER (WHERE status IN ('pending','running')) AS depth,
        min(retry_after) FILTER (WHERE status IN ('pending','running')) AS retry,
        COALESCE(sum(network_calls),0) AS network_calls,COALESCE(sum(cache_hits),0) AS cache_hits,
        COALESCE(sum(secondary_limits),0) AS secondary_limit_hits,
        COALESCE(sum(primary_limit_hits),0) AS primary_limit_hits,
        COALESCE(sum(reserve_deferrals),0) AS reserve_deferrals,COALESCE(sum(config_blocks),0) AS config_blocks
        FROM github_jobs WHERE run_id=%s""", (jd_id,))[0]
    state = quota_state()
    core = state.get("core", {})
    remaining = core.get("remaining")
    reset = core.get("reset_at")
    batch_waits = [date for date in (queue["retry"], core.get("blocked_until")) if date]
    wait = max(0, (max(batch_waits) - now()).total_seconds()) if batch_waits else 0
    spacing = max(0.25, (reset - now()).total_seconds() / max(1, remaining - effective_reserve(core.get("limit")))) \
        if reset and remaining is not None else 1
    total = sum(value for key, value in counts.items() if key not in ("not_queued", "not_provided"))
    scored = fetch("""SELECT count(*) AS scored FROM screening_results WHERE jd_id=%s AND status='done'
        AND github_score IS NOT NULL AND github_json->>'status' IN ('ok','partial','stale')""", (jd_id,))[0]["scored"]
    return {"counts": counts, "queue_depth": queue["depth"], "total": total, "scored": scored,
            "mixed_evidence": bool(queue["depth"]), "estimated_seconds": int(wait + queue["depth"] * 20 * min(60, spacing)),
            "quota": {key: {field: value for field, value in row.items()
                            if field in ("limit", "remaining", "reset_at", "blocked_until")} for key, row in state.items()},
            "network_calls": queue["network_calls"], "cache_hits": queue["cache_hits"],
            "secondary_limits": queue["secondary_limit_hits"], "secondary_limit_hits": queue["secondary_limit_hits"],
            "primary_limit_hits": queue["primary_limit_hits"], "reserve_deferrals": queue["reserve_deferrals"],
            "config_blocks": queue["config_blocks"]}
