import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import httpx
import pytest
from pydantic import SecretStr

from app.db import execute, fetch, get_pool
from app.integrations import github_governor as governor, github_jobs as jobs
from app.integrations.github_client import GithubClient, PrimaryRateLimited
from app.models import GithubResult
from app.workers import github_worker

pytestmark = pytest.mark.skipif(os.environ.get("RUN_POSTGRES_TESTS") != "1", reason="requires PostgreSQL")


@pytest.fixture
def seed():
    ids = []

    def create(count, shared=False):
        jd = uuid4()
        prefix = "test-" + jd.hex[:12]
        with get_pool().connection() as conn:
            conn.execute("INSERT INTO job_descriptions (id,title,raw_text) VALUES (%s,'Developer','Synthetic JD')", (jd,))
            conn.execute("""INSERT INTO resumes (file_name,file_path,clean_text,embed_model,github_username)
                SELECT 'Example',%s || n::text,'Python','test',CASE WHEN %s THEN %s ELSE %s || n::text END
                FROM generate_series(1,%s) n""", (prefix, shared, prefix, prefix, count))
            conn.execute("""INSERT INTO screening_results (jd_id,resume_id,status,skill_report,fit_scores,final_score)
                SELECT %s,id,'done','[]','{"experience":80,"domain":80,"education":80,"notes":""}',40
                FROM resumes WHERE file_path LIKE %s""", (jd, prefix + "%"))
        ids.append((jd, prefix))
        return str(jd)

    yield create
    for jd, prefix in ids:
        execute("DELETE FROM job_descriptions WHERE id=%s", (jd,))
        execute("DELETE FROM resumes WHERE file_path LIKE %s", (prefix + "%",))


def test_username_claim_is_exclusive_and_expired_lease_cannot_publish(seed):
    jd = seed(2, shared=True)
    assert jobs.enqueue(jd, 2) == 2
    with ThreadPoolExecutor(2) as pool:
        claims = list(pool.map(lambda _: jobs.claim(), range(2)))
    first, = [claim for claim in claims if claim]
    execute("UPDATE github_jobs SET lease_until=now()-interval '1 second' WHERE id=%s", (first["id"],))
    replay = jobs.claim()
    assert replay["id"] == first["id"] and replay["lease_id"] != first["lease_id"]
    result = GithubResult(status="ok", github_score=90, evidence_coverage="full")
    assert not jobs.finish(first, result, status="done", attempts=0)
    assert jobs.finish(replay, result, status="done", attempts=0)
    assert jobs.claim()["candidate_id"] != first["candidate_id"]


def test_worker_quota_wait_preserves_attempts_and_transient_failure_backs_off(seed, monkeypatch):
    jd = seed(1)
    assert jobs.enqueue(jd, 1) == 1
    first = jobs.claim()
    due = datetime.now(timezone.utc) + timedelta(minutes=5)

    def limited(*args):
        raise PrimaryRateLimited("core", due)

    monkeypatch.setattr(github_worker, "collect", limited)
    github_worker.work(first)
    row = fetch("SELECT status,attempts,retry_after FROM github_jobs WHERE id=%s", (first["id"],))[0]
    assert row == {"status": "pending", "attempts": 0, "retry_after": due}
    assert jobs.claim() is None
    assert jobs.retry(jd, [first["candidate_id"]]) == 1

    def transient(*args):
        raise httpx.ReadTimeout("synthetic")

    monkeypatch.setattr(github_worker, "collect", transient)
    github_worker.work(jobs.claim())
    row = fetch("SELECT status,attempts,retry_after FROM github_jobs WHERE id=%s", (first["id"],))[0]
    assert row["status"] == "pending" and row["attempts"] == 1 and row["retry_after"]


@pytest.mark.parametrize(("token", "category", "summary"), [
    ("", "no_token", "GitHub token not configured; evidence not collected."),
    ("test-token-value", "reserve_exceeds_limit", "GitHub quota too small for the configured reserve; evidence not collected."),
])
def test_impossible_quota_is_terminal_and_preserves_resume_score(seed, monkeypatch, token, category, summary):
    jd = seed(1)
    monkeypatch.setattr(governor.settings, "github_token", SecretStr(token))
    monkeypatch.setattr(governor.settings, "github_reserve_remaining", 200)
    reset = datetime.now(timezone.utc) + timedelta(hours=1)
    governor.record("test-config", 60, 57, int(reset.timestamp()))
    try:
        with get_pool().connection() as conn:
            baseline = jobs.rerank_transaction(conn, jd)[0]["final_score"]
        assert jobs.enqueue(jd, 1) == 1
        job = jobs.claim()
        monkeypatch.setattr(github_worker, "collect", lambda *args: governor.acquire("test-config", 55, admission=True))
        github_worker.work(job)
        row = fetch("SELECT status,attempts,retry_after,last_error_category FROM github_jobs WHERE id=%s", (job["id"],))[0]
        assert row == {"status": "blocked_config", "attempts": 0, "retry_after": None, "last_error_category": category}
        assert jobs.claim() is None and not jobs.awaiting(jd)
        result = fetch("SELECT final_score,github_score,github_json FROM screening_results WHERE jd_id=%s", (jd,))[0]
        assert result["final_score"] == baseline and result["github_score"] is None
        assert result["github_json"]["status"] == "blocked_config" and result["github_json"]["summary"] == summary
        assert jobs.progress(jd)["config_blocks"] == 1
        if not token:
            assert jobs.retry(jd, [job["candidate_id"]]) == 0
            monkeypatch.setattr(governor.settings, "github_token", SecretStr("test-token-value"))
        assert jobs.retry(jd, [job["candidate_id"]]) == 1
    finally:
        execute("DELETE FROM github_quota_state WHERE bucket='test-config'")


def test_thousand_jobs_are_scoped_paced_and_drained_with_mock_http(seed, monkeypatch):
    jd = seed(1000)
    assert jobs.enqueue(jd, 100) == 100
    assert fetch("SELECT count(*) n FROM github_jobs WHERE run_id=%s", (jd,))[0]["n"] == 100
    assert jobs.enqueue(jd, 900) == 900
    clock = datetime.now(timezone.utc).replace(microsecond=0)
    reset = clock + timedelta(hours=1)
    used = 0
    monkeypatch.setattr(governor, "now", lambda: clock)
    monkeypatch.setattr(governor.settings, "github_reserve_remaining", 200)
    governor.record("test-load", 1200, 1120, int(reset.timestamp()))
    try:
        def response(request):
            nonlocal used
            used += 1
            assert 1120 - used >= 120
            return httpx.Response(200, json={"public_repos": 0})

        transport = httpx.MockTransport(response)
        with GithubClient("synthetic", transport) as github:
            for _ in range(1000):
                job = jobs.claim()
                assert str(job["run_id"]) == jd
                if delay := governor.acquire("test-load"):
                    clock += timedelta(seconds=delay + .000001)
                    assert governor.acquire("test-load") == 0
                github._request("/users/synthetic", {}, None, dict, (404,), "test-load")
                assert jobs.finish(job, GithubResult(status="not_found"), status="not_found", attempts=0)
        assert used == 1000 and not jobs.awaiting(jd)
        with pytest.raises(governor.QuotaWait):
            governor.acquire("test-load")
    finally:
        execute("DELETE FROM github_quota_state WHERE bucket='test-load'")
