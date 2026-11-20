"""Run the independent GitHub evidence queue."""
import logging
from random import uniform
from threading import Event, Thread
from time import sleep
from datetime import timedelta

import httpx

from app.config import settings
from app.db import fetch, init_database
from app.integrations import github_jobs as jobs
from app.integrations.github_client import RateLimited, quota_log, warn_if_unauthenticated
from app.integrations.github_governor import ConfigBlocked
from app.screening.github_agent import GithubEvidence, collect, github_status, match_projects, now, score_profile
from app.screening.jd_parser import load_jd

log = logging.getLogger(__name__)


def work(job: dict) -> None:
    done = Event()

    def heartbeat():
        while not done.wait(20):
            try:
                if not jobs.renew(job):
                    return
            except Exception as exc:
                log.warning("GitHub lease heartbeat failed type=%s", type(exc).__name__)

    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        attempts = job["attempts"]
        try:
            rows = fetch("""SELECT id,file_path,clean_text,github_username,github_repo_links
                FROM resumes WHERE id=%s""", (job["candidate_id"],))
            if not rows:
                return
            resume = rows[0]
            jd = load_jd(str(job["run_id"]))
            evidence = collect(str(job["candidate_id"]), resume, jd)
            if isinstance(evidence, GithubEvidence):
                result = score_profile(evidence, match_projects(evidence, resume), jd)
            else:
                result = evidence
            if result.retry_after:
                status = "pending"
            else:
                status = {"ok": "done", "stale": "done", "not_provided": "done"}.get(result.status, result.status)
        except RateLimited as exc:
            result, status = github_status("rate_limited", limited=exc), "pending"
        except ConfigBlocked as exc:
            result, status = github_status("blocked_config", blocked=exc), "blocked_config"
        except Exception as exc:
            attempts += 1
            transient = isinstance(exc, httpx.TransportError) or (
                isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500)
            result = github_status("failed").model_copy(update={"error_category": "transient" if transient else type(exc).__name__})
            status = "failed"
            if transient and attempts < job["max_attempts"]:
                retry_at = now() + timedelta(seconds=2 ** min(attempts, 8) * 5 + uniform(0, 5))
                result = github_status("pending").model_copy(update={
                    "error_category": "transient", "retry_after": retry_at})
                status = "pending"
        jobs.finish(job, result, status=status, attempts=attempts)
    finally:
        done.set()
        thread.join()


def run_available() -> int:
    job = jobs.claim()
    count = 0
    if job:
        quota_log("start")
        try:
            while job:
                work(job)
                count += 1
                job = jobs.claim()
        finally:
            quota_log("end")
    return count


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    init_database()
    warn_if_unauthenticated()
    if settings.github_enabled:
        quota_log("startup")
    while True:
        try:
            if not run_available():
                sleep(2)
        except Exception as exc:
            log.error("GitHub worker batch failed type=%s", type(exc).__name__)
            sleep(5)


if __name__ == "__main__":
    main()
