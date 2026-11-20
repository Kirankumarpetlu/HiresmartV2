import json
import logging
import asyncio
from contextlib import asynccontextmanager, suppress
from logging.handlers import RotatingFileHandler
from uuid import UUID

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.config import settings
from app.cache import digest
from app.db import fetch, get_pool, init_database
from app.ingest.metadata import normalize_email
from app.ingest.parse import parse
from app.ingest.pipeline import process_upload
from app.models import (GithubExtendRequest, GithubRetryRequest, NotificationApproval, NotificationCandidates,
                        ResumeContact, RunStatus, ScreeningRequest, ScreeningResult, Weights)
from app.notify import notifications
from app.integrations import github_jobs
from app.integrations.github_client import warn_if_unauthenticated
from app.screening.graph import (can_review, create_screening, read_run, recover_runs, schedule,
                                 schedule_delivery, schedule_ready_github_runs, setup_checkpointer, status, write_run)
from app.screening.ranking import rerank
from app.screening.report import report_sections
from app.services import embed_service, llm_service
from app.timing import timed_stage


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(settings.data_dir / "service.log", maxBytes=5_000_000, backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger("app").setLevel(logging.INFO)
    logging.getLogger("app").addHandler(handler)
    try:
        init_database()
        setup_checkpointer()
        warn_if_unauthenticated()
        await run_in_threadpool(embed_service.get_embedder)
        await run_in_threadpool(llm_service.get_llm)
        recover_runs()
        watcher = asyncio.create_task(watch_github())
        try:
            yield
        finally:
            watcher.cancel()
            with suppress(asyncio.CancelledError):
                await watcher
    finally:
        get_pool().close()
        logging.getLogger("app").removeHandler(handler)
        handler.close()


async def watch_github() -> None:
    while True:
        await asyncio.sleep(2)
        try:
            await run_in_threadpool(schedule_ready_github_runs)
        except Exception as exc:
            logging.getLogger(__name__).warning("GitHub readiness check failed type=%s", type(exc).__name__)


app = FastAPI(title="Resume Screening Analyzer", lifespan=lifespan)


def ensure_jd(jd_id: UUID) -> str:
    if not fetch("SELECT id FROM job_descriptions WHERE id=%s", (jd_id,)):
        raise HTTPException(404, "Screening not found")
    return str(jd_id)


@app.post("/resumes")
async def upload_resumes(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            content = await file.read(settings.max_upload_bytes + 1)
            result = await run_in_threadpool(process_upload, content, file.filename or "upload")
            results.append(dict(result, file_name=file.filename))
        except ValueError as exc:
            results.append({"file_name": file.filename, "status": "error", "error": str(exc)})
        except Exception as exc:
            logging.getLogger(__name__).error("Upload failed type=%s", type(exc).__name__)
            results.append({"file_name": file.filename, "status": "error", "error": f"Upload failed ({type(exc).__name__})"})
        finally:
            await file.close()
    return results


@app.get("/resumes")
def list_resumes():
    rows = fetch("""SELECT id,file_name,parse_method,skills,years_exp,education,uploaded_at,github_username,github_repo_links,contact_email
        FROM resumes ORDER BY uploaded_at DESC""")
    return [dict(row, contact_email=notifications.mask_email(row["contact_email"]),
                 contact_status="available" if row["contact_email"] else "no_contact") for row in rows]


@app.patch("/resumes/{resume_id}/contact")
def set_contact(resume_id: UUID, contact: ResumeContact):
    address = normalize_email(contact.email)
    if not address:
        raise HTTPException(422, "Provide one valid email address")
    if not notifications.update_contact(resume_id, address):
        raise HTTPException(404, "Resume not found")
    return {"contact_email": notifications.mask_email(address), "contact_status": "available"}


@app.delete("/resumes/{resume_id}")
def delete_resume(resume_id: UUID):
    from pathlib import Path

    with get_pool().connection() as conn:
        row = conn.execute("SELECT file_path FROM resumes WHERE id=%s FOR UPDATE", (resume_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Resume not found")
        path = Path(row["file_path"]).resolve()
        upload_root = (settings.data_dir / "uploads").resolve()
        if not path.is_relative_to(upload_root):
            raise HTTPException(409, "Stored file path lies outside the upload directory")
        chunk_hashes = [digest(item["content"]) for item in conn.execute(
            "SELECT content FROM resume_chunks WHERE resume_id=%s", (resume_id,)).fetchall()]
        if chunk_hashes:
            conn.execute("DELETE FROM embedding_cache WHERE text_hash=ANY(%s)", (chunk_hashes,))
        conn.execute("DELETE FROM resumes WHERE id=%s", (resume_id,))
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return {"status": "deleted", "warning": "Database record removed; raw file needs manual cleanup"}
    return {"status": "deleted"}


@app.post("/screenings", status_code=202)
async def start_screening(request: Request):
    from pydantic import ValidationError

    try:
        if "multipart/form-data" in request.headers.get("content-type", ""):
            form = await request.form()
            raw = str(form.get("jd_text", ""))
            file = form.get("file")
            if file and hasattr(file, "read"):
                content = await file.read(settings.max_upload_bytes + 1)
                if len(content) > settings.max_upload_bytes:
                    raise ValueError("JD file exceeds the upload size limit")
                if (file.filename or "").lower().endswith(".txt"):
                    raw = content.decode("utf-8-sig")
                else:
                    raw, _ = await run_in_threadpool(parse, content, file.filename or "upload")
                await file.close()
            payload = {"jd_text": raw, "weights": json.loads(str(form.get("weights", "{}")))}
            if form.get("shortlist_size"):
                payload["shortlist_size"] = int(str(form["shortlist_size"]))
            if form.get("resume_ids"):
                payload["resume_ids"] = json.loads(str(form["resume_ids"]))
            if form.get("wait_for_github"):
                payload["wait_for_github"] = str(form["wait_for_github"])
        else:
            payload = await request.json()
        parsed = ScreeningRequest.model_validate(payload)
    except (ValueError, ValidationError, TypeError):
        raise HTTPException(422, "Provide JD text (10-30000 characters) or a readable PDF/DOCX/TXT file, a valid shortlist size, and weights summing to 1")
    jd_id = await run_in_threadpool(create_screening, parsed)
    return {"jd_id": jd_id}


@app.get("/screenings/{jd_id}/status", response_model=RunStatus)
def screening_status(jd_id: UUID):
    return status(ensure_jd(jd_id))


@app.get("/screenings/{jd_id}/github")
def github_progress(jd_id: UUID):
    return github_jobs.progress(ensure_jd(jd_id))


@app.post("/screenings/{jd_id}/github/extend", status_code=202)
def extend_github(jd_id: UUID, request: GithubExtendRequest):
    key = ensure_jd(jd_id)
    if read_run(key).get("status") in ("queued", "running", "evaluating"):
        raise HTTPException(409, "Wait for resume-based ranking before extending the GitHub shortlist")
    return {"queued": github_jobs.enqueue(key, request.count)}


@app.post("/screenings/{jd_id}/github/retry", status_code=202)
def retry_github(jd_id: UUID, request: GithubRetryRequest):
    return {"queued": github_jobs.retry(ensure_jd(jd_id), request.resume_ids)}


@app.get("/screenings/{jd_id}/notifications")
def list_notifications(jd_id: UUID):
    key = ensure_jd(jd_id)
    return {"dry_run": settings.email_dry_run, "awaiting_review": can_review(key) and not notifications.reviewed(key),
            "notifications": notifications.listed(key)}


@app.post("/screenings/{jd_id}/notifications/approve", status_code=202)
def approve_notifications(jd_id: UUID, approval: NotificationApproval):
    key = ensure_jd(jd_id)
    if not can_review(key) and not notifications.reviewed(key):
        raise HTTPException(409, "This screening is not awaiting email review")
    try:
        notifications.review(key, approval.resume_ids or [], all_selected=approval.all_selected,
                             override=approval.override, rejected_ids=approval.rejected_resume_ids)
    except notifications.ReviewError as exc:
        raise HTTPException(409, str(exc))
    schedule(key)
    return {"review_recorded": True, "dry_run": settings.email_dry_run}


@app.post("/screenings/{jd_id}/notifications/reject")
def reject_notifications(jd_id: UUID, candidates: NotificationCandidates):
    key = ensure_jd(jd_id)
    if not can_review(key):
        raise HTTPException(409, "This screening is not awaiting email review")
    try:
        notifications.review(key, candidates.resume_ids, remove_type=candidates.email_type)
    except notifications.ReviewError as exc:
        raise HTTPException(409, str(exc))
    return {"status": "skipped"}


@app.post("/screenings/{jd_id}/notifications/retry", status_code=202)
def retry_notifications(jd_id: UUID, candidates: NotificationCandidates):
    key = ensure_jd(jd_id)
    if read_run(key).get("status") != "done":
        raise HTTPException(409, "Wait until email processing finishes before retrying")
    queued = notifications.retry(key, candidates.resume_ids, candidates.email_type)
    if queued:
        schedule_delivery(key)
    return {"queued": queued, "dry_run": settings.email_dry_run}


@app.get("/screenings/{jd_id}/results", response_model=list[ScreeningResult])
@timed_stage("output.read")
def screening_results(jd_id: UUID):
    ensure_jd(jd_id)
    rows = fetch("""SELECT s.*,r.file_name,r.years_exp,COALESCE(
        g.status IN ('failed','rate_limited') OR (g.status='pending' AND g.retry_after IS NOT NULL) OR
        (g.status='blocked_config' AND (g.last_error_category='reserve_exceeds_limit' OR %s)), false) AS github_retryable
        FROM screening_results s JOIN resumes r ON r.id=s.resume_id
        LEFT JOIN github_jobs g ON g.run_id=s.jd_id AND g.candidate_id=s.resume_id
        WHERE s.jd_id=%s ORDER BY s.rank NULLS LAST,s.similarity DESC,s.resume_id""",
        (bool(settings.github_token.get_secret_value()), jd_id))
    return [dict(row, **report_sections(row), github_status=(row.get("github_json") or {}).get(
        "status", "not_queued" if row["status"] == "done" else None)) for row in rows]


@app.post("/screenings/{jd_id}/rerank", response_model=list[ScreeningResult])
def rerank_screening(jd_id: UUID, weights: Weights):
    key = ensure_jd(jd_id)
    run = read_run(key)
    if run.get("status") in ("queued", "running", "evaluating", "notifying"):
        raise HTTPException(409, "Wait until evaluation finishes before changing weights")
    rerank(key, weights)
    if run and run.get("status") not in ("awaiting_approval", "awaiting_github"):
        request = dict(run["request"], weights=weights.model_dump())
        write_run(key, {"request": request, "weights": weights.model_dump()})
    return screening_results(jd_id)


@app.get("/health")
def health():
    try:
        fetch("SELECT 1")
        database = True
    except Exception:
        database = False
    data = {"database": database, "llm_loaded": llm_service._model is not None,
        "embeddings_loaded": embed_service._model is not None, "gpu_layers_offloaded": llm_service.gpu_layers_offloaded}
    return JSONResponse(data, status_code=200 if all((database, data["llm_loaded"], data["embeddings_loaded"])) else 503)
