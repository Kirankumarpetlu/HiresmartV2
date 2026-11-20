from collections.abc import Sequence
from time import monotonic, sleep
from uuid import UUID

from psycopg.types.json import Jsonb

from app.config import settings
from app.db import execute, fetch, get_pool
from app.ingest.metadata import normalize_email
from app.notify.mailer import DeliveryError, send_email
from app.notify.templates import TEMPLATE_VERSION, EmailType, render

_next_send = 0.0


class ReviewError(ValueError):
    pass


def mask_email(address: str | None) -> str | None:
    if not address:
        return None
    local, _, domain = address.partition("@")
    return f"{local[:1]}***@{domain}"


def draft(jd_id: str, title: str | None, selection: list[dict]) -> None:
    if not selection:
        return
    ids = [item["resume_id"] for item in selection]
    with get_pool().connection() as conn:
        contacts = conn.execute("""SELECT id,candidate_name,contact_email FROM resumes
            WHERE id=ANY(%s::uuid[]) ORDER BY id FOR NO KEY UPDATE""", (ids,)).fetchall()
        existing = conn.execute("""SELECT resume_id,email_type,status,approved_at,template_version
            FROM notifications WHERE jd_id=%s""", (jd_id,)).fetchall()
        by_key = {(str(row["resume_id"]), row["email_type"]): row for row in existing}
        by_id = {str(row["id"]): row for row in contacts}
        for item in selection:
            resume_id, email_type, reason = item["resume_id"], item["email_type"], item["decision_reason"]
            if resume_id not in by_id:
                continue
            previous = by_key.get((resume_id, email_type))
            if previous and (previous["template_version"] == TEMPLATE_VERSION or previous["approved_at"]
                             or previous["status"] not in ("drafted", "no_contact", "skipped")):
                continue
            contact = by_id[resume_id]
            address = normalize_email(contact["contact_email"] or "")
            status = ("drafted" if reason["proposed"] else "skipped") if address else "no_contact"
            subject, body_text, body_html = render(email_type, contact["candidate_name"], title)
            values = (address, subject, body_text, body_html, TEMPLATE_VERSION, Jsonb(reason), status)
            if previous:
                conn.execute("""UPDATE notifications SET to_address=%s,subject=%s,body_text=%s,body_html=%s,
                    template_version=%s,decision_reason=%s,status=%s WHERE jd_id=%s AND resume_id=%s
                    AND email_type=%s AND approved_at IS NULL""", (*values, jd_id, resume_id, email_type))
            else:
                conn.execute("""INSERT INTO notifications
                    (jd_id,resume_id,email_type,to_address,subject,body_text,body_html,template_version,decision_reason,status)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (jd_id,resume_id,email_type) DO NOTHING""", (jd_id, resume_id, email_type, *values))


def listed(jd_id: str) -> list[dict]:
    rows = fetch("""SELECT resume_id,email_type,to_address,subject,body_text,status,decision_reason,attempts,
        last_error,sent_at FROM notifications WHERE jd_id=%s
        ORDER BY (decision_reason->>'rank')::int NULLS LAST,resume_id,email_type""", (jd_id,))
    for row in rows:
        row["masked_address"] = mask_email(row.pop("to_address"))
        row["body_preview"] = row.pop("body_text")
    return rows


def update_contact(resume_id: UUID, address: str) -> bool:
    with get_pool().connection() as conn:
        changed = conn.execute("UPDATE resumes SET contact_email=%s WHERE id=%s RETURNING id",
                               (address, resume_id)).fetchone()
        if changed:
            conn.execute("""UPDATE notifications n SET to_address=%s,
                status=CASE WHEN n.status='no_contact' THEN
                    CASE WHEN (decision_reason->>'proposed')::boolean THEN 'drafted' ELSE 'skipped' END
                    ELSE n.status END
                FROM job_descriptions j WHERE n.jd_id=j.id AND n.resume_id=%s
                AND (n.status='failed' OR (j.email_reviewed_at IS NULL AND n.status IN ('drafted','no_contact','skipped')))""",
                (address, resume_id))
        return bool(changed)


def reviewed(jd_id: str) -> bool:
    return bool(fetch("SELECT id FROM job_descriptions WHERE id=%s AND email_reviewed_at IS NOT NULL", (jd_id,)))


def review(jd_id: str, resume_ids: Sequence[UUID] = (), *, all_selected: bool = False, override: bool = False,
           rejected_ids: Sequence[UUID] = (), remove_type: EmailType | None = None, automatic: bool = False) -> None:
    with get_pool().connection() as conn:
        job = conn.execute("SELECT email_reviewed_at FROM job_descriptions WHERE id=%s FOR UPDATE", (jd_id,)).fetchone()
        if job["email_reviewed_at"]:
            if remove_type:
                raise ReviewError("Email review is already complete")
            return
        rows = conn.execute("""SELECT id,resume_id,email_type,to_address,status,decision_reason
            FROM notifications WHERE jd_id=%s ORDER BY id FOR UPDATE""", (jd_id,)).fetchall()
        candidates = {(row["email_type"], str(row["resume_id"])): row for row in rows}

        def requested(ids: Sequence[UUID], email_type: EmailType) -> list[dict]:
            keys = {(email_type, str(resume_id)) for resume_id in ids}
            if not keys <= candidates.keys():
                raise ReviewError("One or more candidates are unavailable for this review")
            return [row for key, row in candidates.items() if key in keys]

        if remove_type:
            removed = requested(resume_ids, remove_type)
            conn.execute("""UPDATE notifications SET status='skipped',
                decision_reason=decision_reason || '{"recruiter_action":"removed"}'::jsonb
                WHERE id=ANY(%s::uuid[]) AND status IN ('drafted','skipped','no_contact')""",
                ([row["id"] for row in removed],))
            return
        if all_selected:
            chosen = [row for row in rows if row["email_type"] == "selected" and row["status"] == "drafted"
                      and row["decision_reason"]["proposed"]]
        else:
            chosen = requested(resume_ids, "selected")
        rejected = requested(rejected_ids, "rejected")
        if {row["resume_id"] for row in chosen} & {row["resume_id"] for row in rejected}:
            raise ReviewError("A candidate can receive only one decision for this screening")
        chosen += rejected
        for row in chosen:
            if not row["to_address"]:
                raise ReviewError("Add a contact email before approving this candidate")
            if row["status"] != "drafted" and not (row["status"] == "skipped" and override and row["email_type"] == "selected"):
                raise ReviewError("Candidate is not selected. Use an explicit override to promote them")
            if not row["decision_reason"]["proposed"] and not override:
                raise ReviewError("Candidate is not selected. Use an explicit override to promote them")
        # Preserve the review and its decisions in the same commit so an interrupted API can resume safely.
        conn.execute("""UPDATE notifications SET status='approved',approved_at=now(),
            decision_reason=decision_reason || jsonb_build_object('recruiter_action',
                CASE WHEN status='skipped' THEN 'override' ELSE %s::text END, 'delivery',
                CASE WHEN %s THEN 'dry_run' ELSE 'smtp' END)
            WHERE id=ANY(%s::uuid[])""",
            ("auto_send" if automatic else "approved", settings.email_dry_run, [row["id"] for row in chosen]))
        conn.execute("""UPDATE notifications SET status='skipped',
            decision_reason=decision_reason || '{"recruiter_action":"omitted"}'::jsonb
            WHERE jd_id=%s AND status='drafted'""", (jd_id,))
        conn.execute("UPDATE job_descriptions SET email_reviewed_at=now() WHERE id=%s", (jd_id,))


def retry(jd_id: str, resume_ids: list[UUID], email_type: EmailType) -> int:
    return len(fetch("""UPDATE notifications SET status='approved',last_error=NULL,
        decision_reason=decision_reason || jsonb_build_object('delivery',CASE WHEN %s THEN 'dry_run' ELSE 'smtp' END)
        WHERE jd_id=%s AND resume_id=ANY(%s::uuid[]) AND email_type=%s AND status='failed' RETURNING id""",
        (settings.email_dry_run, jd_id, resume_ids, email_type)))


def deliver(jd_id: str) -> None:
    global _next_send
    execute("""UPDATE notifications SET status='skipped',
        decision_reason=decision_reason || '{"delivery":"dry_run"}'::jsonb WHERE jd_id=%s AND status='approved'
        AND (%s OR decision_reason->>'delivery'='dry_run')""", (jd_id, settings.email_dry_run))
    if settings.email_dry_run:
        return
    for item in fetch("SELECT id FROM notifications WHERE jd_id=%s AND status='approved' ORDER BY id", (jd_id,)):
        key = item["id"]
        rows = fetch("""UPDATE notifications SET status='sending', attempts=attempts+1
            WHERE id=%s AND status='approved' RETURNING to_address,subject,body_text,body_html""", (key,))
        if not rows:
            continue
        for attempt in range(settings.email_max_retries + 1):
            sleep(max(0, _next_send - monotonic()))
            _next_send = monotonic() + 60 / settings.email_rate_per_minute
            try:
                send_email(rows[0])
            except DeliveryError as exc:
                if exc.transient and attempt < settings.email_max_retries:
                    execute("UPDATE notifications SET attempts=attempts+1 WHERE id=%s", (key,))
                    sleep(2 ** attempt)
                    continue
                status = "sending" if exc.uncertain else "failed"
                execute("UPDATE notifications SET status=%s,last_error=%s WHERE id=%s", (status, str(exc), key))
            except Exception as exc:
                # A transport error without a known SMTP rejection can follow server acceptance.
                execute("UPDATE notifications SET last_error=%s WHERE id=%s", (type(exc).__name__, key))
            else:
                execute("UPDATE notifications SET status='sent',sent_at=now(),last_error=NULL WHERE id=%s", (key,))
            break
