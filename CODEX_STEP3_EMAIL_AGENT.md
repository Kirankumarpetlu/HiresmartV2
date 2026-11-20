# STEP 3 — Candidate Selection Email Agent

Prerequisites: Step 1 (LangGraph orchestration, caching) and Step 2 (GitHub agent) are complete. Read `README.md` and both earlier step files first. All README invariants apply.

## 1. Goal

After resume evaluation and GitHub evaluation are complete, an **Email agent** (LangGraph node) decides who is selected for the next round and sends each selected candidate an email: "You have been selected for the further round."

## 2. Design rules

- README states recommendations support human review and are not autonomous hiring decisions. Therefore: **human approval is required before any email is sent** (LangGraph `interrupt` / human-in-the-loop). A config flag may enable auto-send, default `false`.
- **Selection is decided in Python**, deterministically, from stored results. The LLM never decides who gets an email.
- Email text comes from **templates**. The LLM, if used at all, may only polish tone inside fixed placeholders, and must not add facts, promises, dates or scores. Default: templates only, no LLM.
- Never put scores, ranks, red flags, GitHub findings or other candidates' information in an email to a candidate.
- Candidate contact data (email, name) is used only by this node. It must never enter any LLM prompt (consistent with the anonymization rule).
- Idempotent: never send the same email twice for the same `(jd_id, resume_id, email_type)`, even after a restart or retry.
- Dry-run is the default (`EMAIL_DRY_RUN=true`): render and store the emails, send nothing.
- Failure of one email never stops the others.

## 3. Task A — Candidate email extraction

- `app/ingest/metadata.py`: extract the primary email from the resume text and hyperlinks (`mailto:`), validate with a strict regex plus `email-validator` (syntax only, no DNS needed). Store `contact_email` and `candidate_name` in `resumes` (private columns, never returned in list endpoints except masked, e.g. `a***@gmail.com`).
- Schema change: `db/init.sql`, models, reads/writes, migration script `scripts/migrate_add_contact.py`, tests.
- If no valid email: status `no_contact`, shown in UI so the recruiter can add one manually via `PATCH /resumes/{id}/contact`.
- Keep anonymization behavior for the Evaluator copy unchanged.

## 4. Task B — Selection policy (Python, `app/screening/notify_policy.py`)

Inputs: ranked results, Step 2 `GithubResult`, recruiter-set options.

Default rule (all configurable in `config.py` + `.env.example`, unit tested):

```text
selected if:
  status == done
  and recommendation == "interview"           (final >= INTERVIEW_THRESHOLD)
  and no missing must-have (cap not applied)
  and (github.status != "ok" or github.github_score >= GITHUB_MIN_SCORE (default 40))
  and rank <= EMAIL_MAX_SELECTED (optional, default unlimited)
```

Notes:
- Missing/unavailable GitHub never blocks selection.
- "maybe" candidates are NOT emailed automatically; surface them for the recruiter to promote manually.
- Rejection emails are **off by default** (`SEND_REJECTION_EMAILS=false`). If enabled later, they use a separate neutral template and also require approval.
- The recruiter can override selection per candidate in the approval step (add/remove).
- Record `decision_reason` (structured, internal only) for audit.

## 5. Task C — Notification storage and idempotency

New table `notifications`:

```text
id UUID PK, jd_id, resume_id, email_type ('selected'|'rejected'),
to_address (stored encrypted-at-rest is out of scope; keep within private DB),
subject, body_text, body_html,
status ('drafted'|'approved'|'sending'|'sent'|'failed'|'skipped'|'no_contact'),
attempts INT, last_error TEXT, created_at, approved_at, sent_at,
UNIQUE (jd_id, resume_id, email_type)
```

Use `INSERT ... ON CONFLICT DO NOTHING` for drafting and a status transition guarded by `UPDATE ... WHERE status='approved' RETURNING` before sending to prevent double-sends under retry/restart. Cascade on resume delete.

## 6. Task D — Email composition (`app/notify/templates.py`)

- Jinja2 templates under `app/notify/templates/` with autoescape on for HTML, plus plain-text alternative.
- Placeholders: candidate first name (fall back to "Candidate"), job title (from parsed JD), company name (`COMPANY_NAME` config), recruiter name/contact (`RECRUITER_NAME`, `RECRUITER_EMAIL`), next-step instruction text (`NEXT_STEP_TEXT` config, e.g. "Please reply with your availability").
- Do not invent interview dates, salary, or links. If the recruiter wants a scheduling link, it is a config value.
- Sanitize all placeholders against header injection (strip CR/LF in subject/to).
- Subject example: `Next steps for your application – {job_title}`.

## 7. Task E — Mail transport (`app/notify/mailer.py`)

- Use stdlib `smtplib` (run in a thread) or `aiosmtplib`; `EmailMessage` with multipart text+HTML.
- Config: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_USE_TLS`/`STARTTLS`, `EMAIL_FROM`, `EMAIL_DRY_RUN=true`, `EMAIL_RATE_PER_MINUTE=20`, `EMAIL_MAX_RETRIES=3`. Never log credentials or full bodies.
- Rate limit sends, retry transient SMTP errors with backoff, mark permanent errors (invalid recipient) as `failed` without retry loops.
- Dev/test: add a `mailpit` service to `docker-compose.yml` (local SMTP catcher, loopback only) and document it. Tests use a fake transport, never real SMTP.
- Add `scripts/check_email.py` acceptance script (sends to Mailpit).

## 8. Task F — LangGraph integration

Extend the top-level graph after ranking and report:

```text
... -> rank -> report -> select_for_email -> draft_emails -> approval_gate (interrupt) -> send_emails -> END
```

- `select_for_email`: applies the policy (Task B), writes the selection into state.
- `draft_emails`: renders and stores `drafted` notifications (cached by Step 1 cache key `(jd_id, resume_id, template_version)`).
- `approval_gate`: LangGraph `interrupt()` (backed by the Postgres checkpointer from Step 1) so the run pauses with status `awaiting_approval` and survives an API restart.
- `send_emails`: processes only `approved` notifications, sequentially with rate limiting, updating status per email. One failure is isolated.
- If `EMAIL_AUTO_SEND=true` skip the interrupt (still respects dry-run). Document the risk.

The screening run status gains `awaiting_approval`; update `GET /screenings/{id}/status`.

## 9. Task G — API and UI

New endpoints (follow existing FastAPI style, validate with Pydantic):

| Method/path | Contract |
|---|---|
| `GET /screenings/{id}/notifications` | list drafts with masked address, subject, body preview, status, decision basis |
| `POST /screenings/{id}/notifications/approve` | body `{resume_ids: [...]}` (or `all_selected: true`); marks approved and resumes the graph |
| `POST /screenings/{id}/notifications/reject` | skip specific candidates |
| `PATCH /resumes/{id}/contact` | set/fix the candidate email |
| `POST /screenings/{id}/notifications/retry` | retry `failed` only |

Streamlit: new "Selected candidates" panel showing checkboxes, email preview, dry-run banner, Approve & Send button, send status per candidate. Show a clear warning when `EMAIL_DRY_RUN=true`.

Re-ranking after weight changes must not send anything. It may invalidate un-sent drafts (status `drafted` → regenerate) but must never touch `sent` ones.

## 10. Tests

- Policy: thresholds, missing must-have, GitHub present/absent/low, rank cap, manual override.
- Email extraction: valid, multiple emails, mailto-only, invalid, injection attempts (`a@b.com\nBcc:...`).
- Template rendering: HTML escaping, name fallback, no scores or red flags present in output (assert on forbidden substrings).
- Idempotency: calling send twice and restart-resume sends exactly one email (fake transport counts).
- Graph: interrupt pauses run, approval resumes it, failure isolation, dry-run sends nothing.
- API: validation, masked addresses, approving a candidate who is not selected is rejected unless explicit override.
- Re-rank does not alter `sent` notifications.

## 11. Acceptance criteria

- With Mailpit running and `EMAIL_DRY_RUN=false`, approving 3 selected candidates sends exactly 3 emails visible in Mailpit; re-clicking approve sends 0 more.
- With `EMAIL_DRY_RUN=true`, nothing leaves the machine and drafts are viewable.
- Restart the API while `awaiting_approval`; the run resumes at the gate.
- No candidate PII appears in LLM prompts or `service.log`.
- All previous tests and scripts still pass.

## 12. Compliance notes (document in README)

- Emailing candidates is subject to local laws (consent, data protection). Keep a record of who was emailed and when; support deleting a candidate's data (existing delete cascade must remove notifications).
- Use a real sending domain with SPF/DKIM before production; do not use free-mail SMTP for bulk sending.
- Keep automated selection explainable (`decision_reason`) and keep the human-approval default.

## 13. Out of scope

Calendar scheduling, inbound reply parsing, SMS/WhatsApp, multi-tenant auth.
