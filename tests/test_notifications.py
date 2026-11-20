import os
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.types.json import Jsonb

from app import api
from app.config import settings
from app.ingest.metadata import contact_metadata, normalize_email
from app.notify import mailer, notifications
from app.notify.templates import render
from app.screening import graph
from app.screening.anonymize import anonymize
from app.screening.notify_policy import decision


def test_contact_metadata_and_header_injection():
    assert contact_metadata("Avery Example\navery@example.org\nsecond@example.org") == {
        "candidate_name": "Avery Example", "contact_email": "avery@example.org"}
    assert contact_metadata("Avery Example", ["mailto:avery%40EXAMPLE.org?subject=Hello"])["contact_email"] == "avery@example.org"
    assert contact_metadata("invalid..email@example.org", ["mailto:a@example.org%0d%0aBcc:b@example.org"])["contact_email"] is None
    injection = "a@example.org\nBcc: target@example.org"
    assert contact_metadata(injection)["contact_email"] is None
    assert normalize_email(injection) is None
    assert "avery" not in anonymize("Contact: mailto:avery%40example.org and +avery@example.org")


@pytest.mark.parametrize(("changes", "proposed"), [
    ({}, True),
    ({"final_score": 74.99}, False),
    ({"recommendation": "maybe"}, False),
    ({"status": "error"}, False),
    ({"github_json": {"status": "rate_limited"}}, True),
    ({"github_json": {"status": "ok", "github_score": 40}}, True),
    ({"github_json": {"status": "ok", "github_score": 39}}, False),
    ({"github_json": {"status": "ok", "github_score": None}}, False),
    ({"rank": 4}, False),
    ({"skill_report": [{"skill": "Python", "level": 0, "is_must_have": True,
                       "reasoning": "", "confidence": 1, "evidence": []}]}, False),
])
def test_selection_policy(monkeypatch, changes, proposed):
    monkeypatch.setattr(settings, "email_max_selected", 3)
    monkeypatch.setattr(settings, "interview_threshold", 75)
    monkeypatch.setattr(settings, "github_min_score", 40)
    row = {"status": "done", "recommendation": "interview", "rank": 3, "final_score": 75}
    assert decision(row | changes)["proposed"] is proposed


def test_email_template_escapes_html_and_keeps_internal_data_out():
    subject, text, html = render("selected", None, "<Engineer>\r\nRole")
    assert subject.endswith("<Engineer> Role") and "\n" not in subject
    assert "Hi Candidate" in text and "&lt;Engineer&gt;" in html and "<Engineer>" not in html
    assert "You have been selected for the further round." in text
    assert all(term not in text + html for term in ("red flags", "github", "final_score", "rank"))
    assert "selected for the further round" not in render("rejected", None, None)[1]


@pytest.mark.parametrize(("error", "transient", "uncertain"), [
    (mailer.smtplib.SMTPDataError(451, "sensitive detail"), True, False),
    (mailer.smtplib.SMTPRecipientsRefused({"private@example.org": (550, "sensitive detail")}), False, False),
    (mailer.smtplib.SMTPServerDisconnected("sensitive detail"), False, True),
])
def test_smtp_failure_classification_without_network(monkeypatch, error, transient, uncertain):
    messages = []

    class FakeSMTP:
        def send_message(self, message):
            messages.append(message)
            raise error

        def close(self):
            pass

    monkeypatch.setattr(settings, "smtp_use_tls", False)
    monkeypatch.setattr(settings, "smtp_starttls", False)
    monkeypatch.setattr(settings, "smtp_username", "")
    monkeypatch.setattr(settings, "email_from", "recruiter@example.org")
    monkeypatch.setattr(settings, "recruiter_email", "")
    monkeypatch.setattr(mailer.smtplib, "SMTP", lambda *args, **kwargs: FakeSMTP())
    with pytest.raises(mailer.DeliveryError) as exc:
        mailer.send_email({"to_address": "private@example.org", "subject": "Next steps",
                           "body_text": "Hello", "body_html": "<p>Hello</p>"})
    assert messages[0].get_content_type() == "multipart/alternative"
    assert (exc.value.transient, exc.value.uncertain) == (transient, uncertain)
    assert "sensitive detail" not in str(exc.value) and "private@example.org" not in str(exc.value)


@pytest.mark.skipif(os.environ.get("RUN_POSTGRES_TESTS") != "1", reason="requires local PostgreSQL/pgvector")
@pytest.mark.parametrize("dry_run", [False, True])
def test_approval_restart_and_delivery_with_fake_transport(monkeypatch, dry_run, caplog):
    caplog.set_level("INFO")
    monkeypatch.setattr(settings, "email_dry_run", dry_run)
    monkeypatch.setattr(settings, "email_auto_send", False)
    monkeypatch.setattr(settings, "send_rejection_emails", dry_run)
    monkeypatch.setattr(settings, "email_max_selected", None)
    monkeypatch.setattr(settings, "email_max_retries", 1)
    monkeypatch.setattr(settings, "interview_threshold", 75)
    monkeypatch.setattr(settings, "github_min_score", 40)
    queued, delivered, attempts = [], [], []
    monkeypatch.setattr(graph, "schedule", queued.append)
    monkeypatch.setattr(api, "schedule", queued.append)
    monkeypatch.setattr(api, "schedule_delivery", queued.append)
    monkeypatch.setattr(notifications, "sleep", lambda seconds: None)

    def fake_send(row):
        address = row["to_address"]
        attempts.append(address)
        if address == "temp@example.org" and attempts.count(address) == 1:
            raise mailer.DeliveryError("Temporary rejection", transient=True)
        if address == "bad@example.org":
            raise mailer.DeliveryError("Permanent rejection")
        if address == "uncertain@example.org":
            raise mailer.DeliveryError("Connection lost during submission", uncertain=True)
        delivered.append(address)

    monkeypatch.setattr(notifications, "send_email", fake_send)
    monkeypatch.setattr(mailer.smtplib, "SMTP", lambda *args, **kwargs: pytest.fail("SMTP must stay mocked"))
    graph.setup_checkpointer()
    jd_id = str(graph.fetch("""INSERT INTO job_descriptions (title,raw_text)
        VALUES ('Engineer','Synthetic email test') RETURNING id""")[0]["id"])
    config = graph.thread_config(jd_id)
    client = TestClient(api.app)
    path = f"/screenings/{jd_id}/notifications"
    ids = [uuid4() for _ in range(8)]
    url = settings.database_url.get_secret_value()
    try:
        for rank, (resume_id, address) in enumerate(zip(ids, (
                "first@example.org", "temp@example.org", "bad@example.org", None, "maybe@example.org",
                "omit@example.org", "uncertain@example.org", "declined@example.org")), 1):
            graph.execute("""INSERT INTO resumes (id,file_name,file_path,clean_text,embed_model,contact_email)
                VALUES (%s,'synthetic',%s,'Python','test',%s)""", (resume_id, str(resume_id), address))
            graph.execute("""INSERT INTO screening_results
                (jd_id,resume_id,status,rank,final_score,recommendation,skill_report,fit_scores)
                VALUES (%s,%s,'done',%s,%s,%s,'[]'::jsonb,%s)""",
                (jd_id, resume_id, rank, {5: 60, 8: 45}.get(rank, 80), {5: "maybe", 8: "not_a_fit"}.get(rank, "interview"),
                 Jsonb({"experience": 80, "education": 70, "domain": 80, "notes": ""})))
        with PostgresSaver.from_conn_string(url) as first:
            monkeypatch.setattr(graph, "graph", graph.builder.compile(checkpointer=first))
            graph.graph.update_state(config, {"jd_id": jd_id, "jd_parsed": {"title": "Engineer"},
                "request": {"jd_text": "Synthetic email test"}, "status": "notifying"}, as_node="report")
            graph.run_screening(jd_id)
            assert graph.status(jd_id)["state"] == "awaiting_approval"
            assert not attempts
        with PostgresSaver.from_conn_string(url) as restarted:
            monkeypatch.setattr(graph, "graph", graph.builder.compile(checkpointer=restarted))
            graph.recover_runs()
            assert jd_id not in queued
            listing = client.get(path)
            assert listing.json()["awaiting_review"]
            assert "first@example.org" not in listing.text and "f***@example.org" in listing.text
            assert listing.json()["notifications"][3]["status"] == "no_contact"
            assert client.post(path + "/approve", json={"all_selected": True, "resume_ids": []}).status_code == 422
            assert client.post(path + "/approve", json={"resume_ids": [str(ids[4])]}).status_code == 409
            contact = client.patch(f"/resumes/{ids[3]}/contact", json={"email": "fixed@example.org"})
            assert contact.json()["contact_email"] == "f***@example.org"
            assert client.post(path + "/reject", json={"resume_ids": [str(ids[5])]}).status_code == 200
            payload = {"resume_ids": list(map(str, ids[:5] + ids[6:7])), "override": True}
            if dry_run:
                payload["rejected_resume_ids"] = [str(ids[7])]
            assert client.post(path + "/approve", json=payload).status_code == 202
            assert client.post(path + "/approve", json=payload).status_code == 202
            graph.recover_runs()
            assert jd_id in queued
            if dry_run:
                # Even after changing settings, a recorded dry-run approval stays a dry run.
                monkeypatch.setattr(settings, "email_dry_run", False)
            graph.run_screening(jd_id)
            graph.run_screening(jd_id)
            assert graph.status(jd_id)["state"] == "done"
            state = client.get(path).json()["notifications"]
            assert state[4]["decision_reason"]["recruiter_action"] == "override"
            assert state[5]["status"] == "skipped"
            if dry_run:
                assert not attempts
                assert all(row["status"] == "skipped" and row["attempts"] == 0 for row in state)
                assert state[7]["email_type"] == "rejected" and state[7]["decision_reason"]["delivery"] == "dry_run"
            else:
                assert set(delivered) == {"first@example.org", "temp@example.org", "fixed@example.org", "maybe@example.org"}
                assert len(delivered) == 4 and state[1]["attempts"] == 2 and state[2]["status"] == "failed"
                assert state[6]["status"] == "sending"
                assert client.post(path + "/retry", json={"resume_ids": [str(ids[6])]}).json()["queued"] == 0
                assert client.post(path + "/retry", json={"resume_ids": [str(ids[0])]}).json()["queued"] == 0
                assert client.post(path + "/retry", json={"resume_ids": [str(ids[2])]}).json()["queued"] == 1
                notifications.deliver(jd_id)
                assert attempts.count("bad@example.org") == 2
                assert attempts.count("uncertain@example.org") == 1
            before_rerank = client.get(path).json()["notifications"]
            assert client.post(f"/screenings/{jd_id}/rerank", json={}).status_code == 200
            assert client.get(path).json()["notifications"] == before_rerank
            assert "@example.org" not in caplog.text
    finally:
        graph.checkpointer.delete_thread(jd_id)
        graph.execute("DELETE FROM job_descriptions WHERE id=%s", (jd_id,))
        graph.execute("DELETE FROM resumes WHERE id=ANY(%s::uuid[])", (ids,))
