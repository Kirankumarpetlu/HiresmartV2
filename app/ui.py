import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
import streamlit as st

from app.config import settings

st.set_page_config(page_title="Resume Screening Analyzer", layout="wide")
st.title("Resume Screening Analyzer")
st.info("This tool recommends. A human makes the final decision.")
if settings.github_enabled and not settings.github_token.get_secret_value():
    st.warning("GitHub is enabled but no token is configured. Evidence collection is limited.")


def api(method: str, path: str, **kwargs):
    try:
        response = requests.request(method, settings.api_url.rstrip("/") + path, timeout=180, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        detail = "Backend unavailable; start uvicorn and wait for model loading."
        if exc.response is not None:
            try:
                detail = str(exc.response.json().get("detail", detail))
            except ValueError:
                pass
        st.error(detail)
        return None


def candidate_emails(key: str):
    path = f"/screenings/{key}/notifications"
    data = api("GET", path)
    if not data or (not data["notifications"] and not data["awaiting_review"]):
        return
    st.subheader("Selected candidates")
    if data["dry_run"]:
        st.warning("Dry run is enabled. Approval records your choices without sending emails.")
    selected, rejected = [], []
    override = False
    for row in data["notifications"]:
        resume_id, email_type = row["resume_id"], row["email_type"]
        suffix = f"{key}-{resume_id}-{email_type}"
        reason = row["decision_reason"]
        address = row["masked_address"] or "no contact"
        status = "dry run complete" if reason.get("delivery") == "dry_run" else row["status"]
        label = f"Review rank {reason['rank']} · {resume_id[:8]} · {email_type} · {address} · {status}"
        selectable = row["masked_address"] and (row["status"] == "drafted"
                                                or email_type == "selected" and row["status"] == "skipped")
        if data["awaiting_review"] and selectable:
            propose = row["status"] == "drafted" and email_type == "selected"
            if st.checkbox(label, value=propose, key=f"approve-{suffix}"):
                if email_type == "selected":
                    selected.append(resume_id)
                    override |= row["status"] == "skipped"
                else:
                    rejected.append(resume_id)
        else:
            st.write(label)
        with st.expander(f"Preview {email_type} email for {resume_id[:8]}"):
            st.write(row["subject"])
            st.text(row["body_preview"])
            if email_type == "selected":
                failed = [check.replace("_", " ") for check, passed in reason["checks"].items() if not passed]
                if failed:
                    st.caption("Manual promotion available. Policy checks not met: " + ", ".join(failed))
            if not row["masked_address"]:
                st.caption("Add a contact email for this candidate in Upload.")
            if row["last_error"]:
                st.caption(row["last_error"])
            if row["status"] == "sending":
                st.caption("Delivery is in progress or awaiting confirmation from the mail server.")
            if row["status"] == "failed" and st.button("Retry failed email", key=f"retry-{suffix}"):
                if api("POST", path + "/retry", json={"resume_ids": [resume_id], "email_type": email_type}):
                    st.rerun()
    if data["awaiting_review"]:
        label = "Record dry-run approval" if data["dry_run"] else "Approve & Send"
        if st.button(label if selected or rejected else "Finish review without sending", key=f"finish-{key}"):
            if api("POST", path + "/approve",
                   json={"resume_ids": selected, "rejected_resume_ids": rejected, "override": override}):
                st.rerun()


upload_tab, screen_tab, results_tab = st.tabs(["Upload", "Screen candidates", "Results"])
with upload_tab:
    files = st.file_uploader("Resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
    if st.button("Upload resumes", disabled=not files):
        with st.spinner("Parsing and indexing resumes..."):
            results = api("POST", "/resumes", files=[("files", (file.name, file.getvalue(), file.type)) for file in files])
            if results is not None:
                st.dataframe(results, hide_index=True)
    rows = api("GET", "/resumes")
    if rows:
        st.dataframe(rows, hide_index=True)
        with st.expander("Add or correct a contact email"):
            contact = st.selectbox("Candidate", rows, format_func=lambda row: f"{row['file_name']} ({row['id'][:8]})")
            address = st.text_input("Contact email")
            if st.button("Save contact email", disabled=not address):
                if api("PATCH", f"/resumes/{contact['id']}/contact", json={"email": address.strip()}):
                    st.rerun()
        with st.expander("Delete a resume"):
            selected = st.selectbox("Resume", rows, format_func=lambda row: f"{row['file_name']} ({row['id'][:8]})")
            if st.checkbox("Delete this resume and its screening results") and st.button("Delete selected resume"):
                if api("DELETE", f"/resumes/{selected['id']}"):
                    st.rerun()

with screen_tab:
    text = st.text_area("Job description", height=220)
    jd_file = st.file_uploader("Or upload a job description", type=["pdf", "docx", "txt"])
    shortlist = st.number_input("Interview shortlist size", min_value=1, max_value=100, value=settings.shortlist_size)
    wait_for_github = st.checkbox("Wait for GitHub evidence before email review")
    if st.button("Run screening", disabled=not (text.strip() or jd_file)):
        if jd_file:
            result = api("POST", "/screenings", data={"shortlist_size": shortlist, "wait_for_github": str(wait_for_github)},
                         files={"file": (jd_file.name, jd_file.getvalue(), jd_file.type)})
        else:
            result = api("POST", "/screenings", json={"jd_text": text, "shortlist_size": shortlist,
                                                     "wait_for_github": wait_for_github})
        if result:
            st.session_state["jd_id"] = result["jd_id"]
            st.success("Screening queued. Open Results for live progress.")

with results_tab:
    jd_id = st.text_input("Screening id", value=st.session_state.get("jd_id", ""))

    @st.fragment(run_every="2s")
    def live_results(key: str):
        if not key:
            st.caption("Run a screening or paste an existing screening id.")
            return
        progress = api("GET", f"/screenings/{key}/status")
        if not progress:
            return
        completed = progress["done"] + progress["errors"] + progress["filtered_out"]
        total = progress["total"]
        st.progress(completed / total if total else (1.0 if progress["state"] == "done" else 0.0), text=f"{completed} of {total} processed — {progress['state']}")
        st.caption(f"Evaluated: {progress['done']} · Errors: {progress['errors']} · Filtered: {progress['filtered_out']}")
        if progress.get("error_message"):
            st.error(progress["error_message"])
        rows = api("GET", f"/screenings/{key}/results")
        if not rows:
            if progress["state"] == "awaiting_approval":
                candidate_emails(key)
            return
        if progress["state"] in ("notifying", "awaiting_github", "awaiting_approval", "done"):
            github_progress = api("GET", f"/screenings/{key}/github")
            if github_progress:
                total = github_progress["total"]
                depth = github_progress["queue_depth"]
                core = github_progress["quota"].get("core", {})
                reset = core.get("reset_at")
                detail = f"GitHub evidence: {max(0, total - depth)} of {total} processed"
                if reset:
                    detail += f" · Quota resets {reset[11:16]} UTC"
                if core.get("remaining") is not None:
                    detail += f" · {core['remaining']} requests remaining"
                st.caption(detail)
                if github_progress["mixed_evidence"]:
                    st.info("Mixed evidence. Rankings update as GitHub scores arrive.")
                if github_progress["counts"].get("not_queued") and st.button("Queue next 25 for GitHub", key=f"github-next-{key}"):
                    api("POST", f"/screenings/{key}/github/extend", json={"count": 25})
        st.dataframe([{"rank": row["rank"], "candidate": row["resume_id"][:8], "score": row["final_score"],
            "GitHub score": row.get("github_score"), "GitHub status": row.get("github_status"),
            "recommendation": row["recommendation"], "status": row["status"]}
            for row in rows], hide_index=True)
        if progress["state"] in ("notifying", "awaiting_approval", "done", "error"):
            candidate_emails(key)
        for row in rows:
            with st.expander(f"Candidate {row['resume_id'][:8]} — {row['recommendation'] or row['status']}"):
                st.caption(f"Uploaded file: {row['file_name']}")
                st.write(row["summary"])
                if row.get("skill_report"):
                    st.dataframe(row["skill_report"], hide_index=True)
                st.write("Strengths", row["strengths"])
                st.write("Gaps", row["gaps"])
                st.write("Red flags", row["red_flags"])
                if github := row.get("github_json"):
                    rating = f"{github['github_score']}/100 · {github['status']}" if github["github_score"] is not None else github["status"]
                    st.write(f"**GitHub** {rating}")
                    if justification := github.get("score_explanation"):
                        st.text(justification)
                    else:
                        st.caption(github["summary"])
                    if github.get("error_category") == "reserve":
                        st.caption("The configured GitHub request reserve is keeping this candidate queued.")
                    if row.get("github_retryable") and st.button("Retry GitHub", key=f"github-retry-{key}-{row['resume_id']}"):
                        api("POST", f"/screenings/{key}/github/retry", json={"resume_ids": [row["resume_id"]]})
                    if github["project_matches"]:
                        st.dataframe(github["project_matches"], hide_index=True)
                    for flag in github["red_flags"]:
                        st.write(flag)
                elif row.get("github_status") == "not_queued":
                    st.caption("GitHub evidence is outside this run's GitHub shortlist.")
                if row.get("error_message"):
                    st.error(row["error_message"])

    live_results(jd_id)
    with st.expander("Adjust ranking weights"):
        sliders = {"proficiency": 55, "experience": 20, "domain": 15, "education": 10}
        values = {name: st.slider(name.title(), 0, 100, default) for name, default in sliders.items()}
        github_weight = st.slider("GitHub weight (%)", 0, 99, 10)
        if st.button("Re-rank stored evaluations", disabled=not jd_id):
            total = sum(values.values())
            if not total:
                st.error("At least one weight must be positive.")
            elif api("POST", f"/screenings/{jd_id}/rerank", json={
                **{name: value / total for name, value in values.items()}, "github": github_weight / 100,
            }) is not None:
                st.success("Ranking updated without another model call.")
