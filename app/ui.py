import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
import streamlit as st

from app.config import settings

st.set_page_config(page_title="Resume Screening Analyzer", layout="wide")
st.title("Resume Screening Analyzer")
st.info("This tool recommends. A human makes the final decision.")


def api(method: str, path: str, **kwargs):
    try:
        response = requests.request(method, settings.api_url.rstrip("/") + path, timeout=180, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        detail = "Backend unavailable. Start uvicorn and wait for the models to load."
        if exc.response is not None:
            try:
                detail = str(exc.response.json().get("detail", detail))
            except ValueError:
                pass
        st.error(detail)
        return None


def show_ranking(rows: list[dict]):
    st.dataframe([{"Rank": row["rank"], "Candidate": row["file_name"], "Candidate ID": row["resume_id"][:8],
                   "Score / 100": row["final_score"], "Recommendation": row["recommendation"],
                   "Status": row["status"]} for row in rows], hide_index=True)
    st.caption("Scores combine verified skill proficiency, experience, domain and education fit. Available GitHub scores can change the order.")
    levels = ["No evidence", "Beginner", "Intermediate", "Advanced", "Expert"]
    for row in rows:
        rank = f"#{row['rank']}" if row["rank"] is not None else row["status"].replace("_", " ").title()
        with st.expander(f"{rank} · {row['file_name']} · {row['resume_id'][:8]}"):
            if row.get("error_message"):
                st.error(row["error_message"])
            if row["status"] == "filtered_out":
                st.caption("The initial resume filter found too many missing required skills.")
            if row["status"] != "done":
                continue
            if row["summary"]:
                st.text(row["summary"])
            if fit := row.get("fit_scores"):
                st.caption(f"Fit out of 100 · Experience {fit['experience']:g} · Domain {fit['domain']:g} · Education {fit['education']:g}")
                if fit["notes"]:
                    st.text(fit["notes"])
            skills = row.get("skill_report") or []
            if any(skill["is_must_have"] and skill["level"] == 0 for skill in skills):
                st.caption("A required skill has no verified evidence. This caps the overall score at 60.")
            if skills:
                st.dataframe([{"Skill": skill["skill"],
                               "Requirement": "Required" if skill["is_must_have"] else "Preferred",
                               "Proficiency": f"{skill['level']} · {levels[skill['level']]}",
                               "Confidence": f"{skill['confidence']:.0%}",
                               "Justification": skill["reasoning"], "Evidence": " | ".join(skill["evidence"])}
                              for skill in skills], hide_index=True)
            if row["gaps"]:
                st.text("Required skill gaps · " + ", ".join(row["gaps"]))
            if row["red_flags"]:
                st.text("\n".join(row["red_flags"]))


def show_github(key: str, rows: list[dict]):
    if not settings.github_enabled:
        st.info("GitHub evidence collection is disabled in the app settings.")
        return
    progress = api("GET", f"/screenings/{key}/github")
    if progress is None:
        return
    st.caption(f"{progress['scored']} scored · {progress['queue_depth']} queued or in progress")
    if progress["mixed_evidence"]:
        st.info("Rankings update as GitHub scores arrive.")
    if progress["counts"].get("not_queued") and st.button("Queue next 25 for GitHub", key=f"github-next-{key}"):
        api("POST", f"/screenings/{key}/github/extend", json={"count": 25})
    for row in rows:
        if row["status"] != "done":
            continue
        status = (row.get("github_status") or "pending").replace("_", " ")
        score = row.get("github_score")
        rating = f"{score}/100 · {status}" if score is not None else status
        with st.expander(f"{row['file_name']} · {row['resume_id'][:8]} · {rating}"):
            if github := row.get("github_json"):
                st.text(github.get("score_explanation") or github["summary"])
                if github["subscores"]:
                    st.dataframe([github["subscores"]], hide_index=True)
                if github["project_matches"]:
                    st.dataframe(github["project_matches"], hide_index=True)
                if github["signals"]:
                    st.caption("Profile signals")
                    st.json(github["signals"], expanded=False)
                if github["red_flags"]:
                    st.text("\n".join(github["red_flags"]))
            elif row.get("github_status") == "not_queued":
                st.caption("GitHub evidence is outside this run's GitHub shortlist.")
            if row.get("github_retryable") and st.button("Retry GitHub", key=f"github-retry-{key}-{row['resume_id']}"):
                api("POST", f"/screenings/{key}/github/retry", json={"resume_ids": [row["resume_id"]]})


upload_tab, screen_tab, results_tab = st.tabs(["Resumes", "Job description", "Results"])
with upload_tab:
    files = st.file_uploader("Resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
    if st.button("Upload resumes", disabled=not files):
        with st.spinner("Parsing and indexing resumes..."):
            results = api("POST", "/resumes", files=[("files", (file.name, file.getvalue(), file.type)) for file in files])
            if results is not None:
                st.dataframe(results, hide_index=True)
                st.session_state["screen_resume_ids"] = list(dict.fromkeys(
                    row["id"] for row in results if row["status"] in ("uploaded", "duplicate")
                ))
    indexed = {row["id"]: row for row in api("GET", "/resumes") or []}
    if indexed:
        with st.expander("Indexed resumes"):
            st.dataframe([{"Candidate": row["file_name"], "Candidate ID": resume_id[:8]}
                          for resume_id, row in indexed.items()], hide_index=True)
            selected = st.selectbox("Resume to delete", list(indexed),
                                    format_func=lambda key: f"{indexed[key]['file_name']} ({key[:8]})")
            if st.checkbox("Delete this resume and its screening results") and st.button("Delete selected resume"):
                if api("DELETE", f"/resumes/{selected}"):
                    st.rerun()

with screen_tab:
    text = st.text_area("Job description", height=220)
    jd_file = st.file_uploader("Or upload a job description", type=["pdf", "docx", "txt"])
    resume_ids = st.multiselect("Resumes to screen", list(indexed),
                                format_func=lambda key: f"{indexed[key]['file_name']} ({key[:8]})",
                                key="screen_resume_ids")
    if st.button("Run screening", disabled=not (resume_ids and (text.strip() or jd_file)), type="primary"):
        path = "/screenings/rank-only"
        if jd_file:
            result = api("POST", path, data={"resume_ids": json.dumps(resume_ids)},
                         files={"file": (jd_file.name, jd_file.getvalue(), jd_file.type)})
        else:
            result = api("POST", path, json={"jd_text": text, "resume_ids": resume_ids})
        if result:
            st.session_state["jd_id"] = result["jd_id"]
            st.success("Screening queued. Open Results for live progress.")

with results_tab:
    jd_id = st.text_input("Screening ID", key="jd_id").strip()

    @st.fragment(run_every="2s")
    def live_results(key: str):
        if not key:
            st.caption("Run a screening or paste an existing screening ID.")
            return
        progress = api("GET", f"/screenings/{key}/status")
        if not progress:
            return
        completed = progress["done"] + progress["errors"] + progress["filtered_out"]
        total = progress["total"]
        st.progress(completed / total if total else float(progress["state"] == "done"),
                    text=f"{completed} of {total} processed · {progress['state'].replace('_', ' ')}")
        st.caption(f"Evaluated {progress['done']} · Errors {progress['errors']} · Filtered {progress['filtered_out']}")
        if progress.get("error_message"):
            st.error(progress["error_message"])
        rows = api("GET", f"/screenings/{key}/results")
        if not rows:
            if rows == [] and progress["state"] == "done":
                st.info("No eligible resumes were found for this job description.")
            return
        ranking_tab, github_tab = st.tabs(["Candidate ranking", "GitHub evidence"])
        with ranking_tab:
            show_ranking(rows)
        with github_tab:
            if any(row["rank"] is not None for row in rows):
                show_github(key, rows)
            else:
                st.caption("GitHub evidence becomes available after candidates are ranked.")

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
