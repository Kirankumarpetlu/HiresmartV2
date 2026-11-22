import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, Send, interrupt
from psycopg.types.json import Jsonb

from app.cache import evaluation_key, get_evaluation, put_evaluation
from app.config import settings
from app.db import execute, fetch, get_pool, pool_object
from app.integrations.github_client import RateLimited
from app.integrations import github_jobs
from app.integrations.github_governor import ConfigBlocked
from app.models import EvaluatorOutput, ParsedJD, ScreeningRequest, Weights
from app.notify import notifications
from app.screening.anonymize import anonymize
from app.screening.evaluator import PROMPT_VERSION as EVALUATOR_PROMPT_VERSION, evaluate
from app.screening.github_agent import GithubEvidence, collect, github_status, match_projects, score_profile
from app.screening.jd_parser import parse_jd
from app.screening.notify_policy import decision
from app.screening.ranking import rerank
from app.screening.report import report_node as write_reports
from app.screening.retrieval import best_chunks, best_chunks_for_resumes, retrieve
from app.screening.stage1_filter import should_filter
from app.screening.state import CandidateState, EvaluationUpdate, ScreeningState
from app.screening.verifier import verify
from app.timing import timed_stage

log = logging.getLogger(__name__)
_runner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="screening-runner")
checkpointer = PostgresSaver(pool_object())
CONFIG_RETRY = RetryPolicy(max_attempts=2, initial_interval=0.5, max_interval=2, jitter=False)


def thread_config(jd_id: str) -> dict:
    # Keep mapped evaluations in shortlist order on the single LLM.
    return {"configurable": {"thread_id": str(jd_id)}, "max_concurrency": 1}


def setup_checkpointer() -> None:
    get_pool()
    # PostgreSQL checkpoint migrations create indexes concurrently.
    with PostgresSaver.from_conn_string(settings.database_url.get_secret_value()) as saver:
        saver.setup()


@timed_stage("screening.jd_parse")
def parse_node(state: ScreeningState) -> dict:
    _, jd, vectors = parse_jd(state["request"]["jd_text"], state["jd_id"])
    return {"jd_parsed": jd.model_dump(mode="json"), "query_vectors": vectors.tolist(), "status": "running"}


@timed_stage("screening.retrieve")
def retrieval_node(state: ScreeningState) -> dict:
    jd = ParsedJD.model_validate(state["jd_parsed"])
    vectors = np.asarray(state["query_vectors"], dtype=np.float32)
    candidates = retrieve(jd, state["request"].get("shortlist_size"), vectors=vectors,
        resume_ids=state["request"].get("resume_ids"), all_candidates=True)
    chunks = best_chunks_for_resumes([str(row["id"]) for row in candidates], vectors) if candidates else {}
    shortlist = []
    for resume in candidates:
        resume_id = str(resume["id"])
        execute("""INSERT INTO screening_results (jd_id,resume_id,similarity,status)
            VALUES (%s,%s,%s,'pending') ON CONFLICT (jd_id,resume_id) DO NOTHING""",
            (state["jd_id"], resume_id, resume["similarity"]))
        shortlist.append(dict(resume, id=resume_id, chunks=chunks.get(resume_id, [])))
    return {"shortlist": shortlist, "status": "evaluating"}


def fan_out_node(state: ScreeningState) -> dict:
    return {"status": "evaluating"}


def dispatch_candidates(state: ScreeningState):
    if not state.get("shortlist"):
        return "rank"
    return [Send("candidate", {"jd_id": state["jd_id"], "resume_id": resume["id"],
        "resume": resume, "jd_parsed": state["jd_parsed"], "query_vectors": state["query_vectors"],
        "chunks": resume.get("chunks", [])}) for resume in state["shortlist"]]


def route_node(state: CandidateState) -> dict:
    decision = "filtered_out" if should_filter(ParsedJD.model_validate(state["jd_parsed"]), state["resume"]) else "evaluate"
    return {"stage1_decision": decision}


def candidate_route(state: CandidateState) -> list[str]:
    return ["filtered_out"] if state["stage1_decision"] == "filtered_out" else ["evaluation_branch"]


def filtered_node(state: CandidateState) -> dict:
    return {"verified": {}}


@timed_stage("candidate.prepare")
def anonymize_node(state: CandidateState) -> dict:
    resume = state["resume"]
    vectors = np.asarray(state["query_vectors"], dtype=np.float32)
    chunks = state.get("chunks") or best_chunks(state["resume_id"], vectors)
    with timed_stage("candidate.anonymize"):
        clean = [dict(item, content=anonymize(item["content"], resume["clean_text"])) for item in chunks]
    key = evaluation_key(resume, state["jd_parsed"], chunks)
    cached = get_evaluation(key, EVALUATOR_PROMPT_VERSION)
    update = {"chunks": chunks, "anonymized_chunks": clean, "cache_hit": bool(cached)}
    if cached:
        update.update(evaluation=cached, verified=cached)
    return update


def after_anonymize(state: CandidateState) -> str:
    return END if state.get("cache_hit") else "evaluator"


@timed_stage("candidate.evaluate")
def evaluator_node(state: CandidateState) -> dict:
    resume = state["resume"]
    result = evaluate(ParsedJD.model_validate(state["jd_parsed"]), state["anonymized_chunks"],
        resume.get("skills") or [], resume.get("years_exp"))
    return {"evaluation": result.model_dump(mode="json")}


@timed_stage("candidate.github_collect")
def github_agent_node(state: CandidateState) -> dict:
    try:
        result = collect(state["resume_id"], state["resume"], ParsedJD.model_validate(state["jd_parsed"]))
        if isinstance(result, GithubEvidence):
            return {"github_evidence": result}
        return {"github": result}
    except RateLimited as exc:
        return {"github": github_status("rate_limited", "GitHub temporarily limited public requests.", limited=exc)}
    except ConfigBlocked as exc:
        return {"github": github_status("blocked_config", blocked=exc)}
    except Exception as exc:
        log.warning("GitHub collection failed type=%s", type(exc).__name__)
        return {"github": github_status("error", "GitHub evidence could not be collected.")}


@timed_stage("candidate.github_match")
def github_match_node(state: CandidateState) -> dict:
    evidence = state.get("github_evidence")
    if evidence is None:
        return {}
    try:
        matches = match_projects(evidence, state["resume"])
        result = score_profile(evidence, matches, ParsedJD.model_validate(state["jd_parsed"]))
        return {"github": result}
    except Exception as exc:
        log.warning("GitHub matching failed type=%s", type(exc).__name__)
        return {"github": github_status("error", "GitHub evidence could not be evaluated.")}


@timed_stage("candidate.verify")
def verifier_node(state: CandidateState) -> dict:
    if state.get("cache_hit"):
        return {}
    chunks = state["anonymized_chunks"]
    checked = verify(EvaluatorOutput.model_validate(state["evaluation"]),
        "\n\n".join(item["content"] for item in chunks), chunks)
    value = checked.model_dump(mode="json")
    put_evaluation(evaluation_key(state["resume"], state["jd_parsed"], state["chunks"]),
        EVALUATOR_PROMPT_VERSION, value)
    return {"verified": value}


@timed_stage("candidate.save")
def save_candidate_node(state: CandidateState) -> dict:
    if state["stage1_decision"] == "filtered_out":
        execute("UPDATE screening_results SET status='filtered_out' WHERE jd_id=%s AND resume_id=%s",
            (state["jd_id"], state["resume_id"]))
        result = {"resume_id": state["resume_id"], "status": "filtered_out"}
    else:
        checked = EvaluatorOutput.model_validate(state["verified"])
        github = state.get("github")
        execute("""UPDATE screening_results SET status='done',error_message=NULL,skill_report=%s,
            fit_scores=%s,red_flags=%s,github_json=%s,github_score=%s WHERE jd_id=%s AND resume_id=%s""",
            (Jsonb([skill.model_dump() for skill in checked.skills]), Jsonb(checked.fit.model_dump()),
             checked.red_flags, Jsonb(github.model_dump(mode="json")) if github else None,
             github.github_score if github else None, state["jd_id"], state["resume_id"]))
        log.info("Candidate saved jd=%s resume=%s cache_hit=%s", state["jd_id"], state["resume_id"], state.get("cache_hit", False))
        result = {"resume_id": state["resume_id"], "status": "done", "cache_hit": state.get("cache_hit", False)}
    return {"candidate_results": [result]}


evaluation_builder = StateGraph(CandidateState, output_schema=EvaluationUpdate)
evaluation_builder.add_node("anonymize", anonymize_node)
evaluation_builder.add_node("evaluator", evaluator_node)
evaluation_builder.add_edge(START, "anonymize")
evaluation_builder.add_conditional_edges("anonymize", after_anonymize, {END: END, "evaluator": "evaluator"})
evaluation_builder.add_edge("evaluator", END)

candidate_builder = StateGraph(CandidateState)
for name, function in (("route", route_node), ("filtered_out", filtered_node),
        ("evaluation_branch", evaluation_builder.compile()), ("verifier", verifier_node), ("save", save_candidate_node)):
    candidate_builder.add_node(name, function)
candidate_builder.add_edge(START, "route")
candidate_builder.add_conditional_edges("route", candidate_route, ["filtered_out", "evaluation_branch"])
candidate_builder.add_edge("filtered_out", "save")
candidate_builder.add_edge("evaluation_branch", "verifier")
candidate_builder.add_edge("verifier", "save")
candidate_builder.add_edge("save", END)
candidate_graph = candidate_builder.compile(checkpointer=False)


def run_candidate(state: CandidateState) -> dict:
    existing = fetch("SELECT status FROM screening_results WHERE jd_id=%s AND resume_id=%s",
        (state["jd_id"], state["resume_id"]))
    if existing and existing[0]["status"] in ("done", "filtered_out", "error"):
        return {"candidate_results": [{"resume_id": state["resume_id"], "status": existing[0]["status"], "resumed": True}]}
    try:
        result = candidate_graph.invoke(state, {"max_concurrency": 2})
        return {"candidate_results": result.get("candidate_results", [])}
    except Exception as exc:
        log.error("Candidate failed jd=%s resume=%s type=%s", state["jd_id"], state["resume_id"], type(exc).__name__)
        execute("UPDATE screening_results SET status='error',error_message=%s WHERE jd_id=%s AND resume_id=%s",
            (f"Evaluation failed ({type(exc).__name__}); inspect the local service log.", state["jd_id"], state["resume_id"]))
        return {"candidate_results": [{"resume_id": state["resume_id"], "status": "error"}]}


@timed_stage("screening.rank")
def rank_node(state: ScreeningState) -> dict:
    ranked = rerank(state["jd_id"], Weights.model_validate(state["weights"]))
    if ranked:
        github_jobs.enqueue(state["jd_id"], settings.github_shortlist_size, ranked=ranked)
    return {"ranked": [{"resume_id": str(row["resume_id"]), "rank": row["rank"],
        "final_score": row["final_score"]} for row in ranked], "status": "notifying"}


def report_node(state: ScreeningState) -> dict:
    write_reports({"jd_id": state["jd_id"]})
    if state.get("request", {}).get("wait_for_github") and github_jobs.awaiting(state["jd_id"]):
        execute("UPDATE job_descriptions SET github_waiting=true WHERE id=%s", (state["jd_id"],))
        return {"report": {"complete": True}, "status": "awaiting_github"}
    return {"report": {"complete": True}}


def github_gate(state: ScreeningState) -> dict:
    waiting = state["status"] == "awaiting_github"
    if waiting:
        interrupt({"jd_id": state["jd_id"], "message": "Waiting for queued GitHub evidence."})
        execute("UPDATE job_descriptions SET github_waiting=false WHERE id=%s", (state["jd_id"],))
    if not state.get("request", {}).get("send_notifications", True):
        return {"status": "done"}
    if waiting:
        return {"status": "notifying"}
    return {}


def select_for_email(state: ScreeningState) -> dict:
    rows = fetch("""SELECT resume_id,status,rank,final_score,recommendation,skill_report,github_json,github_score
        FROM screening_results WHERE jd_id=%s AND status='done' ORDER BY rank,resume_id""", (state["jd_id"],))
    selection = []
    for row in rows:
        reason = decision(row)
        limit = state.get("request", {}).get("shortlist_size") or settings.shortlist_size
        reason["checks"]["interview_shortlist"] = row["rank"] is not None and row["rank"] <= limit
        reason["proposed"] = all(reason["checks"].values())
        item = {"resume_id": str(row["resume_id"]), "decision_reason": reason, "email_type": "selected"}
        selection.append(item)
        if settings.send_rejection_emails and row["recommendation"] == "not_a_fit":
            selection.append(dict(item, email_type="rejected", decision_reason=dict(reason, proposed=True)))
    return {"email_selection": selection}


def draft_emails(state: ScreeningState) -> dict:
    selection = state["email_selection"]
    notifications.draft(state["jd_id"], state["jd_parsed"].get("title"), selection)
    manual = not settings.email_auto_send or any(item["email_type"] == "rejected" for item in selection)
    return {"status": "awaiting_approval" if selection and manual else "notifying"}


def approval_gate(state: ScreeningState) -> dict:
    if state["status"] == "awaiting_approval":
        interrupt({"jd_id": state["jd_id"], "message": "Review candidate email drafts."})
        if not notifications.reviewed(state["jd_id"]):
            raise RuntimeError("Email review was not recorded")
    elif state["email_selection"]:
        notifications.review(state["jd_id"], all_selected=True, automatic=True)
    return {"status": "notifying"}


def send_emails(state: ScreeningState) -> dict:
    if state["email_selection"]:
        notifications.deliver(state["jd_id"])
    return {"status": "done"}


builder = StateGraph(ScreeningState)
builder.add_node("parse_jd", parse_node, retry_policy=CONFIG_RETRY)
builder.add_node("retrieve", retrieval_node, retry_policy=CONFIG_RETRY)
builder.add_node("fan_out_candidates", fan_out_node)
builder.add_node("candidate", run_candidate)
builder.add_node("rank", rank_node, retry_policy=CONFIG_RETRY)
builder.add_node("report", report_node)
builder.add_node("github_gate", github_gate)
builder.add_node("select_for_email", select_for_email)
builder.add_node("draft_emails", draft_emails)
builder.add_node("approval_gate", approval_gate)
builder.add_node("send_emails", send_emails)
builder.add_edge(START, "parse_jd")
builder.add_edge("parse_jd", "retrieve")
builder.add_edge("retrieve", "fan_out_candidates")
builder.add_conditional_edges("fan_out_candidates", dispatch_candidates, ["candidate", "rank"])
builder.add_edge("candidate", "rank")
builder.add_edge("rank", "report")
builder.add_edge("report", "github_gate")
builder.add_conditional_edges("github_gate",
    lambda state: END if state["status"] == "done" else "select_for_email", [END, "select_for_email"])
builder.add_edge("select_for_email", "draft_emails")
builder.add_edge("draft_emails", "approval_gate")
builder.add_edge("approval_gate", "send_emails")
builder.add_edge("send_emails", END)
graph = builder.compile(checkpointer=checkpointer)


def waiting_for_approval(snapshot) -> bool:
    return "approval_gate" in snapshot.next and any(task.interrupts for task in snapshot.tasks)


def waiting_for_github(snapshot) -> bool:
    return "github_gate" in snapshot.next and any(task.interrupts for task in snapshot.tasks)


def can_review(jd_id: str) -> bool:
    return waiting_for_approval(graph.get_state(thread_config(jd_id)))


@timed_stage("screening.total")
def run_screening(jd_id: str) -> None:
    config = thread_config(jd_id)
    try:
        snapshot = graph.get_state(config)
        if not snapshot.next:
            return
        pending = None
        if waiting_for_approval(snapshot):
            if not notifications.reviewed(jd_id):
                return
            pending = Command(resume=True)
        elif waiting_for_github(snapshot):
            if github_jobs.awaiting(jd_id):
                return
            pending = Command(resume=True)
        for event in graph.stream(pending, config, stream_mode="updates"):
            log.info("Screening graph event jd=%s nodes=%s", jd_id, list(event))
    except Exception as exc:
        log.error("Screening failed jd=%s type=%s", jd_id, type(exc).__name__)
        graph.update_state(config, {"status": "error", "error": f"Screening failed ({type(exc).__name__}). Check the local service log."})


def schedule(jd_id: str) -> None:
    _runner.submit(run_screening, str(jd_id))


def schedule_delivery(jd_id: str) -> None:
    _runner.submit(notifications.deliver, jd_id)


def schedule_ready_github_runs() -> None:
    for row in github_jobs.ready_screenings():
        key = str(row["id"])
        if waiting_for_github(graph.get_state(thread_config(key))):
            schedule(key)


def create_screening(request: ScreeningRequest) -> str:
    jd_id = str(fetch("INSERT INTO job_descriptions (raw_text) VALUES (%s) RETURNING id", (request.jd_text,))[0]["id"])
    initial: ScreeningState = {"jd_id": jd_id, "request": request.model_dump(mode="json"),
        "weights": request.weights.model_dump(), "candidate_results": [], "ranked": [],
        "report": {}, "status": "queued", "error": None}
    graph.update_state(thread_config(jd_id), initial, as_node="__start__")
    schedule(jd_id)
    return jd_id


def read_run(jd_id: str) -> dict:
    snapshot = graph.get_state(thread_config(jd_id))
    return dict(snapshot.values) if snapshot.values else {}


def write_run(jd_id: str, values: dict) -> None:
    config = thread_config(jd_id)
    snapshot = graph.get_state(config)
    graph.update_state(config, values, as_node=None if snapshot.values else "__start__")


def recover_runs() -> int:
    recovered = 0
    for row in fetch("SELECT DISTINCT thread_id FROM checkpoints WHERE checkpoint_ns = ''"):
        jd_id = row["thread_id"]
        snapshot = graph.get_state(thread_config(jd_id))
        if (snapshot.next and snapshot.values.get("status") != "error"
                and (not waiting_for_approval(snapshot) or notifications.reviewed(jd_id))
                and (not waiting_for_github(snapshot) or not github_jobs.awaiting(jd_id))):
            schedule(jd_id)
            recovered += 1
    for row in fetch("SELECT DISTINCT jd_id FROM notifications WHERE status='approved'"):
        jd_id = str(row["jd_id"])
        if not graph.get_state(thread_config(jd_id)).next:
            schedule_delivery(jd_id)
    return recovered


def status(jd_id: str) -> dict:
    run = read_run(jd_id)
    rows = fetch("SELECT status,count(*) AS count FROM screening_results WHERE jd_id=%s GROUP BY status", (jd_id,))
    counts = {row["status"]: row["count"] for row in rows}
    return {"total": sum(counts.values()), "done": counts.get("done", 0), "errors": counts.get("error", 0),
        "filtered_out": counts.get("filtered_out", 0), "state": run.get("status", "interrupted"),
        "error_message": run.get("error")}
