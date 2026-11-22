import json
import operator
import os
import runpy
import sys
from typing import get_args, get_type_hints
from uuid import uuid4

import numpy as np
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command, Send

from app.models import EvaluatorOutput, FitScores, ParsedJD, SkillProficiency
from app.screening import graph as module
from app.screening.retrieval import SEARCH_SQL
from app.screening.state import ScreeningState


def candidate(key: str) -> dict:
    return {"id": key, "skills": ["Python"], "clean_text": "Skills\nPython", "years_exp": 2,
        "similarity": 0.8, "chunks": [{"id": f"chunk-{key}", "section": "skills", "content": "Skills: Python"}]}


def test_send_fanout_and_candidate_result_reducer():
    state = {"jd_id": "job", "jd_parsed": {}, "query_vectors": [[0.0]],
        "shortlist": [candidate("a"), candidate("b"), candidate("c")]}
    sends = module.dispatch_candidates(state)
    assert len(sends) == 3 and all(isinstance(item, Send) and item.node == "candidate" for item in sends)
    annotation = get_type_hints(ScreeningState, include_extras=True)["candidate_results"]
    assert operator.add in get_args(annotation)


@pytest.mark.parametrize("filtered", [False, True])
def test_candidate_subgraph_routes_anonymizes_verifies_and_saves(monkeypatch, filtered):
    changes, calls = [], []
    monkeypatch.setattr(module, "should_filter", lambda *args: filtered)
    monkeypatch.setattr(module, "execute", lambda sql, params: changes.append((sql, params)))
    monkeypatch.setattr(module, "get_evaluation", lambda *args: None)
    monkeypatch.setattr(module, "put_evaluation", lambda *args: calls.append("cache"))

    def evaluate(*args):
        calls.append("evaluate")
        assert "person@example.com" not in args[1][0]["content"]
        return EvaluatorOutput(skills=[SkillProficiency(skill="Python", is_must_have=True,
            level=4, evidence=["Invented evidence"], reasoning="", confidence=1)],
            fit=FitScores(experience=50, domain=0, education=0, notes=""), red_flags=[])

    monkeypatch.setattr(module, "evaluate", evaluate)
    resume = candidate("candidate")
    resume["clean_text"] = "Skills: Python\nContact: person@example.com"
    resume["chunks"][0]["content"] += "\nContact: person@example.com"
    updates = list(module.candidate_graph.stream({"jd_id": "job", "resume_id": "candidate",
        "resume": resume, "jd_parsed": ParsedJD(title="Engineer", must_have=["Python"],
        nice_to_have=[], responsibilities=[]).model_dump(mode="json"), "query_vectors": [[0.0]],
        "chunks": resume["chunks"]}, stream_mode="updates"))
    names = [next(iter(update)) for update in updates]
    if filtered:
        assert names == ["route", "filtered_out", "save"]
    else:
        assert names == ["route", "evaluation_branch", "verifier", "save"]
    assert len(changes) == 1
    if filtered:
        assert calls == [] and "status='filtered_out'" in changes[0][0]
    else:
        assert calls == ["evaluate", "cache"] and "status='done'" in changes[0][0]
        skill = changes[0][1][0].obj[0]
        assert skill["level"] == 1 and skill["evidence"] == []
        assert changes[0][1][3:5] == (None, None)


def test_resume_phase_saves_and_queues_without_calling_github(monkeypatch):
    changes, queued = [], []
    jd = ParsedJD(title="Engineer", must_have=[], nice_to_have=[], responsibilities=[])
    resume = candidate("candidate")
    monkeypatch.setattr(module, "fetch", lambda *args: [])
    monkeypatch.setattr(module, "get_evaluation", lambda *args: None)
    monkeypatch.setattr(module, "put_evaluation", lambda *args: None)
    monkeypatch.setattr(module, "execute", lambda sql, params: changes.append((sql, params)))
    monkeypatch.setattr(module, "parse_jd", lambda *args: ("job", jd, np.zeros((1, 2))))
    monkeypatch.setattr(module, "retrieve", lambda *args, **kwargs: [resume])
    monkeypatch.setattr(module, "best_chunks_for_resumes", lambda *args: {resume["id"]: resume["chunks"]})
    monkeypatch.setattr(module, "rerank", lambda *args: [{"resume_id": "candidate", "rank": 1, "final_score": 50}])
    monkeypatch.setattr(module.github_jobs, "enqueue", lambda *args, **kwargs: queued.append(args))
    monkeypatch.setattr(module, "write_reports", lambda *args: {})

    def collect(*args):
        pytest.fail("Resume evaluation must not call GitHub")

    def email(*args, **kwargs):
        pytest.fail("A ranking-only run must skip the email workflow")

    def evaluate(*args):
        return EvaluatorOutput(skills=[], fit=FitScores(experience=50, domain=0, education=0, notes=""), red_flags=[])

    monkeypatch.setattr(module, "collect", collect)
    monkeypatch.setattr(module, "evaluate", evaluate)
    monkeypatch.setattr(module.settings, "email_auto_send", True)
    monkeypatch.setattr(module.settings, "email_dry_run", False)
    for method in ("draft", "review", "deliver"):
        monkeypatch.setattr(module.notifications, method, email)
    local = module.builder.compile(checkpointer=InMemorySaver())
    config = module.thread_config("job")
    updates = list(local.stream({"jd_id": "job", "request": {"jd_text": "Engineer", "send_notifications": False},
        "weights": {}, "candidate_results": []}, config, stream_mode="updates"))
    assert [next(iter(update)) for update in updates][-3:] == ["rank", "report", "github_gate"]
    result = local.get_state(config).values
    assert result["status"] == "done"
    assert queued == [("job", module.settings.github_shortlist_size)]
    assert result["candidate_results"][0]["status"] == "done"
    saved = next(params for sql, params in changes if "status='done'" in sql)
    assert saved[3:5] == (None, None)


def test_opt_in_waits_for_github_before_email_selection(monkeypatch):
    writes = []
    monkeypatch.setattr(module, "write_reports", lambda *args: {})
    monkeypatch.setattr(module.github_jobs, "awaiting", lambda *args: True)
    monkeypatch.setattr(module, "execute", lambda sql, params: writes.append(sql))
    monkeypatch.setattr(module, "fetch", lambda *args: [])
    local = module.builder.compile(checkpointer=InMemorySaver())
    config = module.thread_config("wait-github")
    local.update_state(config, {"jd_id": "wait-github", "request": {"wait_for_github": True},
                               "jd_parsed": {"title": "Engineer"}, "status": "notifying"}, as_node="rank")
    result = local.invoke(None, config)
    assert result["status"] == "awaiting_github" and module.waiting_for_github(local.get_state(config))
    assert ["github_waiting=true" in sql for sql in writes] == [True]
    assert local.invoke(Command(resume=True), config)["status"] == "done"
    assert "github_waiting=false" in writes[-1]


def test_top_graph_send_fanout_failure_isolation_and_resume(monkeypatch):
    jd = ParsedJD(title="Engineer", must_have=["Python"], nice_to_have=[], responsibilities=[])
    resumes = [candidate("done"), candidate("bad"), candidate("good")]
    calls, changes = [], []
    monkeypatch.setattr(module, "parse_jd", lambda *args: ("job", jd, np.zeros((1, 2))))
    monkeypatch.setattr(module, "retrieve", lambda *args, **kwargs: resumes)
    monkeypatch.setattr(module, "best_chunks_for_resumes", lambda ids, vectors: {row["id"]: row["chunks"] for row in resumes})
    monkeypatch.setattr(module, "execute", lambda sql, params: changes.append((sql, params)))

    def fetch(sql, params=None):
        if sql.startswith("SELECT status"):
            return [{"status": "done"}] if params[1] == "done" else [{"status": "pending"}]
        return []

    monkeypatch.setattr(module, "fetch", fetch)

    def invoke(state, config):
        assert config["max_concurrency"] == 2
        calls.append(state["resume_id"])
        if state["resume_id"] == "bad":
            raise RuntimeError("forced")
        return {"candidate_results": [{"resume_id": state["resume_id"], "status": "done"}]}

    monkeypatch.setattr(module.candidate_graph, "invoke", invoke)
    monkeypatch.setattr(module, "rerank", lambda *args: [])
    monkeypatch.setattr(module, "write_reports", lambda *args: {})
    local = module.builder.compile(checkpointer=InMemorySaver())
    result = local.invoke({"jd_id": "job", "request": {"jd_text": "Python engineer", "shortlist_size": 3},
        "weights": {"proficiency": .55, "experience": .2, "domain": .15, "education": .1},
        "candidate_results": [], "status": "queued"}, {"configurable": {"thread_id": "job"}})
    assert calls == ["bad", "good"]
    assert [item["status"] for item in result["candidate_results"]] == ["done", "error", "done"]
    assert any("status='error'" in sql for sql, _ in changes)


def test_evaluation_cache_preserves_scoring(monkeypatch):
    output = EvaluatorOutput(skills=[SkillProficiency(skill="Python", is_must_have=True,
        level=3, evidence=["Built Python APIs"], reasoning="", confidence=.9)],
        fit=FitScores(experience=80, domain=70, education=60, notes=""), red_flags=[])
    from app.screening.ranking import score
    before = score(output)
    restored = EvaluatorOutput.model_validate(output.model_dump(mode="json"))
    assert score(restored) == before


@pytest.mark.skipif(os.environ.get("RUN_POSTGRES_TESTS") != "1", reason="requires local PostgreSQL/pgvector")
def test_postgres_resume_skips_candidate_committed_before_interruption(monkeypatch, tmp_path):
    monkeypatch.setattr(module.settings, "email_dry_run", True)
    monkeypatch.setattr(module.settings, "email_auto_send", True)
    monkeypatch.setattr(module.settings, "send_rejection_emails", False)
    module.setup_checkpointer()
    jd = ParsedJD(title="Engineer", must_have=["Python"], nice_to_have=[], responsibilities=[])
    key = str(module.fetch("INSERT INTO job_descriptions (raw_text) VALUES ('Synthetic recovery JD') RETURNING id")[0]["id"])
    config = module.thread_config(key)
    resumes, evaluated = [], []
    try:
        vector = np.zeros(768, dtype=np.float32)
        vector[0] = 1
        for index in range(2):
            resume_id = str(uuid4())
            module.execute("""INSERT INTO resumes (id,file_name,file_path,clean_text,embed_model)
                VALUES (%s,'Synthetic recovery resume',%s,'Python','test')""", (resume_id, resume_id))
            resumes.append(candidate(resume_id))
            module.execute("""INSERT INTO resume_chunks (resume_id,section,content,embedding)
                VALUES (%s,'skills','Python',%s)""", (resume_id, vector))
        with module.get_pool().connection() as conn:
            conn.execute("SET LOCAL enable_seqscan=off")
            conn.execute("SET LOCAL hnsw.ef_search=100")
            plan = conn.execute("EXPLAIN (ANALYZE, COSTS OFF) " + SEARCH_SQL, {
                "q": vector, "min_years": 0, "embed_model": "test", "chunk_limit": 1000,
                "top_chunks": 3, "n": 2, "resume_ids": [row["id"] for row in resumes],
            }).fetchall()
        assert any("idx_chunks_embedding" in row["QUERY PLAN"] for row in plan)
        monkeypatch.setattr(module, "parse_jd", lambda *args: (key, jd, np.zeros((1, 768))))
        monkeypatch.setattr(module, "retrieve", lambda *args, **kwargs: resumes)
        monkeypatch.setattr(module, "best_chunks_for_resumes", lambda *args: {row["id"]: row["chunks"] for row in resumes})
        monkeypatch.setattr(module, "get_evaluation", lambda *args: None)
        monkeypatch.setattr(module, "put_evaluation", lambda *args: None)
        monkeypatch.setattr(module, "write_reports", lambda *args: {})

        def evaluate(jd, chunks, *args):
            evaluated.append(chunks[0]["id"])
            return EvaluatorOutput(skills=[], fit=FitScores(experience=100, domain=50, education=50, notes=""), red_flags=[])

        monkeypatch.setattr(module, "evaluate", evaluate)
        state = {"jd_id": key, "request": {"jd_text": "Synthetic recovery JD"},
            "weights": {"proficiency": .55, "experience": .2, "domain": .15, "education": .1}, "status": "queued"}
        url = module.settings.database_url.get_secret_value()
        with PostgresSaver.from_conn_string(url) as saver:
            paused = module.builder.compile(checkpointer=saver, interrupt_before=["candidate"])
            paused.invoke(state, config)
            snapshot = paused.get_state(config)
            first = module.dispatch_candidates(snapshot.values)[0]
            # Simulate the candidate committing before its parent graph records the send result.
            module.run_candidate(first.arg)
        scheduled = []
        monkeypatch.setattr(module, "schedule", scheduled.append)
        module.recover_runs()
        assert key in scheduled
        module.run_screening(key)
        resumed = module.read_run(key)
        assert resumed["status"] == "done"
        assert evaluated == [row["chunks"][0]["id"] for row in resumes]
        assert {row["status"] for row in resumed["candidate_results"]} == {"done"}

        failed_id = resumes[1]["id"]
        module.execute("UPDATE screening_results SET status='error' WHERE jd_id=%s AND resume_id=%s", (key, failed_id))
        (tmp_path / "recovery_acceptance.json").write_text(json.dumps({"jd_id": key}), encoding="utf-8")
        monkeypatch.setattr(module.settings, "data_dir", tmp_path)
        monkeypatch.setattr(sys, "argv", ["scripts/check_recovery.py", "retry"])
        with pytest.raises(SystemExit) as retry:
            runpy.run_path(sys.argv[0], run_name="__main__")
        assert retry.value.code == 0
        module.run_screening(key)
        assert module.status(key)["done"] == 2
        assert evaluated[2:] == [resumes[1]["chunks"][0]["id"]]
    finally:
        module.checkpointer.delete_thread(key)
        module.execute("DELETE FROM job_descriptions WHERE id=%s", (key,))
        module.execute("DELETE FROM resumes WHERE id=ANY(%s::uuid[])", ([row["id"] for row in resumes],))
