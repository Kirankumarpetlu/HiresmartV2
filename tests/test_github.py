import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import httpx
import numpy as np
import pytest
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from fastapi.testclient import TestClient

from app import api
from app.config import Settings
from app.ingest.metadata import github_metadata
from app.ingest.parse import parse
from app.integrations import github_client as client
from app.models import EvaluatorOutput, FitScores, GithubProjectMatch, GithubResult, ParsedJD, SkillProficiency, Weights
from app.screening import github_agent as agent, ranking
from app.screening import graph

JD = ParsedJD(title="Engineer", must_have=["Python"], nice_to_have=[], responsibilities=[])


@pytest.mark.parametrize("url", [
    "https://github.com.evil.com/alice", "https://evil.example/github.com/alice",
    "https://github.com@evil.example/alice", "https://evil@github.com/alice",
    "https://github.com:443/alice", "https://github.com/alice%2fevil", "https://github.com/alice/..",
    "ftp://github.com/alice", "https://github.com/-alice", "https://github.com/a_b",
    "https://github.com/" + "a" * 40,
])
def test_rejects_invalid_github_urls(url):
    assert github_metadata(url, [url]) == {"github_username": None, "github_repo_links": []}


def test_extracts_plain_github_profile_and_repo_paths():
    longest = "a" * 39
    result = github_metadata(f"Profile: github.com/{longest}/\nProject: (https://GitHub.com/Org/Repo/).")
    assert result == {"github_username": longest, "github_repo_links": ["https://github.com/org/repo"]}
    assert github_metadata("github.com/ABC/mine")["github_username"] == "abc"
    assert github_metadata("github.com/a/b")["github_repo_links"] == ["https://github.com/a/b"]


def test_extracts_hyperlink_only_github_from_pdf_and_docx():
    import fitz

    document = Document()
    paragraph = document.add_paragraph("Portfolio links: ")
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), paragraph.part.relate_to("https://github.com/Alice/", RELATIONSHIP_TYPE.HYPERLINK, is_external=True))
    run = OxmlElement("w:r")
    label = OxmlElement("w:t")
    label.text = "GitHub"
    run.append(label)
    link.append(run)
    paragraph._p.append(link)
    stream = io.BytesIO()
    document.save(stream)
    text, parsed = parse(stream.getvalue(), "resume.docx")
    assert "github.com" not in text and github_metadata(text, parsed["hyperlinks"])["github_username"] == "alice"

    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((30, 40), "GitHub and other portfolio links belong here.")
        page.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(30, 25, 70, 45), "uri": "https://github.com/Alice/app"})
        text, parsed = parse(pdf.tobytes(), "resume.pdf")
    assert "github.com" not in text
    assert github_metadata(text, parsed["hyperlinks"])["github_repo_links"] == ["https://github.com/alice/app"]


@pytest.fixture
def cache(monkeypatch):
    storage = {}
    monkeypatch.setattr(client.settings, "cache_enabled", True)
    monkeypatch.setattr(client, "_blocked_until", {})
    monkeypatch.setattr(client, "_secondary_failures", {})
    monkeypatch.setattr(client, "fetch", lambda sql, params: [{"payload": storage[params[0]]}] if params[0] in storage else [])
    monkeypatch.setattr(client, "execute", lambda sql, params: storage.setdefault(params[0], {}).update(params[2].obj))
    return storage


def test_http_cache_uses_etag_and_performs_zero_warm_requests(cache, monkeypatch):
    time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(client, "now", lambda: time)
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.host == "api.github.com"
        if request.headers.get("if-none-match"):
            return httpx.Response(304)
        payload = {"created_at": "2020-01-01T00:00:00Z", "followers": 1, "public_repos": 1} if request.url.path == "/users/alice" else [
            {"name": "public", "private": False}, {"name": "private", "private": True}]
        return httpx.Response(200, json=payload, headers={"etag": '"fresh"'})

    transport = httpx.MockTransport(handler)
    with client.GithubClient("Alice", transport) as github:
        first = github.profile()
        assert [repo["name"] for repo in first[1]] == ["public"] and github.network_calls == 2
    with client.GithubClient("ALICE", transport) as github:
        assert github.profile() == first and github.network_calls == 0
    time += timedelta(hours=client.settings.github_cache_ttl_hours + 1)
    with client.GithubClient("alice", transport) as github:
        assert github.profile() == first and github.network_calls == 2
    assert [req.headers.get("if-none-match") for req in requests] == [None, None, '"fresh"', '"fresh"']


@pytest.mark.parametrize("status", [404, 403, 429, "timeout"])
def test_http_missing_rate_limits_timeouts_and_retry(cache, status):
    def handler(request):
        if status == "timeout":
            raise httpx.ReadTimeout("test")
        return httpx.Response(status, headers={"x-ratelimit-reset": str(int(client.now().timestamp()) + 120),
                                              "x-ratelimit-remaining": "0"})

    transport = httpx.MockTransport(handler)
    with client.GithubClient("alice", transport) as github:
        if status == 404:
            assert github.profile() == (None, [])
        elif status == "timeout":
            with pytest.raises(httpx.ReadTimeout):
                github.profile()
        else:
            with pytest.raises(client.RateLimited):
                github.profile()
            with pytest.raises(client.RateLimited):
                github.profile()
        assert github.network_calls == (2 if status == "timeout" else 1)


def test_quota_errors_are_bucket_specific_and_forbidden_has_no_cooldown(cache, monkeypatch):
    current = datetime(2026, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(client, "now", lambda: current)
    reset = current + timedelta(seconds=17)
    responses = iter([
        httpx.Response(403, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(int(reset.timestamp())),
                                     "x-ratelimit-resource": "search"}),
        httpx.Response(403, json={"message": "Resource not accessible"}, headers={"x-ratelimit-remaining": "26"}),
        httpx.Response(429, headers={"retry-after": "8"}),
    ])
    with client.GithubClient("alice", httpx.MockTransport(lambda request: next(responses))) as github:
        with pytest.raises(client.PrimaryRateLimited) as limited:
            github.get("/search/repositories", dict, bucket="search")
        assert limited.value.retry_after == reset
        assert github.primary_limit_hits == 1
        with pytest.raises(client.PrimaryRateLimited):
            github.get("/search/repositories", dict, bucket="search")
        assert github.primary_limit_hits == 1
        with pytest.raises(client.GithubForbidden):
            github.get("/users/alice", dict)
        assert "core" not in client._blocked_until
        with pytest.raises(client.SecondaryRateLimited) as limited:
            github.get("/users/alice", dict)
        assert limited.value.retry_after == current + timedelta(seconds=8)
        assert github.secondary_limits == 1


@pytest.mark.parametrize(("limit", "cap", "expected"), [(60, 200, 6), (5000, 200, 200), (60, 3, 3), (61, 200, 7)])
def test_effective_reserve_scales_and_respects_cap(monkeypatch, limit, cap, expected):
    monkeypatch.setattr(client.settings, "github_reserve_remaining", cap)
    assert client.governor.effective_reserve(limit) == expected


@pytest.mark.parametrize(("enabled", "token", "warns"), [(True, "", True), (True, "test-token-value", False), (False, "", False)])
def test_startup_warning_checks_enabled_and_never_logs_token(monkeypatch, caplog, enabled, token, warns):
    from pydantic import SecretStr
    from app.workers import github_worker

    monkeypatch.setattr(client.settings, "github_enabled", enabled)
    monkeypatch.setattr(client.settings, "github_token", SecretStr(token))
    monkeypatch.setattr(github_worker, "init_database", lambda: None)
    launches = []
    monkeypatch.setattr(github_worker, "quota_log", launches.append)
    def stop():
        raise KeyboardInterrupt
    monkeypatch.setattr(github_worker, "run_available", stop)
    with pytest.raises(KeyboardInterrupt):
        github_worker.main()
    assert launches == (["startup"] if enabled else [])
    assert ("60 requests/hour" in caplog.text) == warns
    assert "test-token-value" not in caplog.text


def test_secondary_backoff_and_stale_cache_do_not_refresh_old_timestamp(cache, monkeypatch):
    current = datetime(2026, 1, 10, tzinfo=timezone.utc)
    at = (current - timedelta(days=8)).isoformat()
    cache["alice"] = {"/users/alice": {"at": at, "value": {"public_repos": 1}, "etag": '"previous"'}}
    monkeypatch.setattr(client, "now", lambda: current)
    transport = httpx.MockTransport(lambda _: httpx.Response(403, json={"message": "secondary rate limit"}))
    with client.GithubClient("alice", transport) as github:
        assert github.get("/users/alice", dict) == {"public_repos": 1}
        assert github.stale_at.isoformat() == at
        assert github.stale_limit.retry_after == current + timedelta(seconds=60)
    assert cache["alice"]["/users/alice"]["at"] == at
    current += timedelta(seconds=61)
    with client.GithubClient("alice", transport) as github:
        assert github.get("/users/alice", dict) == {"public_repos": 1}
        assert github.stale_limit.retry_after == current + timedelta(seconds=120)


def test_tiered_collection_uses_catalog_and_dedupes_username(cache, monkeypatch):
    requests = []

    def response(request):
        requests.append(request)
        path = request.url.path
        if path == "/users/alice":
            return httpx.Response(200, json={"public_repos": 5, "created_at": None, "followers": 0})
        if path == "/users/alice/repos":
            return httpx.Response(200, json=[repository(name=f"repo-{n}", private=False) for n in range(5)])
        if path.endswith("/languages"):
            return httpx.Response(200, json={"Python": 10})
        if path.endswith("/readme"):
            return httpx.Response(404)
        if path.endswith("/commits"):
            assert request.url.params["per_page"] == str(client.settings.github_commits_per_page)
        return httpx.Response(200, json=[])

    transport = httpx.MockTransport(response)
    monkeypatch.setattr(agent, "GithubClient", lambda username: client.GithubClient(username, transport))
    monkeypatch.setattr(agent, "fetch", lambda *args: [])
    resume = {"github_username": "Alice", "github_repo_links": [], "clean_text": ""}
    evidence = agent.collect("first", resume, JD)
    assert evidence.inspected == 5 and evidence.network_calls == 20
    assert sum(request.url.path.endswith("/languages") for request in requests) == 3
    second = agent.collect("second", resume | {"github_username": "alice"}, JD)
    assert second.network_calls == 0 and second.cache_hits == 20


def test_profile_cache_survives_mid_candidate_limit_and_direct_metadata_can_score_partial(cache, monkeypatch):
    def response(request):
        path = request.url.path
        if path == "/users/alice":
            return httpx.Response(200, json={"public_repos": 1, "created_at": None, "followers": 0})
        if path == "/users/alice/repos":
            return httpx.Response(200, json=[repository(private=False)])
        return httpx.Response(429, headers={"retry-after": "60"})

    transport = httpx.MockTransport(response)
    monkeypatch.setattr(agent, "GithubClient", lambda username: client.GithubClient(username, transport))
    monkeypatch.setattr(agent, "fetch", lambda *args: [])
    resume = {"github_username": "alice", "github_repo_links": ["https://github.com/alice/image-processor"], "clean_text": ""}
    evidence = agent.collect("resume", resume, JD)
    result = agent.score_profile(evidence, agent.match_projects(evidence, resume), JD)
    assert result.status == "partial" and result.github_score is not None and result.retry_after
    assert "/users/alice" in cache["alice"]


def repository(**overrides):
    return {"name": "image-processor", "description": "Processes images in Python", "language": "Python",
        "topics": ["python"], "fork": False, "size": 50, "license": True, "pushed_at": "2026-02-15T00:00:00Z",
        "created_at": "2026-01-01T00:00:00Z", "readme": "Processes images in Python. " * 8,
        "commits": [], "tests": True, "ci": True, "dockerfile": True, "license_file": False, "languages": {"Python": 5}} | overrides


def evidence(projects, repos, catalog=None):
    catalog = list(repos.values()) if catalog is None else catalog
    profile = {"created_at": "2021-01-01T00:00:00Z", "followers": 100, "public_repos": len(catalog)}
    return agent.GithubEvidence("alice-handle", profile, catalog, repos, projects)


def test_projects_from_merged_chunks_and_direct_links():
    url = "https://github.com/alice/image-processor"
    chunks = [{"section": "project", "content": "Project - Image Processor:\nImage Processor\nPython pipeline\n"
               "Project - Search:\nSearch\nBuilt using PostgreSQL"},
              {"section": "experience", "content": "Project: Console\nUsed Go"}]
    projects = agent.extract_projects(chunks, [url])
    assert [(project.name, project.tech) for project in projects] == [
        ("Image Processor", ["Python"]), ("Search", ["PostgreSQL"]), ("Console", ["Go"])]
    direct = agent.match_projects(evidence(projects[:1], {url: repository()}), {})
    assert direct[0].verified and direct[0].repo == url
    inaccessible = agent.match_projects(evidence(projects[:1], {}), {})
    assert inaccessible[0].match == "unverified" and not inaccessible[0].verified


def test_extracted_github_links_reach_collection_and_matching(cache, monkeypatch):
    repo = "https://github.com/alice/image-processor"
    clean_text = f"Alice Example\nhttps://github.com/alice\nProjects\nImage Processor\n{repo}"
    resume = {"id": "resume", "clean_text": clean_text, "skills": ["Python"], **github_metadata(clean_text)}
    chunks = [{"id": "chunk", "section": "project", "label": "Image Processor",
               "content": f"Project - Image Processor:\nImage Processor\n{repo}"}]
    authored = {"author": {"login": "alice"}, "commit": {"author": {"date": "2026-02-01T00:00:00Z"}}}
    responses = {"/users/alice": {"public_repos": 1, "created_at": "2021-01-01T00:00:00Z", "followers": 0},
        "/users/alice/repos": [repository(private=False)], "/repos/alice/image-processor/languages": {"Python": 100},
        "/repos/alice/image-processor/readme": {}, "/repos/alice/image-processor/commits": [authored],
        "/repos/alice/image-processor/contents": [{"type": "dir", "name": "tests"}]}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=responses[request.url.path]))
    monkeypatch.setattr(agent, "GithubClient", lambda user: client.GithubClient(user, transport))
    monkeypatch.setattr(agent, "fetch", lambda *args: chunks)
    state = {"resume_id": "resume", "resume": resume, "jd_parsed": JD.model_dump()}
    github = graph.github_match_node(state | graph.github_agent_node(state))["github"]
    assert github.status == "ok" and github.github_score is not None
    assert github.project_matches[0].verified and github.project_matches[0].authored
    assert github.project_matches[0].repo == repo


def test_fuzzy_batch_anonymizes_checks_quotes_and_caches(monkeypatch):
    url = "https://github.com/alice-handle/image-processor"
    projects = [agent.Project("Image Processer", "Alice Example wrote software for alice-handle", [], []),
                agent.Project("Image Processor", "Image conversion software", [], []),
                agent.Project("Unrelated", "Audio", [], [])]
    data = evidence(projects, {url: repository()})
    resume = {"id": "resume", "clean_text": "Alice Example\nProjects"}
    monkeypatch.setattr(agent, "cached_embeddings", lambda *args: np.eye(4))
    cached = {}
    monkeypatch.setattr(agent, "get_evaluation", lambda key, version: cached.get((key, version)))
    monkeypatch.setattr(agent, "put_evaluation", lambda key, version, value: cached.update({(key, version): value}))
    calls = []

    def generate(messages, schema):
        prompt = messages[1]["content"]
        calls.append(prompt)
        assert "alice-handle" not in prompt and "Alice Example" not in prompt
        assert len(json.loads(prompt)) == 2
        return schema.model_validate({"matches": [
            {"repo_number": 1, "match": "verified", "reason": "Similar name", "evidence_quote": "Made up"},
            {"repo_number": 1, "match": "verified", "reason": "Matching tool", "evidence_quote": "PROCESSES  images in Python"},
        ]})

    monkeypatch.setattr(agent, "generate_json", generate)
    result = agent.match_projects(data, resume)
    assert [(item.match, item.verified) for item in result] == [
        ("partial", False), ("verified", True), ("not_found", False)]
    assert result[0].evidence_quote is None
    assert agent.match_projects(data, resume) == result and len(calls) == 1


def test_profile_signals_cover_healthy_fork_bulk_and_empty(monkeypatch):
    monkeypatch.setattr(agent, "now", lambda: datetime(2026, 3, 1, tzinfo=timezone.utc))
    demo = "https://github.com/alice-handle/image-processor"
    repo = repository(commits=["2026-02-15T00:00:00Z", "2026-02-16T00:00:00Z"])
    report = agent.score_profile(evidence([], {demo: repo}), [], JD)
    assert report.subscores.jd_relevance == report.subscores.code_quality == 100
    assert report.signals.authored_repos == 1 and report.signals.active_days == 2 and not report.red_flags
    assert "8 of 8 quality checks passed" in report.score_explanation
    assert "2 attributed commits over 2 active days" in report.score_explanation
    assert "Matched 1 of 1 required skills" in report.score_explanation
    fork = repository(fork=True, size=0, commits=["2026-02-15T00:00:00Z"] * 5)
    report = agent.score_profile(evidence([], {demo: fork}, [fork] * 3), [], JD)
    assert report.signals.original_repos == 0 and report.signals.recent_commits == 5
    assert all(any(word in flag for flag in report.red_flags) for word in ("forks", "empty", "created on one day", "occurred on one day"))
    assert agent.score_profile(evidence([], {}), [], JD).github_score == 0
    with pytest.raises(ValueError, match="GITHUB_SCORE_WEIGHTS"):
        Settings(github_score_weights=(0.5, 0.2, 0.2, 0.2))


def test_score_explanation_preserves_run_weights_and_handles_legacy_and_unavailable(monkeypatch):
    monkeypatch.setattr(agent.settings, "github_score_weights", (0.5, 0.2, 0.2, 0.1))
    matches = [
        GithubProjectMatch(resume_project="Verified", match="verified", reason="Direct link", verified=True),
        GithubProjectMatch(resume_project="Partial", match="partial", reason="Some matching evidence"),
    ]
    result = agent.score_profile(evidence([], {}), matches, JD)
    assert result.github_score == 38 and result.score_weights == (0.5, 0.2, 0.2, 0.1)
    assert "Project match 75/100 at 50% contributes 37.5 points." in result.score_explanation
    assert "Verified 1 of 2 project claims. Another 1 matched partially" in result.score_explanation
    monkeypatch.setattr(agent.settings, "github_score_weights", (0.4, 0.2, 0.2, 0.2))
    restored = GithubResult.model_validate(result.model_dump(mode="json"))
    assert restored.score_explanation == result.score_explanation
    legacy = restored.model_copy(update={"score_weights": None, "score_evidence": {}})
    assert "Project match 75/100." in legacy.score_explanation and "contributes" not in legacy.score_explanation
    missing = agent.github_status("not_provided")
    assert missing.score_explanation == "No GitHub score. No GitHub profile found on resume."


def test_weights_and_rerank_use_only_stored_github_result(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    output = EvaluatorOutput(skills=[], fit=FitScores(experience=70, domain=0, education=0, notes=""), red_flags=[])
    weights = Weights(proficiency=0, experience=0.9, domain=0, education=0, github=0.10)
    legacy = Weights(proficiency=0, experience=1, domain=0, education=0)
    assert ranking.score(output, weights) == ranking.score(output, legacy) == 70
    assert ranking.score(output, weights, 100) == 73
    output.skills = [SkillProficiency(skill="Python", is_must_have=True, level=0, confidence=1, evidence=[], reasoning="")]
    assert ranking.score(output, weights, 100) == 60
    row = {"id": str(uuid4()), "resume_id": str(uuid4()), "skill_report": [], "fit_scores": output.fit.model_dump()}
    rows = [dict(row, github_json={"status": "ok"}, github_score=100),
            dict(row, id=str(uuid4()), github_json={"status": "rate_limited"}, github_score=100)]
    changed = []
    def execute(sql, params):
        if sql.startswith("SELECT ranking_weights"):
            return SimpleNamespace(fetchone=lambda: {"ranking_weights": None})
        if sql.startswith("SELECT s.*"):
            return SimpleNamespace(fetchall=lambda: rows)
        if sql.startswith("UPDATE screening_results"):
            changed.append(params)

    connection = SimpleNamespace(execute=execute)
    monkeypatch.setattr(ranking, "get_pool", lambda: SimpleNamespace(connection=lambda: nullcontext(connection)))
    ranked = ranking.rerank("test", weights)
    assert [result["final_score"] for result in ranked] == [73, 70] and len(changed) == 2
    monkeypatch.setattr(ranking.settings, "github_partial_weight_factor", .5)
    partial = dict(row, github_json={"status": "partial"}, github_score=100)
    stale = dict(row, github_json={"status": "stale"}, github_score=100)
    assert [result["final_score"] for result in ranking.rank_candidates([partial, stale], weights)] == [71.5, 71.5]


def test_api_exposes_stored_github_result(monkeypatch):
    key = str(uuid4())
    result = agent.score_profile(evidence([], {}), [], JD)
    row = {"id": str(uuid4()), "jd_id": key, "resume_id": str(uuid4()), "file_name": "resume.pdf",
           "status": "done", "error_message": None, "similarity": 0.5, "years_exp": None, "final_score": 60,
           "rank": 1, "skill_report": [], "fit_scores": None, "red_flags": [], "explanation": None,
           "recommendation": "maybe", "created_at": None, "github_score": result.github_score,
           "github_json": result.model_dump(mode="json")}
    monkeypatch.setattr(api, "fetch", lambda sql, args: [{"id": key}] if sql.startswith("SELECT id") else [row])
    endpoint = f"/screenings/{key}"
    test = TestClient(api.app)
    response = test.get(endpoint + "/results").json()[0]
    assert response["github_status"] == "ok" and response["github_json"] == row["github_json"]
    assert "Project match" in response["github_json"]["score_explanation"]


def test_ui_runs_ranking_only_and_renders_skill_and_github_evidence(monkeypatch):
    from streamlit.testing.v1 import AppTest

    job_id = str(uuid4())
    github = GithubResult(status="ok", github_score=80, summary="Checked a public repository.",
        project_matches=[GithubProjectMatch(resume_project="Project", match="verified", reason="Direct link", verified=True)],
        red_flags=["An example flag."]).model_dump()
    row = {"rank": 1, "resume_id": str(uuid4()), "file_name": "demo.pdf", "final_score": 84,
           "status": "done", "recommendation": "interview", "summary": "Example candidate.", "strengths": [],
           "gaps": [], "red_flags": [], "github_json": github, "github_score": 80, "github_status": "ok",
           "skill_report": [{"skill": "Python", "is_must_have": True, "level": 3, "confidence": .9,
                             "reasoning": "Owned a service.", "evidence": ["Built Python services."]}]}
    requests = []

    def respond(method, url, **kwargs):
        requests.append((method, url, kwargs))
        responses = {
            ("GET", "resumes"): [{"id": row["resume_id"], "file_name": row["file_name"]}],
            ("POST", "rank-only"): {"jd_id": job_id},
            ("GET", "status"): {"done": 1, "errors": 0, "filtered_out": 0, "total": 1, "state": "done"},
            ("GET", "github"): {"counts": {"ok": 1}, "scored": 1, "queue_depth": 0, "mixed_evidence": False},
            ("GET", "results"): [row],
            ("POST", "rerank"): [row],
        }
        return httpx.Response(200, json=responses[(method, url.rsplit("/", 1)[-1])], request=httpx.Request(method, url))

    monkeypatch.setattr("requests.request", respond)
    test = AppTest.from_file(Path(__file__).resolve().parents[1] / "app" / "ui.py", default_timeout=30).run()
    test.multiselect[0].set_value([row["resume_id"]])
    test.text_area[0].set_value("Python developer required").run()
    next(button for button in test.button if button.label == "Run screening").click().run()
    assert not test.exception
    assert test.text_input[0].value == job_id
    assert any(url.endswith("/screenings/rank-only") and data["json"]["resume_ids"] == [row["resume_id"]]
               for _, url, data in requests)
    assert any("Proficiency" in item.value.columns for item in test.dataframe)
    assert any("Example candidate." == item.value for item in test.text)
    assert any("GitHub score 80/100." in item.value for item in test.text)
    rerank = next(button for button in test.button if button.label == "Re-rank stored evaluations")
    rerank.click().run()
    assert not test.exception
    assert any(url.endswith("/rerank") and data["json"]["github"] == .10 for _, url, data in requests)
    assert all("/notifications" not in url for _, url, _ in requests)
