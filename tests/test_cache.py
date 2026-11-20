import numpy as np
from pgvector import Vector

from app import cache
from app.models import EvaluatorOutput, FitScores
from app.screening import graph, jd_parser, report
from app.services import embed_service


def test_embedding_cache_encodes_only_misses(monkeypatch):
    monkeypatch.setattr(cache.settings, "cache_enabled", True)
    existing_hash = cache.digest("same")
    monkeypatch.setattr(cache, "fetch", lambda *args: [{"text_hash": existing_hash, "embedding": Vector([1.0, 0.0])}])
    writes, encoded = [], []
    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def executemany(self, sql, params): writes.extend(params)
    class Connection:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def cursor(self): return Cursor()
    class Pool:
        def connection(self): return Connection()
    monkeypatch.setattr(cache, "get_pool", lambda: Pool())

    def encode(texts):
        encoded.extend(texts)
        return np.asarray([[0.0, 1.0] for _ in texts])

    result = cache.cached_embeddings(["same", "new", "same"], encode)
    assert encoded == ["new"] and len(writes) == 1
    np.testing.assert_array_equal(result, [[1, 0], [0, 1], [1, 0]])


def test_query_lru_batches_cold_values_and_reuses_them(monkeypatch):
    embed_service._query_cache.clear()
    calls = []
    monkeypatch.setattr(embed_service, "encode", lambda texts: calls.append(texts) or
        np.asarray([[index, 1.0] for index, _ in enumerate(texts)]))
    first = embed_service.encode_queries(["required", "responsibilities"])
    second = embed_service.encode_queries(["responsibilities", "required"])
    assert calls == [["required", "responsibilities"]]
    np.testing.assert_array_equal(second, first[::-1])


def test_jd_parse_cache_skips_llm_and_embedding(monkeypatch):
    parsed = {"title": "Engineer", "must_have": ["Python"], "nice_to_have": [],
        "min_years": None, "responsibilities": []}
    monkeypatch.setattr(jd_parser, "get_jd", lambda *args: {"parsed": parsed, "query_vectors": [[1.0, 0.0]]})
    monkeypatch.setattr(jd_parser, "generate_json", lambda *args: (_ for _ in ()).throw(AssertionError("LLM called")))
    monkeypatch.setattr(jd_parser, "jd_vectors", lambda *args: (_ for _ in ()).throw(AssertionError("embed called")))
    monkeypatch.setattr(jd_parser, "execute", lambda *args: None)
    _, result, vectors = jd_parser.parse_jd("Python engineer", "job")
    assert result.must_have == ["Python"]
    np.testing.assert_array_equal(vectors, [[1, 0]])


def test_verified_evaluation_cache_skips_evaluator(monkeypatch):
    verified = EvaluatorOutput(skills=[], fit=FitScores(experience=100, domain=50, education=50, notes=""), red_flags=[]).model_dump(mode="json")
    monkeypatch.setattr(graph, "should_filter", lambda *args: False)
    monkeypatch.setattr(graph, "get_evaluation", lambda *args: verified)
    monkeypatch.setattr(graph, "evaluate", lambda *args: (_ for _ in ()).throw(AssertionError("Evaluator called")))
    monkeypatch.setattr(graph, "execute", lambda *args: None)
    resume = {"id": "candidate", "skills": [], "years_exp": None, "clean_text": "Python"}
    result = graph.candidate_graph.invoke({"jd_id": "job", "resume_id": "candidate", "resume": resume,
        "jd_parsed": {"title": "Engineer", "must_have": [], "nice_to_have": [], "min_years": None, "responsibilities": []},
        "query_vectors": [[0.0]], "chunks": [{"id": "chunk", "section": "skills", "content": "Python"}]})
    assert result["cache_hit"] is True and result["candidate_results"][0]["status"] == "done"


def test_report_cache_skips_llm(monkeypatch):
    monkeypatch.setattr(report, "get_report", lambda *args: "Cached summary.")
    monkeypatch.setattr(report, "generate_json", lambda *args: (_ for _ in ()).throw(AssertionError("Report LLM called")))
    assert report.summarize({"skill_report": [], "red_flags": []}) == "Cached summary."
