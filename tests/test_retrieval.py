import numpy as np

from app.models import ParsedJD
from app.screening import retrieval


def test_scoped_retrieval_passes_ids_and_merges_query_scores(monkeypatch):
    key = "00000000-0000-0000-0000-000000000001"
    calls = []

    def search(params):
        calls.append(params)
        return [{"resume_id": key, "similarity": 0.7 if len(calls) == 1 else 0.9}]

    monkeypatch.setattr(retrieval, "search_chunks", search)
    monkeypatch.setattr(retrieval, "fetch", lambda *args: [{"id": key, "skills": ["Python"]}])
    jd = ParsedJD(title="Engineer", must_have=["Python"], nice_to_have=[], responsibilities=[])
    result = retrieval.retrieve(jd, 1, vectors=np.zeros((2, 768)), resume_ids=[key])
    assert result[0]["id"] == key
    assert result[0]["similarity"] == 0.9
    assert all(call["resume_ids"] == [key] for call in calls)
