import json

from app.screening import report as module
from app.screening.report import report_sections
from app.models import Summary


def test_report_uses_verified_stored_evidence():
    row = {"skill_report": [{"skill": "Python", "level": 3, "is_must_have": True, "evidence": ["Built Python APIs"]},
        {"skill": "SQL", "level": 0, "is_must_have": True, "evidence": []}], "red_flags": ["Review dates"], "recommendation": "maybe"}
    report = report_sections(row)
    assert report["strengths"][0]["evidence"] == ["Built Python APIs"]
    assert report["gaps"] == ["SQL"]
    assert report["recommendation"] == "maybe"


def test_report_node_saves_fallback_and_continues(monkeypatch):
    rows = [{"id": "failed"}, {"id": "good"}]
    saved = []
    monkeypatch.setattr(module, "fetch", lambda *args: rows)
    monkeypatch.setattr(module, "execute", lambda sql, params: saved.append(params))

    def summarize(row):
        if row["id"] == "failed":
            raise RuntimeError("Model unavailable")
        return "Verified summary."

    monkeypatch.setattr(module, "summarize", summarize)
    assert module.report_node({"jd_id": "test"}) == {}
    assert saved == [
        ("Review the verified skill evidence and gaps below. A human must make the final decision.", "failed"),
        ("Verified summary.", "good"),
    ]


def test_summary_prompt_includes_each_quote_once_without_changing_report(monkeypatch):
    row = {"skill_report": [{"skill": "Python", "level": 3, "is_must_have": True,
        "evidence": ["Owned Python API delivery"]}], "red_flags": []}
    calls = []
    monkeypatch.setattr(module, "get_report", lambda *args: None)
    monkeypatch.setattr(module, "put_report", lambda *args: None)
    monkeypatch.setattr(module, "generate_json", lambda messages, *args:
        calls.append(messages) or Summary(summary="Built Python APIs. Evidence supports production work."))
    module.summarize(row)
    assert calls[0][1]["content"].count("Owned Python API delivery") == 1
    facts = json.loads(calls[0][1]["content"])
    assert facts["strengths"] == [{"skill": "Python", "level": 3}]
    assert facts["evidence"][0]["quotes"] == ["Owned Python API delivery"]
    assert module.report_sections(row)["strengths"][0]["evidence"] == ["Owned Python API delivery"]
