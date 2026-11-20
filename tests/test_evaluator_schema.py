import pytest
from pydantic import ValidationError

from app.models import ParsedJD
from app.screening import evaluator
from app.screening.evaluator import evaluation_schema


def skill(name):
    return {"skill": name, "is_must_have": True, "level": 2, "evidence": ["Built Python APIs"], "reasoning": "Production work", "confidence": 0.8}


def test_evaluator_schema_preserves_alternatives_and_requires_complete_unique_skills():
    group = "Any of: Java | Python | C++"
    schema = evaluation_schema([group, "Collaboration"])
    fit = {"experience": 100, "domain": 50, "education": 100, "notes": "Test"}
    payload = {"skills": [skill(group), skill("Collaboration")], "fit": fit, "red_flags": []}
    assert schema.model_validate(payload).skills[0].skill == group
    assert schema.model_json_schema()["properties"]["skills"]["maxItems"] == 2
    assert len(schema.model_json_schema()["properties"]["skills"]["prefixItems"]) == 2
    with pytest.raises(ValidationError):
        schema.model_validate(dict(payload, skills=[skill("Python"), skill("Collaboration")]))
    with pytest.raises(ValidationError):
        schema.model_validate(dict(payload, skills=[skill(group), skill(group)]))
    with pytest.raises(ValidationError):
        schema.model_validate(dict(payload, skills=[skill(group)]))


@pytest.mark.parametrize("split", [False, True])
def test_compact_schema_restores_skill_names_and_keeps_validation(split):
    names = ["Any of: Java | Python | C++", "Collaboration"]
    schema = evaluation_schema(names, split)
    payload = {"skills": [{key: value for key, value in skill(name).items()
        if key not in ("skill", "is_must_have")} for name in names], "red_flags": []}
    if not split:
        payload["fit"] = {"experience": 0, "domain": 50, "education": 100, "notes": "Test"}
    output = schema.model_validate(payload)
    assert [item.skill for item in output.skills] == names
    definitions = schema.model_json_schema()["$defs"]
    for index in range(2):
        properties = definitions[f"RequestedSkill{index}"]["properties"]
        assert "skill" not in properties and "is_must_have" not in properties
        assert {"evidence", "reasoning", "level", "confidence"} <= properties.keys()
    payload["skills"][0]["confidence"] = 80
    with pytest.raises(ValidationError):
        schema.model_validate(payload)


@pytest.mark.parametrize("split", [False, True])
def test_compact_evaluation_preserves_public_fields_and_code_scoring(monkeypatch, split):
    jd = ParsedJD(title="Engineer", must_have=["Python"], nice_to_have=["SQL"], min_years=4, responsibilities=[])
    monkeypatch.setattr(evaluator.settings, "evaluator_split_calls", split)

    def generate(messages, schema):
        assert "1. Must-have: Python\n2. Nice-to-have: SQL" in messages[0]["content"]
        assert messages[1]["content"].startswith("Skills: Python\nEvidence:")
        if schema is evaluator.FitScores:
            return schema(experience=0, domain=75, education=100, notes="Test")
        payload = {"skills": [{key: value for key, value in skill(name).items()
            if key not in ("skill", "is_must_have")} for name in ("Python", "SQL")], "red_flags": []}
        if not split:
            payload["fit"] = {"experience": 0, "domain": 75, "education": 100, "notes": "Test"}
        return schema.model_validate(payload)

    monkeypatch.setattr(evaluator, "generate_json", generate)
    output = evaluator.evaluate(jd, [{"content": "Built Python APIs"}], ["Python"], 2)
    output.model_dump(mode="json", warnings="error")
    assert [(item.skill, item.is_must_have) for item in output.skills] == [("Python", True), ("SQL", False)]
    assert output.fit.experience == 50
    assert output.skills[0].evidence == ["Built Python APIs"]
