import pytest

from app.models import EvaluatorOutput, FitScores, ParsedJD, SkillProficiency, Weights
from app.screening.anonymize import anonymize
from app.screening.ranking import rank_candidates, score
from app.screening.stage1_filter import should_filter
from app.screening.verifier import verify


def output(level=4, confidence=1, must=True, evidence=None):
    return EvaluatorOutput(skills=[SkillProficiency(skill="Python", is_must_have=must, level=level, confidence=confidence,
        evidence=evidence or [], reasoning="Test")], fit=FitScores(experience=100, domain=100, education=100, notes="Test"), red_flags=[])


def test_aliases_and_conservative_filter():
    jd = ParsedJD(title="Engineer", must_have=["JavaScript", "Kubernetes", "Python"], nice_to_have=[], responsibilities=[])
    assert not should_filter(jd, {"skills": ["js", "k8s"], "clean_text": ""})
    assert not should_filter(jd, {"skills": [], "clean_text": "Used JavaScript and Kubernetes in production"})
    assert should_filter(jd, {"skills": ["Excel"], "clean_text": "Excel analyst"})


def test_anonymize_removes_pii_and_preserves_dates_and_skills():
    original = "Jane Doe\nEmail: jane@example.com\nPhone: +91 9876543210\nAddress: 21 Main Street\nGender: female\nAge: 27\nSkills\nC++ .NET\nExperience\nEngineer 2021 - 2024\nlinkedin.com/in/jane"
    clean = anonymize(original)
    for value in ("Jane Doe", "jane@example.com", "9876543210", "21 Main", "female", "27", "linkedin.com"):
        assert value not in clean
    assert "2021 - 2024" in clean and "C++ .NET" in clean
    assert "Jane Doe" not in anonymize("Summary - Jane Doe: Built APIs", original)
    assert "2021-01 - 2024-01" in anonymize("Experience\nEngineer 2021-01 - 2024-01")
    assert "Candidate Example1" not in anonymize("Candidate Example1\nSkills\nPython")


def test_fabricated_quote_is_removed_and_confidence_halved():
    result = verify(output(4, 0.8, evidence=["Invented leadership claim"]), "Skills: Python")
    assert result.skills[0].confidence == 0.4
    assert result.skills[0].level == 1
    assert result.skills[0].evidence == []
    assert result.red_flags == ["Unverified evidence for Python"]
    result = verify(output(3, evidence=["Built   Python APIs"]), "Built Python\nAPIs")
    assert result.skills[0].level == 3


def test_ranking_formula_cap_and_no_nice_to_have():
    assert score(output()) == 100
    assert score(output(0)) == 45
    missing = output(0)
    assert score(missing, Weights(proficiency=0, experience=1, domain=0, education=0)) == 60
    assert score(output(2)) == pytest.approx(75.25)


def test_tiebreaks_and_rerank_without_llm():
    value = output(3).model_dump()
    rows = [dict(resume_id="a", skill_report=value["skills"], fit_scores=value["fit"], similarity=0.8, years_exp=3),
        dict(resume_id="b", skill_report=value["skills"], fit_scores=value["fit"], similarity=0.9, years_exp=1)]
    assert rank_candidates(rows)[0]["resume_id"] == "b"
    assert rank_candidates(rows, Weights(proficiency=0, experience=1, domain=0, education=0))[0]["final_score"] == 100
