from datetime import date

from app.ingest.chunk import chunk
from app.ingest.metadata import contains_skill, experience_years
from app.ingest.preprocess import preprocess


def test_preprocess_preserves_skills_and_bullets():
    text = preprocess("Jane Doe\nTechnical Skills\n• C++, .NET, C#, Node.js\nExperience\nEngineer Jan 2021 – Present\n• Built APIs\n• Ran tests\nPage 1 of 1")
    assert "C++, .NET, C#, Node.js" in text
    assert "2021-01 - Present" in text
    assert "\n- Built APIs\n- Ran tests" in text
    assert "Page 1" not in text
    assert contains_skill(text, "C++") and contains_skill(text, ".NET")
    assert not contains_skill("JavaScript", "Java")


def test_experience_merges_overlap_and_ignores_education():
    text = "Education\n2010 - 2014\nExperience\nEngineer 2020-01 - 2022-01\nLead 2021-01 - 2023-01"
    assert experience_years(text, date(2025, 1, 1)) == 3
    assert experience_years("Education\n2010 - 2014") is None


def test_chunks_have_context_and_token_limit():
    text = "Skills\nPython SQL\nExperience\nEngineer 2020 - 2024\n" + "\n".join("- Built " + "useful " * 20 + "APIs." for _ in range(40))
    chunks = chunk(text, lambda value: len(value.split()))
    assert {item["section"] for item in chunks} == {"skills", "experience"}
    assert all(item["token_count"] <= 400 for item in chunks)
    assert all(":" in item["content"].splitlines()[0] for item in chunks)
