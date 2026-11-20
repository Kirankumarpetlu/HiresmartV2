"""Create synthetic fixtures for human review; never call synthetic labels human judgments."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docx import Document

from app.config import settings


def create() -> Path:
    root = settings.data_dir / "labeled"
    root.mkdir(parents=True, exist_ok=True)
    output = root / "labels.json"
    if output.exists():
        print("Existing labels preserved:", output)
        return output
    resumes, jobs = [], []
    domains = [("Backend engineer", ["Python", "SQL", "FastAPI", "PostgreSQL"]),
        ("Data engineer", ["Python", "SQL", "Spark", "Airflow"]),
        ("Frontend engineer", ["JavaScript", "TypeScript", "React", "CSS"])]
    for domain_index, (title, skills) in enumerate(domains):
        relevant = []
        for level in (4, 3, 2, 1):
            number = domain_index * 4 + (5 - level)
            name = f"candidate_{number:02d}.docx"
            relevant.append(name)
            document = Document()
            lines = [f"Candidate Example{number}", "candidate@example.invalid", "Skills", ", ".join(skills)]
            if level > 1:
                start = {4: 2017, 3: 2020, 2: 2024}[level]
                lines += ["Experience", f"{title} at Example Systems {start}-01 - Present"]
                verb = {4: "Architected and led a team delivering large-scale", 3: "Owned delivery and reduced latency by 40 percent in", 2: "Implemented and tested production"}[level]
                lines += [f"- {verb} {skill} services and projects." for skill in skills]
            else:
                lines += ["Summary", "Entry-level applicant seeking a first engineering role."]
            lines += ["Education", "BSc Computer Science"]
            for line in lines:
                document.add_paragraph(line)
            path = root / name
            if not path.exists():
                document.save(path)
            resumes.append({"file": name, "expected_levels": {skill: level for skill in skills}})
        jobs.append({"name": title, "jd_text": f"{title}. Required: {', '.join(skills)}. Minimum 1 year of professional experience. Build reliable production services. BSc preferred.",
            "parsed": {"title": title, "must_have": skills, "nice_to_have": [], "min_years": 1, "responsibilities": ["Build reliable production services"]},
            "relevant": relevant, "expected_ranking": relevant})
    output.write_text(json.dumps({"human_reviewed": False, "label_origin": "synthetic provisional expectations; requires human review", "resumes": resumes, "jobs": jobs}, indent=2), encoding="utf-8")
    print("Created 12 synthetic resumes and 3 JDs. Review labels before setting human_reviewed=true.")
    return output


if __name__ == "__main__":
    create()
