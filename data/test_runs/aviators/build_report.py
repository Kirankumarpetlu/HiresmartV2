import json
import re
import shutil
from pathlib import Path

from app.db import fetch
from app.models import EvaluatorOutput, Weights
from app.screening.anonymize import anonymize
from app.screening.graph import read_run
from app.screening.ranking import score
from app.screening.verifier import normalized

root = Path("data/test_runs/aviators")
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
jd_name, run = next(iter(manifest["screenings"].items()))
key = run["jd_id"]
weights = Weights.model_validate(read_run(key)["request"]["weights"])
rows = json.loads((root / f"{key}_results.json").read_text(encoding="utf-8"))
ids = list(dict.fromkeys(item["id"] for item in manifest["uploads"] if "id" in item))
assert all(row["resume_id"] in ids for row in rows)
stats = fetch("""SELECT count(DISTINCT r.id) AS resumes,count(c.id) AS chunks,
    max(c.token_count) AS max_chunk_tokens FROM resumes r JOIN resume_chunks c ON c.resume_id=r.id
    WHERE r.id=ANY(%s::uuid[])""", (ids,))[0]
jd = fetch("SELECT title,must_have,nice_to_have,min_years,responsibilities FROM job_descriptions WHERE id=%s", (key,))[0]
checked_quotes = 0
unverified_flags = sum(flag.startswith("Unverified evidence") for row in rows for flag in row.get("red_flags", []))
summary_fallbacks = sum((row.get("summary") or "").startswith("Review the verified skill evidence and gaps below.") for row in rows)
for row in rows:
    if row["status"] != "done":
        continue
    output = EvaluatorOutput(skills=row["skill_report"], fit=row["fit_scores"], red_flags=row["red_flags"])
    assert abs(score(output, weights) - row["final_score"]) < 1e-5
    original = fetch("SELECT clean_text FROM resumes WHERE id=%s", (row["resume_id"],))[0]["clean_text"]
    chunks = fetch("SELECT content FROM resume_chunks WHERE resume_id=%s", (row["resume_id"],))
    evidence_text = normalized("\n".join(anonymize(chunk["content"], original) for chunk in chunks))
    for skill in output.skills:
        for quote in skill.evidence:
            assert normalized(quote) in evidence_text
            checked_quotes += 1
    assert 0 <= row["final_score"] <= 100
summary = {"jd_id": key, "jd_file": jd_name, "ingestion": stats, "status": run["status"],
    "evidence_quotes_checked": checked_quotes, "unverified_evidence_flags": unverified_flags, "summary_fallbacks": summary_fallbacks,
    "score_formula_checked": True, "selected_resumes_only": True, "human_accuracy_labels_available": False}
(root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
shutil.copyfile(root / f"{key}_results.csv", root / "results.csv")
shutil.copyfile(root / f"{key}_results.json", root / "results.json")
lines = ["# Aviators resumes: pipeline test", "", f"Job description: {jd_name}", f"Screening id: `{key}`", "",
    f"- Uploaded/indexed: {stats['resumes']} resumes, {stats['chunks']} chunks; maximum {stats['max_chunk_tokens']} tokens per chunk.",
    f"- Shortlisted: {len(rows)}; evaluated: {run['status']['done']}; errors: {run['status']['errors']}; filtered: {run['status']['filtered_out']}.",
    f"- Evidence quotes checked against anonymized stored resume chunks: {checked_quotes}.",
    f"- Unverified evidence flags: {unverified_flags}; these quotes were removed and confidence reduced. Template summary fallbacks: {summary_fallbacks}.",
    "- Stored scores match the code scoring formula. All candidates belong to the selected folder.", "",
    "## Parsed requirements", "", *[f"- {item}" for item in jd["must_have"]], "",
    "Degree/graduation eligibility remains in the job context for fit assessment. The language alternatives are one requirement; no preferred skills were invented.", "",
    "## Ranked results", "", "| Rank | Resume | Score | Recommendation | Flags |", "|---|---|---:|---|---:|"]
for row in rows:
    if row["status"] == "done":
        lines.append(f"| {row['rank']} | {row['file_name'].replace('|', ' ')} | {row['final_score']:.2f} | {row['recommendation']} | {len(row.get('red_flags', []))} |")
lines += ["", "## Findings and limits", "",
    "- Fixed test scoping so synthetic calibration resumes cannot enter this run.",
    "- The initial LLM extraction mixed degree eligibility into skill proficiency and inferred preferences from possible work areas. A regression-tested normalization step corrects these cases; the first attempt is retained separately in the manifest.",
    "- Constrained per-position skill names prevent omission or duplication of grouped requirements. Invalid numeric scales are rejected and retried with field-specific feedback, without logging resume text.",
    "- The configured shortlist evaluates 10 of the 25 indexed resumes. The other 15 received no detailed LLM evaluation in this test.",
    "- No human expected ranking was supplied; execution, evidence grounding, and score arithmetic were tested, while ranking accuracy and graduation eligibility require human review.",
    "- This tool recommends. A human makes the final decision.", ""]
(root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(summary, indent=2))
