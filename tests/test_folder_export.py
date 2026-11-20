import csv
import json

from scripts.test_folders import export_csv


def test_folder_results_export_preserves_unicode_and_status(tmp_path):
    path = tmp_path / "results.csv"
    github = {"status": "ok", "score_explanation": "GitHub score 80/100.\nProject match 100/100.",
              "project_matches": [{"resume_project": "Démo", "match": "verified"}]}
    export_csv(path, [{"rank": 1, "file_name": "Résumé.pdf", "status": "done", "final_score": 75,
                       "skill_report": [], "github_json": github, "github_status": "ok", "github_score": 80}])
    with path.open(encoding="utf-8-sig", newline="") as file:
        row = next(csv.DictReader(file))
    assert row["file_name"] == "Résumé.pdf"
    assert row["status"] == "done"
    assert row["final_score"] == "75"
    assert row["github_status"] == "ok" and row["github_score"] == "80"
    assert row["github_explanation"] == github["score_explanation"]
    assert json.loads(row["github_json"]) == github
