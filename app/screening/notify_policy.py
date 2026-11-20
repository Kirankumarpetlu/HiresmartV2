from app.config import settings
from app.models import SkillProficiency
from app.screening.ranking import missing_must_have


def decision(row: dict) -> dict:
    github = row.get("github_json") or {}
    github_status = github.get("status", "not_provided")
    github_score = github.get("github_score", row.get("github_score"))
    rank = row.get("rank")
    final_score = row.get("final_score")
    recommendation = row.get("recommendation")
    checks = {
        "evaluated": row["status"] == "done",
        "interview_threshold": recommendation == "interview" and final_score is not None
        and final_score >= settings.interview_threshold,
        "must_haves_met": not missing_must_have(SkillProficiency.model_validate(skill) for skill in row.get("skill_report") or []),
        "github_threshold": github_status != "ok" or github_score is not None and github_score >= settings.github_min_score,
        "rank_limit": rank is not None and (settings.email_max_selected is None or rank <= settings.email_max_selected),
    }
    return {"proposed": all(checks.values()), "checks": checks, "rank": rank, "final_score": final_score,
            "recommendation": recommendation, "github_status": github_status, "github_score": github_score}
