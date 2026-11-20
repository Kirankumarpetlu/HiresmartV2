from collections.abc import Iterable

from psycopg.types.json import Jsonb

from app.config import settings
from app.db import get_pool
from app.models import EvaluatorOutput, SkillProficiency, Weights

LEVEL_WEIGHT = {0: 0.0, 1: 0.25, 2: 0.55, 3: 0.85, 4: 1.0}


def missing_must_have(skills: Iterable[SkillProficiency]) -> bool:
    return any(skill.is_must_have and skill.level == 0 for skill in skills)


def proficiency(output: EvaluatorOutput) -> float:
    must = [skill for skill in output.skills if skill.is_must_have]
    nice = [skill for skill in output.skills if not skill.is_must_have]

    def avg(items):
        return sum(LEVEL_WEIGHT[item.level] * item.confidence for item in items) / len(items) if items else 0

    return 100 * (0.75 * avg(must) + 0.25 * avg(nice)) if nice else 100 * avg(must)


def score(output: EvaluatorOutput, weights: Weights | None = None, github_score: int | None = None,
          github_weight_factor: float = 1) -> float:
    weights = weights or Weights()
    final = weights.proficiency * proficiency(output) + weights.experience * output.fit.experience + weights.domain * output.fit.domain + weights.education * output.fit.education
    final /= weights.proficiency + weights.experience + weights.domain + weights.education
    if github_score is not None:
        share = weights.github * github_weight_factor
        final = (1 - share) * final + share * github_score
    if missing_must_have(output.skills):
        final = min(final, 60)
    return round(final, 4)


def recommendation(final: float) -> str:
    return "interview" if final >= settings.interview_threshold else "maybe" if final >= settings.maybe_threshold else "not_a_fit"


def rank_candidates(rows: list[dict], weights: Weights | None = None) -> list[dict]:
    ranked = []
    for row in rows:
        output = EvaluatorOutput(skills=row["skill_report"], fit=row["fit_scores"], red_flags=row.get("red_flags") or [])
        github = row.get("github_json") or {}
        status = github.get("status")
        github_score = row.get("github_score") if status in ("ok", "partial", "stale") else None
        factor = settings.github_partial_weight_factor if status in ("partial", "stale") or github.get("stale") else 1
        final = score(output, weights, github_score, factor)
        ranked.append(dict(row, final_score=final, recommendation=recommendation(final),
            advanced_must=sum(skill.is_must_have and skill.level >= 3 for skill in output.skills), proficiency_score=proficiency(output)))
    ranked.sort(key=lambda row: (-row["final_score"], -row["advanced_must"], -(row.get("similarity") or 0), -(row.get("years_exp") or 0), str(row["resume_id"])))
    return [dict(row, rank=index) for index, row in enumerate(ranked, 1)]


def rerank_transaction(conn, jd_id: str, weights: Weights | None = None) -> list[dict]:
    job = conn.execute("SELECT ranking_weights FROM job_descriptions WHERE id=%s FOR UPDATE", (jd_id,)).fetchone()
    if not job:
        return []
    if weights is not None:
        conn.execute("UPDATE job_descriptions SET ranking_weights=%s WHERE id=%s", (Jsonb(weights.model_dump()), jd_id))
    else:
        weights = Weights.model_validate(job["ranking_weights"] or {})
    rows = conn.execute("""SELECT s.*,r.years_exp FROM screening_results s JOIN resumes r ON r.id=s.resume_id
        WHERE s.jd_id=%s AND s.status='done'""", (jd_id,)).fetchall()
    ranked = rank_candidates(rows, weights)
    previous = {row["id"]: (row.get("final_score"), row.get("rank"), row.get("recommendation")) for row in rows}
    for row in ranked:
        current = (row["final_score"], row["rank"], row["recommendation"])
        if current == previous[row["id"]]:
            continue
        conn.execute("UPDATE screening_results SET final_score=%s,rank=%s,recommendation=%s WHERE id=%s",
            (*current, row["id"]))
    return ranked


def rerank(jd_id: str, weights: Weights | None = None) -> list[dict]:
    with get_pool().connection() as conn:
        return rerank_transaction(conn, jd_id, weights)
