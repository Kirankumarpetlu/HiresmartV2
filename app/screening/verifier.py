import re
from datetime import date

from app.ingest.metadata import contains_skill
from app.models import EvaluatorOutput


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def verify(output: EvaluatorOutput, text: str, chunks: list[dict] | None = None) -> EvaluatorOutput:
    result = output.model_copy(deep=True)
    haystack = normalized(text)
    for skill in result.skills:
        verified = [quote for quote in skill.evidence if normalized(quote) and normalized(quote) in haystack]
        if len(verified) != len(skill.evidence) or (skill.level > 0 and not verified):
            skill.confidence *= 0.5
            result.red_flags.append(f"Unverified evidence for {skill.skill}")
        skill.evidence = verified
        if skill.level > 0 and not verified:
            skill.level = 1 if contains_skill(text, skill.skill) else 0
        if verified and chunks:
            supported = [item for item in chunks if item["section"] != "skills" and any(normalized(quote) in normalized(item["content"]) for quote in verified)]
            if not supported:
                skill.level = min(skill.level, 1)
        # The evaluator returns the pre-recency level so this happens exactly once.
        if skill.last_used_year and date.today().year - skill.last_used_year >= 5:
            skill.level = max(0, skill.level - 1)
    result.red_flags = list(dict.fromkeys(result.red_flags))
    return result
