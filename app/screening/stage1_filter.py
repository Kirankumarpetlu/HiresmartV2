from app.ingest.metadata import SKILLS, canonical, contains_skill
from app.models import ParsedJD


def should_filter(jd: ParsedJD, resume: dict) -> bool:
    if not jd.must_have:
        return False
    known = {canonical(skill) for skill in resume.get("skills", [])}
    supported = {canonical(skill) for skill in SKILLS}
    # Dictionary omissions are not misses: check the full text before discarding anyone.
    missing = sum(canonical(skill) in supported and canonical(skill) not in known and not contains_skill(resume.get("clean_text", ""), skill) for skill in jd.must_have)
    return missing > len(jd.must_have) / 2
