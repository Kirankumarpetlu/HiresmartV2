import re

from app.ingest.metadata import DATE_RANGE, EMAIL_ADDRESS, candidate_names

CONTACT_LINE = re.compile(r"^\s*(?:name|candidate(?: name)?|address|location|gender|sex|age|date of birth|dob|marital status|nationality|photo(?:graph)?|pronouns)\s*[:|].*$", re.I | re.M)


def anonymize(text: str, original: str | None = None) -> str:
    names = candidate_names(original if original is not None else text)
    text = CONTACT_LINE.sub("[PERSONAL DETAILS REMOVED]", text)
    for name in names:
        text = re.sub(re.escape(name), "[CANDIDATE]", text, flags=re.I)
    text = re.sub(r"(?i)\bmailto:\S+", "[EMAIL]", text)
    text = EMAIL_ADDRESS.sub("[EMAIL]", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", text)
    text = re.sub(r"(?i)(?:https?://|www\.)\S+|\b(?:linkedin\.com|github\.com)/\S+", "[PROFILE]", text)
    text = re.sub(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)", lambda m: "[PHONE]" if len(re.sub(r"\D", "", m[0])) >= 10 and not DATE_RANGE.fullmatch(m[0]) else m[0], text)
    text = re.sub(r"\b\d{1,5}\s+[\w .'-]+\s(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr)\b[^\n]*", "[ADDRESS]", text, flags=re.I)
    text = re.sub(r"\b(?:male|female|married|unmarried|single|divorced|widowed|he/him|she/her|they/them)\b|\b\d{1,2}\s*(?:years? old|y/o)\b", "[PERSONAL]", text, flags=re.I)
    return text
