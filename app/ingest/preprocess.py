import re
import unicodedata
from collections import Counter

HEADINGS = {
    "summary": "summary", "profile": "summary", "objective": "summary", "professional summary": "summary",
    "skills": "skills", "technical skills": "skills", "core competencies": "skills",
    "experience": "experience", "work experience": "experience", "professional experience": "experience",
    "work history": "experience", "employment": "experience",
    "projects": "project", "personal projects": "project", "education": "education",
    "certifications": "certifications",
}
MONTHS = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}


def heading(line: str) -> str | None:
    return HEADINGS.get(line.strip().rstrip(":").casefold())


def preprocess(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\u00a0", " ")
    text = re.sub(r"[•●▪◦]", "-", text)
    pages = text.split("\f")
    edges = Counter()
    for page in pages:
        lines = [line.strip() for line in page.splitlines() if line.strip()]
        edges.update(set(lines[:2] + lines[-2:]))
    repeated = {line for line, count in edges.items() if count > 1 and not heading(line)}
    cleaned = []
    for page in pages:
        lines = []
        for line in page.splitlines():
            line = re.sub(r"[ \t]+", " ", line).strip()
            if line in repeated or re.fullmatch(r"Page\s+\d+(?:\s+of\s+\d+)?", line, re.I):
                continue
            if lines and line and line[0].islower() and lines[-1] and not heading(lines[-1]):
                lines[-1] = lines[-1].rstrip("-") + ("" if lines[-1].endswith("-") else " ") + line
            else:
                lines.append(line)
        cleaned.append("\n".join(lines))
    text = "\n".join(cleaned)
    text = re.sub(r"\b(" + "|".join(MONTHS) + r")[a-z]*\.?\s+(\d{4})\b", lambda m: f"{m[2]}-{MONTHS[m[1].lower()]:02d}", text, flags=re.I)
    text = re.sub(r"(?<=\d)\s*[–—]\s*(?=\d|Present|Current)", " - ", text, flags=re.I)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
