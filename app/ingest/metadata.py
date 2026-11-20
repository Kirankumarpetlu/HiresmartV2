import re
from collections.abc import Sequence
from datetime import date
from urllib.parse import unquote, urlsplit

from email_validator import EmailNotValidError, validate_email

from app.ingest.preprocess import heading

ALIASES = {"js": "javascript", "nodejs": "node.js", "k8s": "kubernetes", "postgres": "postgresql", "py": "python", "tf": "tensorflow", "dotnet": ".net"}
SKILLS = ("Python", "SQL", "Java", "JavaScript", "TypeScript", "C++", "C#", ".NET", "Node.js", "React", "Angular", "Vue", "Go", "Rust", "AWS", "Azure", "GCP", "Docker", "Kubernetes", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Spark", "Kafka", "Airflow", "dbt", "TensorFlow", "PyTorch", "pandas", "NumPy", "scikit-learn", "Git", "Linux", "Excel", "Tableau", "Power BI", "HTML", "CSS", "FastAPI", "Django", "Flask")
GITHUB_USER = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}")
GITHUB_REPO = re.compile(r"(?!\.{1,2}$)[A-Za-z0-9_.-]{1,100}")
GITHUB_TEXT = re.compile(r"(?<![\w:/@.%?#&=+\\-])(?:https?://)?github\.com/[^\s<>\[\]()\"']+", re.I)
EMAIL_ADDRESS = re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
HEADER_BREAK = re.compile(r"[\r\n]+[ \t]*(?:bcc|cc|to|from|subject|reply-to)\s*:", re.I)


def normalize_email(value: str) -> str | None:
    if not EMAIL_ADDRESS.fullmatch(value):
        return None
    try:
        return validate_email(value, check_deliverability=False, allow_smtputf8=False).normalized
    except EmailNotValidError:
        return None


def contact_email(text: str, hyperlinks: Sequence[str] = ()) -> str | None:
    if not HEADER_BREAK.search(text):
        for match in EMAIL_ADDRESS.finditer(text):
            start, end = match.span()
            before = text[start - 1] if start else ""
            after = text[end:end + 1]
            if before.isalnum() or before in ("@", ".", "_") or after.isalnum() or after in ("@", "_"):
                continue
            if address := normalize_email(match[0]):
                return address
    for link in hyperlinks:
        if link.casefold().startswith("mailto:"):
            target = unquote(link[7:])
            if not re.search(r"[\x00-\x1f\x7f]", target):
                if address := normalize_email(unquote(link[7:].partition("?")[0])):
                    return address
    return None


def candidate_names(original: str) -> list[str]:
    names = re.findall(r"^\s*(?:name|candidate name)\s*:\s*([^\n]+)", original, re.I | re.M)
    for line in original.splitlines()[:3]:
        line = line.strip()
        if heading(line):
            break
        if re.fullmatch(r"[^\W\d_][\w'.-]*(?:\s+[^\W\d_][\w'.-]*){1,4}", line) and not re.search(r"engineer|developer|manager|analyst|scientist|resume|curriculum|vitae", line, re.I):
            names.append(line)
            break
    return names


def contact_metadata(text: str, hyperlinks: Sequence[str] = (), *, raw_text: str | None = None) -> dict:
    names = candidate_names(text)
    return {"contact_email": contact_email(text if raw_text is None else raw_text, hyperlinks),
            "candidate_name": names[0].strip() if names else None}


def github_path(link: str) -> tuple[str, str | None] | None:
    if re.search(r"[\s\\\x00-\x1f]", link):
        return None
    try:
        url = urlsplit(link if "://" in link else "https://" + link)
    except ValueError:
        return None
    if url.scheme not in {"http", "https"} or url.netloc.casefold() != "github.com":
        return None
    parts = url.path.removesuffix("/").split("/")[1:]
    if not parts or not GITHUB_USER.fullmatch(parts[0]):
        return None
    if len(parts) == 1:
        return parts[0].lower(), None
    repo = parts[1].lower().removesuffix(".git")
    if GITHUB_REPO.fullmatch(repo) and not any(part in {"", ".", ".."} for part in parts[2:]):
        return parts[0].lower(), repo
    return None


def github_metadata(text: str, hyperlinks: Sequence[str] = ()) -> dict:
    visible = [match[0].rstrip(",;:!?") for match in GITHUB_TEXT.finditer(text)]
    paths = dict.fromkeys(
        path for link in [*hyperlinks, *(link.removesuffix(".") if not link.endswith("..") else link for link in visible)]
        if (path := github_path(link))
    )
    username = next((owner for owner, repo in paths if repo is None), None)
    username = username or next((owner for owner, _ in paths), None)
    return {"github_username": username, "github_repo_links": [
        f"https://github.com/{owner}/{repo}" for owner, repo in paths if repo
    ]}
DATE_RANGE = re.compile(r"\b((?:19|20)\d{2})(?:-(\d{2}))?\s*(?:-|–|—|to)\s*(?:(present|current|now)|((?:19|20)\d{2})(?:-(\d{2}))?)\b", re.I)


def canonical(skill: str) -> str:
    value = skill.strip().casefold()
    return ALIASES.get(value, value)


def contains_skill(text: str, skill: str) -> bool:
    variants = [canonical(skill)] + [alias for alias, value in ALIASES.items() if value == canonical(skill)]
    return any(re.search(r"(?<![\w+#.])" + re.escape(value) + r"(?![\w+#])", text, re.I) for value in variants)


def experience_years(text: str, today: date | None = None) -> float | None:
    today = today or date.today()
    section, found_section = "other", any(heading(line) == "experience" for line in text.splitlines())
    ranges = []
    for line in text.splitlines():
        section = heading(line) or section
        if found_section and section != "experience":
            continue
        if not found_section:
            continue  # Unknown sections cannot reliably separate education from employment.
        for match in DATE_RANGE.finditer(line):
            sy, sm, present, ey, em = match.groups()
            sm, em = int(sm or 1), int(em or 1)
            if not 1 <= sm <= 12 or not 1 <= em <= 12:
                continue
            start = int(sy) * 12 + sm
            end = today.year * 12 + today.month if present else int(ey) * 12 + em
            if start <= end <= today.year * 12 + today.month:
                ranges.append((start, end))
    if not ranges:
        return None
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return round(sum(end - start for start, end in merged) / 12, 2)


def metadata(text: str, hyperlinks: Sequence[str] = (), *, link_text: str | None = None) -> dict:
    section, education = "other", []
    for line in text.splitlines():
        new_section = heading(line)
        if new_section:
            section = new_section
        elif section == "education" and line:
            education.append(line)
    return {"skills": [skill for skill in SKILLS if contains_skill(text, skill)], "years_exp": experience_years(text),
            "education": "\n".join(education) or None,
            **github_metadata(text if link_text is None else link_text, hyperlinks),
            **contact_metadata(text, hyperlinks, raw_text=link_text)}
