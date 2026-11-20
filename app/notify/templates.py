from pathlib import Path
from typing import Literal

from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.config import settings

EmailType = Literal["selected", "rejected"]
TEMPLATE_VERSION = "1"
_templates = Environment(loader=FileSystemLoader(Path(__file__).with_name("templates")), autoescape=select_autoescape())


def header(value: str) -> str:
    return " ".join(value.split())


def render(email_type: EmailType, name: str | None, job_title: str | None) -> tuple[str, str, str]:
    title = header(job_title or "") or "the role"
    first_name = ((name or "").split(maxsplit=1) or ["Candidate"])[0]
    data = {"first_name": "Candidate" if "@" in first_name else first_name,
            "job_title": title, "company": settings.company_name, "recruiter": settings.recruiter_name,
            "recruiter_email": settings.recruiter_email, "next_step": settings.next_step_text,
            "selected": email_type == "selected"}
    prefix = "Next steps for your application" if data["selected"] else "Update on your application"
    return (f"{prefix} – {title}", _templates.get_template("candidate.txt").render(data),
            _templates.get_template("candidate.html").render(data))
