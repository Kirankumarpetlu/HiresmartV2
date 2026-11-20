"""Send one synthetic message to the loopback Mailpit catcher when run explicitly."""
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.notify.mailer import send_email
from app.notify.templates import render


def main() -> None:
    # Require the matching local web service before connecting to the fixed SMTP port.
    requests.get("http://127.0.0.1:8025/api/v1/messages", timeout=5).raise_for_status()
    settings.smtp_host, settings.smtp_port = "127.0.0.1", 1025
    settings.smtp_use_tls = settings.smtp_starttls = False
    settings.smtp_username = settings.recruiter_email = ""
    settings.email_from = "recruiting@example.org"
    subject, body_text, body_html = render("selected", "Sample Candidate", "Synthetic Role")
    send_email({"to_address": "check@example.org", "subject": subject, "body_text": body_text, "body_html": body_html})
    print("Synthetic message submitted to Mailpit at http://127.0.0.1:8025.")


if __name__ == "__main__":
    main()
