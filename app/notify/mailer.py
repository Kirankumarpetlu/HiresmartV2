import smtplib
import ssl
from contextlib import suppress
from email.message import EmailMessage

from app.config import settings
from app.ingest.metadata import normalize_email
from app.notify.templates import header


class DeliveryError(Exception):
    def __init__(self, reason: str, *, transient: bool = False, uncertain: bool = False):
        super().__init__(reason)
        self.transient = transient
        self.uncertain = uncertain


def send_email(row: dict) -> None:
    sender = normalize_email(settings.email_from)
    recipient = normalize_email(row["to_address"] or "")
    reply_to = normalize_email(settings.recruiter_email) if settings.recruiter_email else None
    if not sender or not recipient or settings.recruiter_email and not reply_to:
        raise DeliveryError("Invalid sender, recipient or reply-to address")
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = header(row["subject"])
    if reply_to:
        message["Reply-To"] = reply_to
    message.set_content(row["body_text"])
    message.add_alternative(row["body_html"], subtype="html")

    client = None
    submitting = False
    try:
        if settings.smtp_use_tls:
            client = smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, timeout=15, context=ssl.create_default_context())
        else:
            client = smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15)
            if settings.smtp_starttls:
                client.starttls(context=ssl.create_default_context())
        if settings.smtp_username:
            client.login(settings.smtp_username, settings.smtp_password.get_secret_value())
        submitting = True
        client.send_message(message)
    except smtplib.SMTPRecipientsRefused as exc:
        code = next(iter(exc.recipients.values()))[0]
        raise DeliveryError(f"SMTP refused recipient ({code})", transient=400 <= code < 500) from None
    except smtplib.SMTPResponseException as exc:
        code = exc.smtp_code
        raise DeliveryError(f"SMTP returned {code}", transient=400 <= code < 500,
                            uncertain=submitting and not 400 <= code < 600) from None
    except ssl.SSLCertVerificationError:
        raise DeliveryError("SMTP certificate validation failed") from None
    except (OSError, smtplib.SMTPException) as exc:
        raise DeliveryError(f"SMTP connection failed ({type(exc).__name__})",
                            transient=not submitting and not isinstance(exc, smtplib.SMTPNotSupportedError),
                            uncertain=submitting) from None
    finally:
        if client is not None:
            with suppress(OSError):
                client.close()
