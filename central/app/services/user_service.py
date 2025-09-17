from sqlalchemy.orm import Session
import hashlib
import time
from datetime import timedelta, datetime, timezone

from .. import models, schemas
from ..core.security import get_password_hash, create_access_token
from ..core.config import settings

# Email dependencies
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import smtplib

def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()


def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()


def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = get_password_hash(user.password)
    site_id = hashlib.sha1(str(time.time()).encode()).hexdigest()[:8]
    db_user = models.User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        institution=user.institution,
        site_id=site_id
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user



def send_email_alert(from_email_addr: str, to_email_addr: str, message_text: str):
    """Send an approval email using Gmail (app password) with approve.txt attachment.

    If Gmail creds missing, this becomes a no-op (logs and returns) instead of raising.
    """
    import os

    if not settings.gmail_ready():
        print("[email] Gmail credentials missing; skipping Gmail send.")
        return

    gmail_user = os.getenv("GMAIL_USER", settings.GMAIL_USER or "")
    gmail_pass = os.getenv("GMAIL_APP_PASSWORD", settings.GMAIL_APP_PASSWORD or "")

    msg = MIMEMultipart()
    display_from = from_email_addr or gmail_user
    msg['From'] = display_from
    msg['To'] = to_email_addr
    msg['Subject'] = "MIDistNet Federated Learning Remote Site Registration Approved - Do not reply"

    body = (
        "Your site has been approved. See attached approve.txt for details.\n\n"
        "If you did not request this registration, please ignore this email."
    )
    msg.attach(MIMEText(body, 'plain'))

    part = MIMEBase('application', 'octet-stream')
    part.set_payload(message_text.encode('utf-8'))
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment', filename='approve.txt')
    msg.attach(part)

    try:
        with smtplib.SMTP('smtp.gmail.com', 587, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(gmail_user, gmail_pass)
            server.sendmail(msg['From'], to_email_addr, msg.as_string())
        print(f"[email] Gmail message sent to {to_email_addr}")
    except Exception as e:
        print(f"[email] Failed Gmail send: {e}")
 

def send_email_alert_ut_smtp(from_email_addr: str, to_email_addr: str, message_text: str):
    """
    Send an approval email with details attached as approve.txt via test SMTP.
    NOTE: Uses unencrypted SMTP on 129.106.31.45:7725 per request.
    """
    msg = MIMEMultipart()
    msg['From'] = from_email_addr
    msg['To'] = to_email_addr
    msg['Subject'] = "MIDistNet Federated Learning Remote Site Registration Approved - Do not reply"

    # Minimal body
    msg.attach(MIMEText("Your site has been approved. See attached approve.txt for details.", 'plain'))

    # Attachment approve.txt with the full message_text
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(message_text.encode('utf-8'))
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment', filename='approve.txt')
    msg.attach(part)

    server = smtplib.SMTP('129.106.31.45', 7725, timeout=5)
    try:
        server.sendmail(msg['From'], to_email_addr, msg.as_string())
        print(f"Email sent to {to_email_addr}")
    finally:
        server.quit()



def approve_user(db: Session, user_id: int, central_url: str = "http://127.0.0.1:8000", expires_days: int = 30):
    """Approve a user and email them credentials.

    central_url logic:
      - If caller passes a value different from the legacy default, use it.
      - Else attempt to use environment variable / settings.CENTRAL_URL.
      - If that still points to localhost while running in a hosted environment (detected via common env vars),
        try to synthesize an https URL based on X_SERVICE_URL or similar future variable (extensible hook).
    """
    import os

    db_user = get_user(db, user_id)
    if not db_user:
        return None
    if not db_user.is_approved:
        db_user.is_approved = True
    # Issue a JWT where sub == site_id
    token = create_access_token({"sub": db_user.site_id}, expires_delta=timedelta(days=expires_days))
    db_user.jwt_token = token
    db_user.jwt_expires_at = datetime.utcnow().replace(tzinfo=timezone.utc) + timedelta(days=expires_days)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    # Resolve central_url dynamically if default value still in place
    if central_url == "http://127.0.0.1:8000":
        # Prefer settings.CENTRAL_URL if provided and not localhost
        cfg_url = getattr(settings, "CENTRAL_URL", None)
        if cfg_url and cfg_url not in {"http://127.0.0.1:8000", "http://localhost:8000", "ws://127.0.0.1:8000", "ws://localhost:8000"}:
            central_url = cfg_url.replace("ws://", "http://").replace("wss://", "https://")
        else:
            # Check for a platform-provided base URL (placeholder for future expansion)
            inferred = os.getenv("SERVICE_PUBLIC_URL") or os.getenv("PUBLIC_BASE_URL")
            if inferred:
                central_url = inferred.rstrip('/')
            else:
                # If running under Cloud Run / App Engine, user should set CENTRAL_URL env; keep fallback.
                pass

    exp_at = db_user.jwt_expires_at.strftime("%Y-%m-%d %H:%M:%S %Z")
    message = (
        f"Hello {db_user.username},\n\n"
        f"Your registration has been approved.\n\n"
        f"Central URL: {central_url}\n"
        f"Your site_id: {db_user.site_id}\n"
        f"Your JWT token: {db_user.jwt_token}\n"
        f"Token expires at: {exp_at} (in {expires_days} days)\n\n"
        f"Configure your remote client with the Central URL, site_id, and JWT token.\n"
        f"This is an automated message; do not reply.\n"
    )

    # From and To
    from_addr = "no-reply@localhost"
    to_addr = db_user.email
    try:
        send_email_alert(from_addr, to_addr, message)
    except Exception as e:
        # Log error; in real system use a logger
        print(f"Email send failed: {e}")
    return db_user
