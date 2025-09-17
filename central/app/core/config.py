import os
import json
from pathlib import Path

# Load environment variables from project root .env if present
_root_env = Path(__file__).resolve().parents[3] / '.env'
if _root_env.exists():  # best-effort load without extra dependency
    try:
        for line in _root_env.read_text().splitlines():
            if not line.strip() or line.strip().startswith('#'):
                continue
            if '=' not in line:
                continue
            k, v = line.split('=', 1)
            k = k.strip()
            v = v.strip()
            # Do not override if already in environment
            os.environ.setdefault(k, v)
    except Exception:
        pass

class Settings:
    PROJECT_NAME: str = "Federated Imputation - Central Server"
    PROJECT_VERSION: str = "1.0.0"

    SECRET_KEY: str = os.getenv("SECRET_KEY", "a_very_secret_key")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./central.db")

    # Admin password for GUI/API protection
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "admin123")

    # Optional Gmail credentials (used by send_email_alert)
    GMAIL_USER: str | None = os.getenv("GMAIL_USER")
    GMAIL_APP_PASSWORD: str | None = os.getenv("GMAIL_APP_PASSWORD")
    CENTRAL_URL: str = os.getenv("central_url", "http://localhost:8000")

    # Supported algorithms configured here (upper-case names)
    _ALG = ["SIMI","SIMICE"]

settings = Settings()
