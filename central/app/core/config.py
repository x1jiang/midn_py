import os
import json
from pathlib import Path
from typing import Optional

# DEBUG: show when this module is imported and path
print(f"[config] Importing central.app.core.config from {__file__}")

# Prefer python-dotenv for reliability (handles quoting, export, spaces)
try:
    from dotenv import load_dotenv  # type: ignore
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False

# Locate nearest .env up the tree
_env_path: Optional[Path] = None
for parent in Path(__file__).resolve().parents:
    candidate = parent / '.env'
    if candidate.exists():
        _env_path = candidate
        break

if _DOTENV_AVAILABLE and _env_path:
    load_dotenv(dotenv_path=_env_path, override=False)
elif _env_path:
    # Manual fallback parser (minimal)
    try:
        for line in _env_path.read_text().splitlines():
            line=line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k,v=line.split('=',1)
            os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass

_VALIDATED = False  # sentinel to avoid duplicate validation spam

class Settings:
    PROJECT_NAME: str = "Federated Imputation - Central Server"
    PROJECT_VERSION: str = "1.0.0"

    SECRET_KEY: str = os.getenv("SECRET_KEY", "a_very_secret_key")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Use database under repo_root/data by default (absolute path for robustness)
    _ROOT = Path(__file__).resolve().parents[3]
    _DEFAULT_DB = _ROOT / "data" / "central.db"
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{_DEFAULT_DB}")

    # Admin password for GUI/API protection
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "admin123")

    # Optional Gmail credentials (used by send_email_alert)
    GMAIL_USER: Optional[str] = os.getenv("GMAIL_USER")
    GMAIL_APP_PASSWORD: Optional[str] = os.getenv("GMAIL_APP_PASSWORD")
    CENTRAL_URL: str = os.getenv("central_url", "http://localhost:8000")

    # Supported algorithms configured here (upper-case names)
    _ALG = ["SIMI","SIMICE"]

    def gmail_ready(self) -> bool:
        return bool(self.GMAIL_USER and self.GMAIL_APP_PASSWORD)

    def validate(self):
        global _VALIDATED
        if _VALIDATED:
            return
        if self.gmail_ready():
            print(f"[config] Gmail credentials loaded for user {self.GMAIL_USER}")
        else:
            missing = []
            if not self.GMAIL_USER:
                missing.append('GMAIL_USER')
            if not self.GMAIL_APP_PASSWORD:
                missing.append('GMAIL_APP_PASSWORD')
            print(f"[config] Gmail credentials not fully configured; missing: {', '.join(missing)}. File loaded: {_env_path if _env_path else 'None found'}")
            if _env_path:
                try:
                    print(f"[config] Debug first 3 lines of .env:\n" + '\n'.join(_env_path.read_text().splitlines()[:3]))
                except Exception:
                    pass
        _VALIDATED = True

settings = Settings()
settings.validate()
