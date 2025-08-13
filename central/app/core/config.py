import os
import json

class Settings:
    PROJECT_NAME: str = "Federated Imputation - Central Server"
    PROJECT_VERSION: str = "1.0.0"

    SECRET_KEY: str = os.getenv("SECRET_KEY", "a_very_secret_key")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./central.db")

    # Admin password for GUI/API protection
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "admin123")

    # Supported algorithms configured here (upper-case names)
    _ALG = ["SIMI","SIMICE"]

settings = Settings()
