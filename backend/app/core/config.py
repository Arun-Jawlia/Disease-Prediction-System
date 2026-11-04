from pathlib import Path

from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


# ------------------------------------------------------------
# Project Paths
# ------------------------------------------------------------

BACKEND_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)

PROJECT_ROOT = (
    BACKEND_DIR.parent
)

MODELS_DIR = (
    PROJECT_ROOT / "models"
)


# ------------------------------------------------------------
# Application Settings
# ------------------------------------------------------------

class Settings(BaseSettings):

    app_name: str = (
        "Disease Prediction System"
    )

    app_version: str = "1.0.0"

    app_environment: str = (
        "development"
    )

    debug: bool = True

    secret_key: str = (
        "change-this-secret-key"
    )

    model_base_path: str = str(
        MODELS_DIR
    )

    cors_origins: str = (
        "http://localhost:5173"
    )

    model_checksum_enabled: bool = True

    model_dir: str = str(
        MODELS_DIR
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def cors_origin_list(self):
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]


settings = Settings()