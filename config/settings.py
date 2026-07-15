"""Configuration settings for the Entangled Retrievals API."""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App metadata
    app_name: str = "Entangled Retrievals"
    app_version: str = "1.0.0"
    env: str = os.getenv("ENV", "development")

    # Server configuration
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    reload: bool = os.getenv("RELOAD", "true").lower() == "true"

    # Logging configuration
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # CORS configuration
    cors_origins: List[str] = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8000",
    ]

    class Config:
        """Pydantic config."""
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
