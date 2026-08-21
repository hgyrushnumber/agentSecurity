"""Application settings (pydantic-settings).

Environment variables use the AGENTSEC_ prefix, e.g. AGENTSEC_DATABASE_URL.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENTSEC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage
    database_url: str = "sqlite:///./agent.db"
    runs_dir: Path = Path("./runs")
    logs_dir: Path = Path("./logs")
    data_dir: Path = Path("./data")

    # Server
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # Worker
    worker_poll_interval: float = 2.0
    worker_max_parallel: int = 1


settings = Settings()
