from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    model: str = "z-ai/glm-5.3-flash"
    db_path: Path = Path("receipts.db")
    cost_log_path: Path = Path("logs/cost.jsonl")
    max_image_bytes: int = 8 * 1024 * 1024
    max_attempts: int = Field(default=3, ge=1)
    total_deadline_seconds: float = Field(default=120.0, gt=0)
    retry_base_delay_seconds: float = Field(default=1.0, ge=0)
    request_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
