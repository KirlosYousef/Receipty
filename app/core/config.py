from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        secrets_dir=Path("/run/secrets") if Path("/run/secrets").is_dir() else None,
        extra="ignore",
    )

    openrouter_api_key: SecretStr = SecretStr("")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    model: str = "google/gemini-3.1-flash-lite"
    escalation_model: str = ""

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int | None = 42
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_cache_size: int = Field(default=256, ge=0)

    db_path: Path = Path("receipts.db")
    database_url: str | None = None
    cost_log_path: Path = Path("logs/cost.jsonl")
    trace_log_path: Path = Path("logs/traces.jsonl")
    max_image_bytes: int = 8 * 1024 * 1024
    max_attempts: int = Field(default=3, ge=1)
    total_deadline_seconds: float = Field(default=120.0, gt=0)
    retry_base_delay_seconds: float = Field(default=1.0, ge=0)
    request_timeout_seconds: float = Field(default=60.0, gt=0)
    max_agent_steps: int = Field(default=8, ge=1, le=20)
    max_agent_seconds: float = Field(default=60.0, gt=0)
    max_agent_tokens: int = Field(default=16000, ge=1)
    rate_limit_requests: int = Field(default=60, ge=0)
    rate_limit_window_seconds: float = Field(default=60.0, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
