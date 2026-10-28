"""Application settings loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings. Model IDs live only here (INSTRUCCIONES.md section 5)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    model_api_key: str = Field(default="")
    meta_mock: bool = Field(default=False)
    meta_budget_usd: float = Field(default=60.0, gt=0)
    meta_spark_model: str = Field(default="muse-spark-1.3")
    meta_sam_model: str = Field(default="sam-3.1")
    meta_voice_model: str = Field(default="muse-voice-transcribe-1.0")
    numba_num_threads: int = Field(default=8, ge=1)

    meta_timeout_s: float = Field(default=60.0, gt=0)
    meta_max_retries: int = Field(default=3, ge=0)
    sam_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    cache_dir: str = Field(default=".cache/meta_client")
    runs_dir: str = Field(default="runs")

    @property
    def mock(self) -> bool:
        """True when mock mode is on (fixtures, no network, no real cost)."""
        return bool(self.meta_mock)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached settings."""
    return Settings()
