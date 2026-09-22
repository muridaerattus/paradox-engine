from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    openrouter_api_key: str | None = None
    local_llm_api_key: str | None = None
    local_llm_api_base: str | None = None

    alchemy_model: str = "meta-llama/llama-3.3-70b-instruct"
    classpect_mode: Literal["classifier", "llm"] = "classifier"
    classpect_classifier_model: str = "jev-latest"
    classpect_model: str = "z-ai/glm-5.3-flash"
    fraymotif_model: str = "z-ai/glm-5.3"

    class_quiz_filename: Path = Path("class_quiz.json")
    aspect_quiz_filename: Path = Path("aspect_quiz.json")
    prompts_directory: Path = Path("prompts")
    database_url: str = "sqlite+aiosqlite:///./paradox.db"
    database_echo: bool = False

    api_root_path: str = "/api"
    enable_docs: bool = False
    cors_origins: str = Field(
        default="http://localhost:3000",
        description="Comma-separated browser origins allowed by CORS.",
    )

    @property
    def parsed_cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
