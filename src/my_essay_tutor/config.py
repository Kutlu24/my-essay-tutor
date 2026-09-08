from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Grading LLM (grammar + CEFR scoring)
    grading_provider: str = "glm"  # "glm" | "gemini" | "anthropic"
    glm_api_key: str = ""
    glm_model: str = "glm-4.5-flash"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-5"

    # Handwriting/print text extraction (vision LLM, not a local HTR model --
    # see ocr.py for why)
    ocr_provider: str = "gemini"  # "glm" | "gemini" -- glm-4.5v needs a paid GLM
    # resource package (confirmed: text chat completions work on a free GLM key,
    # vision returns "Insufficient balance", error 1113); Gemini's vision models
    # are covered by the same free-tier key already used elsewhere this session.
    ocr_model: str = "glm-4.5v"

    max_upload_mb: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
