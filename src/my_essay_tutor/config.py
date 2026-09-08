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

    # Handwriting/print OCR (Hugging Face Inference API, TrOCR)
    hf_api_token: str = ""
    hf_trocr_model: str = "microsoft/trocr-large-handwritten"

    max_upload_mb: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
