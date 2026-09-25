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

    # Independent, non-LLM grammar cross-check (grammar_check.py) - off by
    # default. Exists so the grading pipeline isn't the only thing checking
    # its own grammar-error list, but not yet wired to influence grading or
    # shown in the frontend - flip on to see LanguageTool's own match list
    # alongside the LLM's in the API response. Always via language_tool_
    # python's hosted Public API (see grammar_check.py's module docstring)
    # - never its default local-server mode, which would spin up a
    # resident Java process on Render the same way the OCR redesign
    # (see README's "Why a vision LLM" section) was specifically chosen to
    # avoid after DSG Compliance got OOM-killed on the free tier.
    enable_grammar_crosscheck: bool = False
    # Separate opt-in, only consulted when enable_grammar_crosscheck is
    # already True: switches grammar_check.py from the Public API to
    # language_tool_python's default local-server mode (a resident local
    # Java process). Included now, off by default, for a future deploy
    # target with enough RAM to actually run it - do NOT turn this on on
    # Render's free tier (512MB) or anywhere else without headroom well
    # above LanguageTool's own JVM footprint; that's the exact failure
    # mode this setting's sibling flag was added to avoid in the first
    # place. See grammar_check.py's module docstring before flipping this.
    grammar_crosscheck_local_server: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
