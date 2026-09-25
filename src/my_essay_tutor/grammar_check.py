"""Independent, non-LLM grammar cross-check against `language_tool_python`
(LanguageTool) - a second, rule-based opinion on grammar errors, since the
grading pipeline is otherwise one LLM call asked to both find every error
and score the essay against it, with nothing else checking whether that
error list is actually complete or correct.

Off by default (`Settings.enable_grammar_crosscheck`) - this exists as a
ready-to-enable capability, not something wired into grading yet. Flip the
setting on to have `grade_essay()`'s callers see LanguageTool's own match
list; nothing currently reconciles it against the LLM's `grammar_errors`
automatically.

Defaults to `language_tool_python.LanguageToolPublicAPI` - the hosted
public API - never the package's default local-server mode on its own,
since that spins up a resident local Java process. That is exactly the
kind of local resident ML/JVM load this project's OCR design deliberately
avoided (see README's "Why a vision LLM, not TrOCR/Kraken/PyLaia" section)
after a sibling project (DSG Compliance) was OOM-killed on Render's 512MB
free tier.

A second, separate flag - `Settings.grammar_crosscheck_local_server`,
also off by default - switches to the local-server constructor instead,
for a future deploy target with real memory headroom (not Render's free
tier). It only has any effect once `enable_grammar_crosscheck` is already
on, so there's no way to reach local-server mode by flipping one setting
alone; both have to be deliberately turned on together.
"""
from __future__ import annotations

from .config import get_settings

# LanguageTool's own locale codes, mapped from this project's two-letter
# language_code (see grading.py's LANGUAGE_NAMES for the sibling mapping).
_LANGUAGE_TOOL_CODES = {"de": "de-DE", "en": "en-US", "fr": "fr"}


def grammar_crosscheck(text: str, language_code: str) -> list[dict] | None:
    """Returns LanguageTool's raw match list, or None if the feature is
    disabled (the default) - a None return means "not run", not "no
    errors found", so callers shouldn't treat it as a clean bill of
    health."""
    settings = get_settings()
    if not settings.enable_grammar_crosscheck:
        return None

    import language_tool_python

    lt_language = _LANGUAGE_TOOL_CODES.get(language_code)
    if lt_language is None:
        raise ValueError(f"Unsupported language_code for grammar cross-check: {language_code}")

    # Local-server mode is a second, deliberate opt-in on top of the
    # feature already being enabled - see module docstring for why this
    # must never be the default path on a memory-constrained deploy.
    if settings.grammar_crosscheck_local_server:
        tool = language_tool_python.LanguageTool(lt_language)
    else:
        tool = language_tool_python.LanguageToolPublicAPI(lt_language)
    try:
        matches = tool.check(text)
    finally:
        tool.close()

    return [
        {
            "message": m.message,
            "context": m.context,
            "offset": m.offset,
            "length": m.errorLength,
            "replacements": m.replacements[:3],
            "rule_id": m.ruleId,
        }
        for m in matches
    ]
