"""grammar_check.py's off-by-default gate. Does not call the real
LanguageTool Public API (that's an external network dependency, exercised
manually - see README) - just confirms the feature is a true no-op unless
explicitly enabled, and that enabling it never routes through
language_tool_python's default local-server constructor."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from my_essay_tutor.config import get_settings
from my_essay_tutor.grammar_check import grammar_crosscheck


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_disabled_by_default_returns_none(monkeypatch):
    monkeypatch.delenv("ENABLE_GRAMMAR_CROSSCHECK", raising=False)
    assert grammar_crosscheck("Some text.", "en") is None


def test_disabled_never_imports_language_tool_python(monkeypatch):
    """If this accidentally imported language_tool_python (and therefore
    could accidentally construct a local server) even while disabled,
    patching it to raise would catch it."""
    monkeypatch.delenv("ENABLE_GRAMMAR_CROSSCHECK", raising=False)
    with patch.dict("sys.modules", {"language_tool_python": None}):
        assert grammar_crosscheck("Some text.", "en") is None


def test_enabled_uses_public_api_not_local_server(monkeypatch):
    monkeypatch.setenv("ENABLE_GRAMMAR_CROSSCHECK", "true")
    with patch("language_tool_python.LanguageToolPublicAPI") as mock_public, \
         patch("language_tool_python.LanguageTool") as mock_local:
        mock_public.return_value.check.return_value = []
        grammar_crosscheck("Some text.", "en")
        mock_public.assert_called_once_with("en-US")
        mock_local.assert_not_called()


def test_unsupported_language_raises(monkeypatch):
    monkeypatch.setenv("ENABLE_GRAMMAR_CROSSCHECK", "true")
    with pytest.raises(ValueError):
        grammar_crosscheck("Some text.", "tr")


def test_local_server_flag_alone_does_not_enable_local_mode(monkeypatch):
    """grammar_crosscheck_local_server must be inert while the feature
    itself is off - there's no path to local-server mode via this flag
    alone."""
    monkeypatch.delenv("ENABLE_GRAMMAR_CROSSCHECK", raising=False)
    monkeypatch.setenv("GRAMMAR_CROSSCHECK_LOCAL_SERVER", "true")
    assert grammar_crosscheck("Some text.", "en") is None


def test_both_flags_on_uses_local_server_not_public_api(monkeypatch):
    monkeypatch.setenv("ENABLE_GRAMMAR_CROSSCHECK", "true")
    monkeypatch.setenv("GRAMMAR_CROSSCHECK_LOCAL_SERVER", "true")
    with patch("language_tool_python.LanguageToolPublicAPI") as mock_public, \
         patch("language_tool_python.LanguageTool") as mock_local:
        mock_local.return_value.check.return_value = []
        grammar_crosscheck("Some text.", "en")
        mock_local.assert_called_once_with("en-US")
        mock_public.assert_not_called()
