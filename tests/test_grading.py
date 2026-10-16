"""_GradingResponse validation - the schema instructor validates each
provider's structured output against (see grading.py's module docstring
for why this replaced hand-rolled JSON parsing)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from my_essay_tutor.grading import _GradingResponse

_VALID_CRITERIA = {
    "vocabulary": {"score": 8, "comment": "ok"},
    "coherence": {"score": 8, "comment": "ok"},
    "grammar": {"score": 8, "comment": "ok"},
    "content_relevance": {"score": 8, "comment": "ok"},
}


def _payload(**overrides) -> dict:
    base = {
        "grammar_errors": [],
        "achieved_level": "B1",
        "level_confidence": "at",
        "criteria": _VALID_CRITERIA,
        "strengths": ["clear structure"],
        "weaknesses": ["a few tense errors"],
        "overall_feedback": "Good effort overall.",
    }
    base.update(overrides)
    return base


def test_valid_payload_parses():
    result = _GradingResponse.model_validate(_payload())
    assert result.achieved_level == "B1"
    assert result.criteria.vocabulary.score == 8


def test_rejects_unknown_achieved_level():
    with pytest.raises(ValidationError):
        _GradingResponse.model_validate(_payload(achieved_level="Z9"))


def test_rejects_unknown_level_confidence():
    with pytest.raises(ValidationError):
        _GradingResponse.model_validate(_payload(level_confidence="somewhat"))


def test_rejects_criteria_as_a_string():
    # The real failure mode hit against GLM's tool-calling mode before the
    # Mode.JSON fix: the provider returned `criteria` as a JSON-encoded
    # string instead of a nested object.
    with pytest.raises(ValidationError):
        _GradingResponse.model_validate(_payload(criteria='{"vocabulary": {"score": 8}}'))


def test_rejects_missing_field():
    payload = _payload()
    del payload["overall_feedback"]
    with pytest.raises(ValidationError):
        _GradingResponse.model_validate(payload)


def test_grading_survives_crosscheck_failure_and_exposes_extra(monkeypatch):
    """A LanguageTool outage / rate limit must not fail the grading; working matches are
    reconciled against the LLM's own error list."""
    from my_essay_tutor import grading

    data = _GradingResponse.model_validate(_payload())
    monkeypatch.setattr(grading, "_CALLERS", {grading.get_settings().grading_provider: lambda p, t: data})
    text = "Er gehen nach Hause. " * 5

    def boom(*a):
        raise RuntimeError("rate limit")

    monkeypatch.setattr(grading, "grammar_crosscheck", boom)
    out = grading.grade_essay(text, "de", "B1")
    assert out.grammar_crosscheck is None and out.grammar_crosscheck_extra is None

    match = {"message": "Verb", "offset": 3, "length": 5, "replacements": ["geht"], "rule_id": "X"}
    monkeypatch.setattr(grading, "grammar_crosscheck", lambda *a: [match])
    out = grading.grade_essay(text, "de", "B1")
    assert out.grammar_crosscheck_extra[0]["span"] == "gehen"
