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
