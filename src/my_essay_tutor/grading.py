"""Grammar-error detection + CEFR (A1-C1) scoring, via an LLM.

Follows this session's established provider-abstraction pattern
(Settings.grading_provider selects "glm" | "gemini" | "anthropic"). GLM is
the default: it has no daily/per-minute quota fragility on this account,
unlike the Gemini free tier.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel

from .config import get_settings
from .grammar_check import grammar_crosscheck
from .models import CriteriaScores, GradingResult, GrammarError


class _GradingResponse(BaseModel):
    """The exact shape the grading prompt asks for (see _PROMPT_TEMPLATE's
    JSON schema below) - passed to instructor as `response_model` so a
    provider's structured-output call returns this directly, validated,
    with malformed/incomplete output (wrong achieved_level string, a
    missing field, etc.) triggering instructor's own retry-with-the-
    validation-error-fed-back-to-the-model loop instead of a bare
    json.loads() crash. Deliberately not GradingResult itself: this model
    covers only what the LLM actually produces - GradingResult additionally
    carries language/target_level/extracted_text/score_out_of_100, which
    grade_essay() derives itself (score_out_of_100 in particular is
    computed from the criteria, never trusted from the model - see the
    comment at its call site)."""

    grammar_errors: list[GrammarError]
    achieved_level: Literal["A1", "A2", "B1", "B2", "C1"]
    level_confidence: Literal["below", "at", "above"]
    criteria: CriteriaScores
    strengths: list[str]
    weaknesses: list[str]
    overall_feedback: str

LANGUAGE_NAMES = {"de": "German", "en": "English", "fr": "French"}

CEFR_DESCRIPTORS = """
A1: Understands and uses familiar everyday expressions and very basic phrases. Simple present-tense sentences; frequent basic errors are expected.
A2: Communicates in simple, routine tasks. Short, simple sentences, some past tense; noticeable but not overwhelming errors.
B1: Deals with most everyday situations; produces connected text on familiar topics. Some complex sentences; generally comprehensible, occasional errors.
B2: Interacts with fluency and spontaneity; produces clear, detailed text. Wide range of structures, good grammatical control; errors are rare and rarely impede understanding.
C1: Expresses ideas fluently and flexibly for social, academic and professional purposes; well-structured, complex text with only occasional minor slips.
""".strip()

_PROMPT_TEMPLATE = """You are an expert {language} language teacher grading a student's written composition against the CEFR framework.

The student was asked to write at the {target_level} level. CEFR level descriptors for reference:
{descriptors}

Grade the essay below using this four-criterion rubric, 12 points each (same rubric at every CEFR level, calibrated against the target level {target_level} each time):
- vocabulary (Wortschatz): range and precision of vocabulary for the target level.
- coherence (roter Faden): logical flow, paragraphing, connectors -- does the text read as one connected argument/narrative.
- grammar (Grammatik): grammatical accuracy and range of structures for the target level.
- content_relevance (Inhalt): how well the content addresses the topic/title and task.

Return ONLY a JSON object (no markdown fences, no commentary) matching exactly this schema:
{{
  "grammar_errors": [
    {{"original": "<exact span from the essay>", "correction": "<corrected form>", "explanation": "<short reason, in {language}>", "category": "<e.g. verb conjugation, article/gender, word order, spelling, preposition>"}}
  ],
  "achieved_level": "A1"|"A2"|"B1"|"B2"|"C1",
  "level_confidence": "below"|"at"|"above",
  "criteria": {{
    "vocabulary": {{"score": <integer 0-12>, "comment": "<1 sentence, in {language}>"}},
    "coherence": {{"score": <integer 0-12>, "comment": "<1 sentence, in {language}>"}},
    "grammar": {{"score": <integer 0-12>, "comment": "<1 sentence, in {language}>"}},
    "content_relevance": {{"score": <integer 0-12>, "comment": "<1 sentence, in {language}>"}}
  }},
  "strengths": ["..."],
  "weaknesses": ["..."],
  "overall_feedback": "2-4 sentences of constructive feedback, written directly to the student, in {language}."
}}

Rules:
- Quote exact substrings from the essay in "original" so they can be located and highlighted.
- Find every grammar, spelling and syntax error, however small.
- Score each criterion realistically against the target level {target_level}: flawless work at that level scores 10-12 on a criterion; systematic weakness scores under 6.
- If the essay text looks garbled or clearly broken by OCR (isolated nonsense characters, no coherent words), say so plainly in overall_feedback instead of inventing errors for text that likely isn't what the student wrote.
"""


def _build_prompt(language: str, target_level: str) -> str:
    return _PROMPT_TEMPLATE.format(language=language, target_level=target_level, descriptors=CEFR_DESCRIPTORS)


def _call_glm(prompt: str, essay_text: str) -> _GradingResponse:
    import instructor
    from openai import OpenAI

    settings = get_settings()
    # Mode.JSON, not the default Mode.TOOLS: GLM's tool-calling returns
    # nested objects (the `criteria` field) as a JSON-encoded STRING
    # instead of an actual object, which fails _GradingResponse's
    # validation outright (Pydantic: "Input should be an object, got
    # str") - a real compatibility gap in GLM's OpenAI-compatible tool-use
    # mode for nested schemas, found by running this against the real API,
    # not from documentation. Mode.JSON avoids tool-calling entirely.
    client = instructor.from_openai(
        OpenAI(api_key=settings.glm_api_key, base_url="https://api.z.ai/api/paas/v4/"),
        mode=instructor.Mode.JSON,
    )

    last_err: Exception | None = None
    for attempt in range(5):
        try:
            return client.chat.completions.create(
                model=settings.glm_model,
                response_model=_GradingResponse,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": essay_text},
                ],
                temperature=0.2,
            )
        except Exception as e:  # rate limit (1302) / content filter (1301) / validation failures -- all transient
            last_err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"GLM grading failed after retries: {last_err}")


def _call_gemini(prompt: str, essay_text: str) -> _GradingResponse:
    import instructor
    from google import genai

    settings = get_settings()
    client = instructor.from_genai(genai.Client(api_key=settings.gemini_api_key))

    last_err: Exception | None = None
    for attempt in range(5):
        try:
            return client.chat.completions.create(
                model=settings.gemini_model,
                response_model=_GradingResponse,
                messages=[{"role": "user", "content": f"{prompt}\n\nESSAY:\n{essay_text}"}],
            )
        except Exception as e:
            last_err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Gemini grading failed after retries: {last_err}")


def _call_anthropic(prompt: str, essay_text: str) -> _GradingResponse:
    import anthropic
    import instructor

    settings = get_settings()
    client = instructor.from_anthropic(anthropic.Anthropic(api_key=settings.anthropic_api_key))

    return client.chat.completions.create(
        model=settings.anthropic_model,
        response_model=_GradingResponse,
        max_tokens=4096,
        system=prompt,
        messages=[{"role": "user", "content": essay_text}],
    )


_CALLERS = {"glm": _call_glm, "gemini": _call_gemini, "anthropic": _call_anthropic}


def grade_essay(text: str, language_code: str, target_level: str) -> GradingResult:
    if language_code not in LANGUAGE_NAMES:
        raise ValueError(f"Unsupported language_code: {language_code}")

    settings = get_settings()
    language = LANGUAGE_NAMES[language_code]
    prompt = _build_prompt(language, target_level)

    caller = _CALLERS.get(settings.grading_provider)
    if caller is None:
        raise ValueError(f"Unknown grading_provider: {settings.grading_provider}")

    data = caller(prompt, text)  # a validated _GradingResponse, not a raw dict

    criteria = data.criteria
    # Derived, not trusted from the model's own arithmetic: sum of the four
    # 12-point criteria, scaled to /100 (48 points max -> x100/48).
    criteria_total = (
        criteria.vocabulary.score + criteria.coherence.score + criteria.grammar.score + criteria.content_relevance.score
    )
    score_out_of_100 = round(criteria_total * 100 / 48)

    return GradingResult(
        language=language_code,
        target_level=target_level,
        extracted_text=text,
        grammar_errors=data.grammar_errors,
        achieved_level=data.achieved_level,
        level_confidence=data.level_confidence,
        criteria=criteria,
        score_out_of_100=score_out_of_100,
        strengths=data.strengths,
        weaknesses=data.weaknesses,
        overall_feedback=data.overall_feedback,
        grammar_crosscheck=grammar_crosscheck(text, language_code),
    )
