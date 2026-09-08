"""Grammar-error detection + CEFR (A1-C1) scoring, via an LLM.

Follows this session's established provider-abstraction pattern
(Settings.grading_provider selects "glm" | "gemini" | "anthropic"). GLM is
the default: it has no daily/per-minute quota fragility on this account,
unlike the Gemini free tier.
"""

from __future__ import annotations

import json
import time

from .config import get_settings
from .models import GradingResult, GrammarError

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

Grade the essay below. Return ONLY a JSON object (no markdown fences, no commentary) matching exactly this schema:
{{
  "grammar_errors": [
    {{"original": "<exact span from the essay>", "correction": "<corrected form>", "explanation": "<short reason, in {language}>", "category": "<e.g. verb conjugation, article/gender, word order, spelling, preposition>"}}
  ],
  "achieved_level": "A1"|"A2"|"B1"|"B2"|"C1",
  "level_confidence": "below"|"at"|"above",
  "score_out_of_100": <integer, calibrated against the target level {target_level}>,
  "strengths": ["..."],
  "weaknesses": ["..."],
  "overall_feedback": "2-4 sentences of constructive feedback, written directly to the student, in {language}."
}}

Rules:
- Quote exact substrings from the essay in "original" so they can be located and highlighted.
- Find every grammar, spelling and syntax error, however small.
- Score realistically: flawless writing at the target level scores 85-100; systematic errors well below the target level score under 50.
- If the essay text looks garbled or clearly broken by OCR (isolated nonsense characters, no coherent words), say so plainly in overall_feedback instead of inventing errors for text that likely isn't what the student wrote.
"""


def _parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        raw = raw.removeprefix("json").strip()
    return json.loads(raw)


def _build_prompt(language: str, target_level: str) -> str:
    return _PROMPT_TEMPLATE.format(language=language, target_level=target_level, descriptors=CEFR_DESCRIPTORS)


def _call_glm(prompt: str, essay_text: str) -> dict:
    from openai import OpenAI

    settings = get_settings()
    client = OpenAI(api_key=settings.glm_api_key, base_url="https://api.z.ai/api/paas/v4/")

    last_err: Exception | None = None
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=settings.glm_model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": essay_text},
                ],
                temperature=0.2,
            )
            return _parse_json_response(resp.choices[0].message.content)
        except Exception as e:  # rate limit (1302) / content filter (1301) / json errors -- all transient
            last_err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"GLM grading failed after retries: {last_err}")


def _call_gemini(prompt: str, essay_text: str) -> dict:
    from google import genai

    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key)

    last_err: Exception | None = None
    for attempt in range(5):
        try:
            resp = client.models.generate_content(
                model=settings.gemini_model,
                contents=f"{prompt}\n\nESSAY:\n{essay_text}",
            )
            return _parse_json_response(resp.text)
        except Exception as e:
            last_err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Gemini grading failed after retries: {last_err}")


def _call_anthropic(prompt: str, essay_text: str) -> dict:
    import anthropic

    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    resp = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=4096,
        system=prompt,
        messages=[{"role": "user", "content": essay_text}],
    )
    raw = "".join(block.text for block in resp.content if block.type == "text")
    return _parse_json_response(raw)


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

    data = caller(prompt, text)

    return GradingResult(
        language=language_code,
        target_level=target_level,
        extracted_text=text,
        grammar_errors=[GrammarError(**e) for e in data["grammar_errors"]],
        achieved_level=data["achieved_level"],
        level_confidence=data["level_confidence"],
        score_out_of_100=int(data["score_out_of_100"]),
        strengths=data["strengths"],
        weaknesses=data["weaknesses"],
        overall_feedback=data["overall_feedback"],
    )
