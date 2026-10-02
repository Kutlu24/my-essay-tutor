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

# telc "Schreiben" / "Schriftlicher Ausdruck" assessment criteria, per exam.
# Sources: telc exam handbooks, Tipps-fuer-Teilnehmer PDFs and the telc C1
# Hochschule Bewertungsraster (telc.net). B1/B2 Allgemein grade THREE criteria
# (Aufgabenbewältigung, Kommunikative Gestaltung, Formale Richtigkeit) in
# A-D bands; C1 Hochschule grades FOUR criteria with fixed anchors (12/8/4/0);
# A1/A2 are task-based (form + short message) with the Leitpunkte as the core
# criterion. Our fixed four-way schema always maps onto these per level.
_TELC_EXAMS = {
    "A1": 'telc Deutsch A1 ("Start Deutsch 1")',
    "A2": 'telc Deutsch A2 ("Start Deutsch 2")',
    "B1": "telc Deutsch B1 (Allgemein)",
    "B2": "telc Deutsch B2 (Allgemein)",
    "C1": "telc Deutsch C1 Hochschule",
}

_TELC_RUBRICS = {
    "A1": """Task type: filling in a form plus a short everyday message (~30 words, three mandatory content points / Leitpunkte). What telc raters weigh at A1:
- content_relevance (Inhalt): every given Leitpunkt must appear -- one simple sentence per point is sufficient. A missing Leitpunkt is the most serious telc deduction at this level.
- coherence (Kommunikative Gestaltung): a simple, understandable everyday message in the right text type (note/email); short clear sentences; a minimal greeting/closing where the text type expects one.
- grammar + vocabulary (Formale Richtigkeit): very basic words and structures only. Frequent basic errors are EXPECTED at A1 and only matter when they block understanding ("Verständlichkeit geht vor").
telc bands mapped to 0-12: A = all points covered, understandable (10-12); B = all points, some errors (7-9); C = one point missing or many errors (4-6); D = barely understandable or most points missing (0-3).""",
    "A2": """Task type: a short semi-formal everyday message/letter covering given Leitpunkte. What telc raters weigh at A2:
- content_relevance (Inhalt): all Leitpunkte treated, content relevant to the given situation; a missing one is the most serious telc deduction at this level.
- coherence (Kommunikative Gestaltung): a simple connected text with basic connectors (und, aber, dann, weil), appropriate to the everyday situation and its text type/register.
- grammar + vocabulary (Formale Richtigkeit): elementary structures (present, Perfekt, simple requests); noticeable errors are tolerated at A2 as long as the text stays comprehensible ("Verständlichkeit geht vor").
telc bands mapped to 0-12: A = all points, clear and largely correct (10-12); B = all points, noticeable errors (7-9); C = one point missing or errors that partly hinder understanding (4-6); D = most points missing or hard to understand (0-3).""",
    "B1": """Task type: a personal or semi-formal letter/email (~150 words, 30 minutes) covering four given Leitpunkte. Official telc B1 rubric -- three criteria, graded A-D:
- Aufgabenbewaeltigung -> content_relevance: ALL four Leitpunkte addressed, at least one full sentence each. A single missing Leitpunkt drops the band (A to B or worse).
- Kommunikative Gestaltung -> coherence: clear letter structure (Anrede, purpose, body, Grußformel), connectors (ausserdem, deshalb, trotzdem), the register the task demands (Sie vs. du) held consistently; avoid starting every sentence with "Ich"/"Wir".
- Formale Richtigkeit -> grammar + vocabulary: B1-level grammar, syntax, spelling and vocabulary, weighted by "primacy of comprehensibility" -- errors that do not impede understanding cost less than ones that confuse the reader.
telc bands mapped to 0-12: A = all points, clean structure/register, few errors (10-12); B = solid with some slips (7-9); C = one point missing or frequent errors (4-6); D = several points missing or seriously hindering errors (0-3).""",
    "B2": """Task type: a formal letter (e.g. request or complaint, ~150+ words) treating the required Leitpunkte in a strictly formal register. Official telc B2 rubric -- three criteria, graded A-D:
- Aufgabenbewaeltigung -> content_relevance: the required Leitpunkte covered fully and appropriately for the formal situation.
- Kommunikative Gestaltung -> coherence: formal register (Sie) held even in an emotional complaint; letter conventions (Betreff, Anrede, Schlussformel); logical structure with varied connectors.
- Formale Richtigkeit -> grammar + vocabulary: differentiated B2 repertoire expected -- complex structures (Passiv, Konjunktiv II, Nominalstil), precise vocabulary; errors are rare and must not impede understanding.
telc bands mapped to 0-12: A = self-assured, differentiated, near-clean (10-12); B = generally clear and correct with some slips (7-9); C = register/structure problems or frequent errors (4-6); D = task largely missed or comprehension seriously hindered (0-3).""",
    "C1": """Task type: an academic Erörterung essay (>= 350 words, 70 minutes) taking a position on two given opposing statements. Official telc C1 Hochschule rubric -- four criteria with fixed anchors A=12, B=8, C=4, D=0:
- Aufgabengerechtheit -> content_relevance: BOTH given statements explicitly acknowledged, a clear well-reasoned personal position taken, arguments developed substantively. Missing a statement or drifting off-topic costs severely.
- Korrektheit -> grammar: grammar, morphology, syntax, spelling and punctuation. Minor slips in complex structures are acceptable at C1 if readability is untouched; frequent errors in basic structures lower the band.
- Repertoire -> vocabulary: range beyond everyday language -- academic vocabulary, passive voice, participial constructions; persistent simple/repetitive verbs and structures cap the band.
- Kommunikative Gestaltung -> coherence: clear paragraphing (4-5 paragraphs), varied connectors, precise cohesive references; formal, objective academic tone, minimal "ich" perspectivity.
Score anchors (telc C1 HS, per criterion): 12 = A, 8 = B, 4 = C, 0 = D; intermediate values (10-11, 6-7, 2-3) interpolate within a band.""",
}

_PROMPT_TEMPLATE = """You are an expert {language} language teacher grading a student's written composition against the telc "Schreiben" assessment framework.

The student was asked to write at the {target_level} level -- calibrated for the {telc_exam} exam:

{telc_rubric}

CEFR level descriptors (for judging achieved_level, the writer's ACTUAL level -- not the target):
{descriptors}

Grade the essay below. Score these four schema criteria, 12 points each (48 total), applying the telc criteria and band mapping above:
- vocabulary: telc Wortschatz / Repertoire.
- coherence: telc Kommunikative Gestaltung (structure, connectors, register, cohesion).
- grammar: telc Formale Richtigkeit / Korrektheit (grammar, syntax, spelling, punctuation).
- content_relevance: telc Aufgabenbewaeltigung / Aufgabengerechtheit (Leitpunkte/task fully treated).

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
    if target_level not in _TELC_RUBRICS:
        raise ValueError(f"Unsupported target_level: {target_level}")
    return _PROMPT_TEMPLATE.format(
        language=language,
        target_level=target_level,
        telc_exam=_TELC_EXAMS[target_level],
        telc_rubric=_TELC_RUBRICS[target_level],
        descriptors=CEFR_DESCRIPTORS,
    )


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
