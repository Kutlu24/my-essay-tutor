from pydantic import BaseModel


class GrammarError(BaseModel):
    original: str
    correction: str
    explanation: str
    category: str


class CriterionScore(BaseModel):
    score: int  # 0-12
    comment: str


class CriteriaScores(BaseModel):
    """Four-criterion rubric (Wortschatz / roter Faden / Grammatik / Inhalt),
    12 points each, 48 total -- same rubric used across every CEFR level."""

    vocabulary: CriterionScore
    coherence: CriterionScore
    grammar: CriterionScore
    content_relevance: CriterionScore


class GradingResult(BaseModel):
    language: str
    target_level: str
    extracted_text: str
    grammar_errors: list[GrammarError]
    achieved_level: str
    level_confidence: str  # "below" | "at" | "above"
    criteria: CriteriaScores
    score_out_of_100: int
    strengths: list[str]
    weaknesses: list[str]
    overall_feedback: str
    # LanguageTool's independent rule-based match list, or None if
    # Settings.enable_grammar_crosscheck is off (the default) - see
    # grammar_check.py. Not reconciled against grammar_errors above; the
    # two lists are shown side by side, not merged.
    grammar_crosscheck: list[dict] | None = None
