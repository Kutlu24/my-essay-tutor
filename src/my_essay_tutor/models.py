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
    12 points each, 48 total -- scored per target level against the telc
    "Schreiben" criteria (see grading._TELC_RUBRICS: A1/A2 Leitpunkt-focused,
    B1/B2 Allgemein's three A-D criteria and C1 Hochschule's four 12/8/4/0
    criteria all map onto these four buckets)."""

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
