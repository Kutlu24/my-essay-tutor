from pydantic import BaseModel


class GrammarError(BaseModel):
    original: str
    correction: str
    explanation: str
    category: str


class GradingResult(BaseModel):
    language: str
    target_level: str
    extracted_text: str
    grammar_errors: list[GrammarError]
    achieved_level: str
    level_confidence: str  # "below" | "at" | "above"
    score_out_of_100: int
    strengths: list[str]
    weaknesses: list[str]
    overall_feedback: str
