"""/api/exams and /api/grade exam/task plumbing, with the LLM call stubbed."""
from __future__ import annotations

from fastapi.testclient import TestClient

from my_essay_tutor import grading
from my_essay_tutor.api.app import app

client = TestClient(app)


def _stub(prompt: str, essay_text: str) -> grading._GradingResponse:
    c = {"score": 9, "comment": "ok"}
    criteria = {k: c for k in ("vocabulary", "coherence", "grammar", "content_relevance", "sociolinguistic")}
    _stub.last_prompt = prompt
    return grading._GradingResponse.model_validate({
        "grammar_errors": [], "achieved_level": "B2", "level_confidence": "at", "criteria": criteria,
        "strengths": [], "weaknesses": [], "overall_feedback": "ok",
    })


def test_catalog_endpoint():
    data = client.get("/api/exams").json()
    assert {e["id"] for e in data["de"]} == {"telc", "goethe"}
    goethe = next(e for e in data["de"] if e["id"] == "goethe")
    assert [t["id"] for t in next(l for l in goethe["levels"] if l["level"] == "B1")["tasks"]] == ["t1", "t2", "t3"]


def test_grade_delf_returns_exam_score(monkeypatch):
    monkeypatch.setitem(grading._CALLERS, "glm", _stub)
    text = "mot " * 260
    r = client.post("/api/grade", json={"text": text, "language_code": "fr", "target_level": "B2", "exam": "delf"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["exam"] == "delf"
    assert body["exam_score"]["max_total"] == 25.0
    assert len(body["exam_score"]["criteria"]) == 5
    assert "DELF B2" in _stub.last_prompt


def test_grade_goethe_task(monkeypatch):
    monkeypatch.setitem(grading._CALLERS, "glm", _stub)
    r = client.post("/api/grade", json={"text": "Wort " * 150, "language_code": "de", "target_level": "B2",
                                        "exam": "goethe", "task": "t1"})
    assert r.status_code == 200, r.text
    assert r.json()["exam_score"]["max_total"] == 60.0


def test_invalid_exam_for_language_is_400():
    r = client.post("/api/grade", json={"text": "x", "language_code": "fr", "target_level": "B1", "exam": "goethe"})
    assert r.status_code == 400
