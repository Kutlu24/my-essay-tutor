"""Exam catalog, validation and exam-native scoring (exams.py) - no LLM calls."""
from __future__ import annotations

import pytest

from my_essay_tutor import exams
from my_essay_tutor.grading import _build_prompt
from my_essay_tutor.models import CriteriaScores, CriterionScore


def _criteria(score: int, **overrides) -> CriteriaScores:
    def c(v):
        return CriterionScore(score=v, comment="")

    base = dict(vocabulary=c(score), coherence=c(score), grammar=c(score), content_relevance=c(score),
                sociolinguistic=c(score), objectivity=c(score))
    base.update({k: c(v) for k, v in overrides.items()})
    return CriteriaScores(**base)


def test_defaults_per_language():
    assert exams.resolve("de", None, "B1", None)[0].id == "telc"
    assert exams.resolve("fr", None, "B2", None)[0].id == "delf"
    assert exams.resolve("fr", None, "C1", None)[0].id == "dalf"
    assert exams.resolve("en", None, "B2", None)[0].id == "cefr"


def test_rejects_invalid_selections():
    with pytest.raises(ValueError):
        exams.resolve("fr", "goethe", "B1", None)  # wrong language
    with pytest.raises(ValueError):
        exams.resolve("fr", "delf", "C1", None)  # DELF stops at B2
    with pytest.raises(ValueError):
        exams.resolve("de", "goethe", "A1", None)  # no verified Goethe A1 sheet
    with pytest.raises(ValueError):
        exams.resolve("de", "goethe", "B1", "t9")


def test_catalog_lists_every_exam_for_its_languages():
    cat = exams.catalog()
    assert {e["id"] for e in cat["de"]} == {"telc", "goethe"}
    assert {e["id"] for e in cat["fr"]} == {"delf", "dalf", "cefr"}
    assert {e["id"] for e in cat["en"]} == {"cefr"}


def test_delf_b2_full_marks_is_25():
    exam, task = exams.resolve("fr", "delf", "B2", None)
    s = exams.score_task(exam, "B2", task, _criteria(12), word_count=260, off_topic="none")
    assert (s.total, s.max_total) == (25.0, 25.0)
    assert all(r.band == "B2+" for r in s.criteria)


def test_delf_a1_official_point_values():
    exam, task = exams.resolve("fr", "delf", "A1", None)
    s = exams.score_task(exam, "A1", task, _criteria(8), word_count=40, off_topic="none")  # "at level" band
    assert [r.points for r in s.criteria] == [2.0] * 5
    assert s.max_total == 15.0


def test_delf_under_half_the_words_scores_zero():
    exam, task = exams.resolve("fr", "delf", "B2", None)
    s = exams.score_task(exam, "B2", task, _criteria(12), word_count=124, off_topic="none")  # <= 124 -> 0
    assert s.total == 0
    s = exams.score_task(exam, "B2", task, _criteria(12), word_count=125, off_topic="none")
    assert s.total == 25.0


def test_delf_thematic_off_topic_caps_task_and_lexique_below_plus():
    exam, task = exams.resolve("fr", "delf", "B2", None)
    s = exams.score_task(exam, "B2", task, _criteria(12), word_count=250, off_topic="thematic")
    by_label = {r.label: r.band for r in s.criteria}
    assert by_label["Réalisation de la tâche"] == "B2"
    assert by_label["Lexique"] == "B2"
    assert by_label["Morphosyntaxe"] == "B2+"


def test_dalf_synthese_max_is_12_and_has_objectivity():
    exam, task = exams.resolve("fr", "dalf", "C1", "synthese")
    s = exams.score_task(exam, "C1", task, _criteria(12), word_count=200, off_topic="none")
    assert s.max_total == 12.0
    assert any("objectivité" in r.label for r in s.criteria)


def test_goethe_c1_t1_weights_sum_to_60():
    exam, task = exams.resolve("de", "goethe", "C1", "t1")
    s = exams.score_task(exam, "C1", task, _criteria(12), word_count=230, off_topic="none")
    assert s.max_total == 60.0 and s.total == 60.0


def test_goethe_b1_t3_weights_sum_to_20():
    exam, task = exams.resolve("de", "goethe", "B1", "t3")
    assert sum(c.points[0] for c in task.criteria) == 20


def test_goethe_e_on_aufgabenerfuellung_zeroes_task():
    exam, task = exams.resolve("de", "goethe", "B2", "t1")
    s = exams.score_task(exam, "B2", task, _criteria(12, content_relevance=0), word_count=150, off_topic="none")
    assert s.total == 0
    assert s.adjustments


def test_goethe_a2_uses_nonlinear_sheet_points():
    exam, task = exams.resolve("de", "goethe", "A2", "t1")
    s = exams.score_task(exam, "A2", task, _criteria(5), word_count=25, off_topic="none")  # band C
    assert [r.points for r in s.criteria] == [2.0, 2.0]


@pytest.mark.parametrize("language,exam_id,level,task", [
    ("de", "telc", "B1", None), ("de", "goethe", "B2", "t2"), ("fr", "delf", "A2", None),
    ("fr", "dalf", "C1", "essai"), ("en", "cefr", "B1", None),
])
def test_prompt_builds_and_lists_the_schema_criteria(language, exam_id, level, task):
    exam, t = exams.resolve(language, exam_id, level, task)
    prompt = _build_prompt({"de": "German", "en": "English", "fr": "French"}[language], level, exam, t, 100)
    assert '"content_relevance"' in prompt
    assert ('"sociolinguistic"' in prompt) == (exam.scheme == "delf")
    assert ('"objectivity"' in prompt) == (task is not None and t.id == "synthese")


def test_german_prompt_accepts_swiss_spelling_but_other_languages_do_not_mention_it():
    exam, task = exams.resolve("de", "goethe", "B1", "t1")
    assert "Swiss Standard German" in _build_prompt("German", "B1", exam, task, 80)
    exam, task = exams.resolve("fr", "delf", "B1", None)
    assert "Swiss Standard German" not in _build_prompt("French", "B1", exam, task, 160)


def test_exam_notes_follow_the_language():
    exam, task = exams.resolve("fr", "delf", "B2", None)
    fr = exams.score_task(exam, "B2", task, _criteria(12), word_count=10, off_topic="none", lang="fr")
    assert "mots" in fr.adjustments[0] and fr.note.startswith("DELF/DALF : le diplôme")
    en = exams.score_task(exam, "B2", task, _criteria(12), word_count=10, off_topic="none", lang="en")
    assert "words is under 50%" in en.adjustments[0]
    exam, task = exams.resolve("de", "goethe", "B1", "t1")
    de = exams.score_task(exam, "B1", task, _criteria(12), word_count=80, off_topic="none", lang="de")
    assert de.note.startswith("Goethe: Für das Modul")
