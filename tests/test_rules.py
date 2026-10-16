from my_essay_tutor import rules
from my_essay_tutor.grading import _build_prompt
from my_essay_tutor import exams
from my_essay_tutor.models import GrammarError


def _err(rule_id):
    return GrammarError(original="x", correction="y", explanation="e", category="c", rule_id=rule_id)


def test_known_rule_gets_link_and_title():
    out = rules.attach_rules([_err("b1-relativsaetze")])[0]
    assert out.rule_level == "B1" and out.rule_title == "Relativsätze"
    assert out.rule_url.endswith("b1.html#b1-relativsaetze") and out.rule_url.startswith("http")


def test_invented_or_missing_rule_id_is_dropped():
    for rid in ("does-not-exist", None, ""):
        out = rules.attach_rules([_err(rid)])[0]
        assert out.rule_id is None and out.rule_url is None


def test_prompt_lists_catalog_for_german_only():
    exam_obj, task_obj = exams.resolve("de", None, "B1", None)
    de = _build_prompt("German", "B1", exam_obj, task_obj, 100)
    assert "a1-artikel-genus" in de and '"rule_id"' in de
    en_exam, en_task = exams.resolve("en", None, "B1", None)
    assert "rule_id" not in _build_prompt("English", "B1", en_exam, en_task, 100)


def test_every_catalog_url_points_to_an_existing_rule_page():
    cat = rules._catalog()["rules"]
    assert len(cat) >= 50 and all(r["url"].split("#")[1] == r["id"] for r in cat.values())
