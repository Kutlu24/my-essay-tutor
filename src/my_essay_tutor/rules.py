"""Link grammar errors to the matching rule on the DeutschGrammatikMeister site (separate site).

`grammar_rules.json` is a copy of that site's `regeln.json` catalog (stable rule ids). Refresh it
with: cp ../deutsch-grammatik-meister/regeln.json src/my_essay_tutor/grammar_rules.json
Only German essays get rule links; an id the model invents that is not in the catalog is dropped
rather than shown, so a link never points to a page that does not exist."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .config import get_settings
from .models import GrammarError


@lru_cache
def _catalog() -> dict:
    data = json.loads((Path(__file__).parent / "grammar_rules.json").read_text(encoding="utf-8"))
    return {"site_url": data["site_url"], "rules": {r["id"]: r for r in data["rules"]}}


def prompt_section() -> str:
    lines = "\n".join(f'- {r["id"]}: {r["title"]} ({r["level"]})' for r in _catalog()["rules"].values())
    return (
        "\nRule catalog: for each grammar error, set \"rule_id\" to the id of the ONE catalog rule that "
        "best explains it, or null if no rule fits (spelling, vocabulary, punctuation). Never invent ids.\n"
        + lines + "\n"
    )


def attach_rules(errors: list[GrammarError]) -> list[GrammarError]:
    cat = _catalog()
    base = (get_settings().grammar_site_url or cat["site_url"]).rstrip("/") + "/"
    out = []
    for e in errors:
        rule = cat["rules"].get(e.rule_id or "")
        out.append(e.model_copy(update={
            "rule_id": rule["id"] if rule else None,
            "rule_title": rule["title"] if rule else None,
            "rule_level": rule["level"] if rule else None,
            "rule_url": base + rule["url"] if rule else None,
        }))
    return out
