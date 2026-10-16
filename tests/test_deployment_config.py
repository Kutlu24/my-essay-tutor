"""ALLOWED_LANGUAGES / APP_TITLE: the standalone site offers de/en/fr, an embedded German-only
deployment (German Trainer platform) restricts catalog and grading to German."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from my_essay_tutor.api.app import app
from my_essay_tutor.config import get_settings

client = TestClient(app)


@pytest.fixture(autouse=True)
def _fresh_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_default_is_three_languages():
    cfg = client.get("/api/config").json()
    assert cfg["languages"] == ["de", "en", "fr"] and cfg["title"] == "My Essay Tutor"
    assert set(client.get("/api/exams").json()) == {"de", "en", "fr"}


def test_german_only_deployment(monkeypatch):
    monkeypatch.setenv("ALLOWED_LANGUAGES", "de")
    monkeypatch.setenv("APP_TITLE", "Schreibwerkstatt")
    cfg = client.get("/api/config").json()
    assert cfg["languages"] == ["de"] and cfg["title"] == "Schreibwerkstatt" and "EN" not in cfg["kicker"]
    assert set(client.get("/api/exams").json()) == {"de"}
    r = client.post("/api/grade", json={"text": "Hello world", "language_code": "en", "target_level": "B1"})
    assert r.status_code == 400 and "de" in r.json()["detail"]
