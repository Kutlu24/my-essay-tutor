"""Limits in front of the costly endpoints (guard.py) and what errors may say. No model is called."""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from my_essay_tutor import grading, guard, ocr
from my_essay_tutor.api import app as app_mod


@pytest.fixture
def c(monkeypatch):
    guard.limiter.reset()
    monkeypatch.setattr(ocr, "extract_text_from_page", lambda page: "gelesener Text")
    return TestClient(app_mod.app, client=("127.0.0.1", 50000))


def png(size=(60, 40)):
    buf = io.BytesIO(); Image.new("RGB", size, "white").save(buf, "PNG")
    return buf.getvalue()


GRADE = {"text": "Ich lerne Deutsch.", "language_code": "de", "target_level": "B1"}


def test_grade_is_limited_per_client_and_globally(c, monkeypatch):
    monkeypatch.setattr(grading, "grade_essay", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stub")))
    monkeypatch.setitem(guard.RULES, "/api/grade", (2, 4))
    assert [c.post("/api/grade", json=GRADE).status_code for _ in range(4)] == [502, 502, 429, 429]
    assert int(c.post("/api/grade", json=GRADE).headers["retry-after"]) >= 1
    # left-hand X-Forwarded-For entries are the caller's own text: they do not make a new client
    assert c.post("/api/grade", json=GRADE, headers={"X-Forwarded-For": "10.1.1.1, 127.0.0.1"}).status_code == 429
    codes = [c.post("/api/grade", json=GRADE, headers={"X-Forwarded-For": f"198.51.100.{i}"}).status_code for i in range(4)]
    assert codes == [502, 502, 429, 429]          # the global budget (4) is used up after two more
    assert c.get("/api/exams").status_code == 200 and c.get("/health").status_code == 200


def test_limits_also_apply_under_a_mount_prefix(monkeypatch):
    from fastapi import FastAPI
    guard.limiter.reset()
    monkeypatch.setattr(grading, "grade_essay", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stub")))
    monkeypatch.setitem(guard.RULES, "/api/grade", (1, 50))
    hub = FastAPI(); hub.mount("/schreiben", app_mod.app)
    h = TestClient(hub, client=("127.0.0.1", 50000))
    assert [h.post("/schreiben/api/grade", json=GRADE).status_code for _ in range(2)] == [502, 429]


def test_extract_works_and_refuses_oversized_input(c, monkeypatch):
    r = c.post("/api/extract", files={"file": ("p.png", png(), "image/png")})
    assert r.status_code == 200 and r.json() == {"extracted_text": "gelesener Text", "pages": 1}
    monkeypatch.setattr(app_mod, "MAX_PIXELS", 1000)
    assert c.post("/api/extract", files={"file": ("p.png", png(), "image/png")}).status_code == 413
    monkeypatch.setitem(guard.BODY_LIMITS, "/api/extract", 3000)
    assert c.post("/api/extract", files={"file": ("p.png", b"\x89PNG" + b"z" * 9000, "image/png")}).status_code == 413
    big = b'{"text": "' + b"x" * (guard.DEFAULT_BODY + 10) + b'"}'
    assert c.post("/api/grade", content=big, headers={"Content-Type": "application/json"}).status_code == 413
    assert c.post("/api/grade", content=(b"y" * 8192 for _ in range(60)), headers={"Content-Type": "application/json"}).status_code == 413


def test_errors_do_not_echo_internals(c, monkeypatch):
    r = c.post("/api/extract", files={"file": ("p.png", b"definitely not an image /srv/secret", "image/png")})
    assert r.status_code == 400 and "secret" not in r.text
    monkeypatch.setattr(ocr, "extract_text_from_page", lambda page: (_ for _ in ()).throw(ocr.OCRError("key=AIza-secret quota")))
    r = c.post("/api/extract", files={"file": ("p.png", png(), "image/png")})
    assert r.status_code == 502 and "AIza" not in r.text
    monkeypatch.setattr(grading, "grade_essay", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Bearer sk-secret")))
    r = c.post("/api/grade", json=GRADE)
    assert r.status_code == 502 and "sk-secret" not in r.text and "RuntimeError" in r.text
    assert c.post("/api/grade", json={**GRADE, "exam": "x" * 500}).status_code == 422


def test_requests_refused_by_validation_are_not_charged(c, monkeypatch):
    monkeypatch.setattr(grading, "grade_essay", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stub")))
    monkeypatch.setitem(guard.RULES, "/api/grade", (2, 3))
    for _ in range(10):
        assert c.post("/api/grade", json={**GRADE, "target_level": "Z9"}).status_code == 400
    assert [c.post("/api/grade", json=GRADE).status_code for _ in range(3)] == [502, 502, 429]
    assert guard.bucket("2a02:1:2:300::1") == guard.bucket("2a02:1:2:3ff::9") != guard.bucket("2a02:1:2:400::1")


def test_pdf_pages_are_rendered_under_a_pixel_cap():
    import pypdfium2 as pdfium
    from my_essay_tutor import pdf_utils
    assert pdf_utils.page_scale(595, 842, 2.0) == 2.0                      # A4 is untouched
    s = pdf_utils.page_scale(14400, 14400, 2.0)                             # a "200 x 200 inch" page
    assert s < 2.0 and (14400 * s) * (14400 * s) <= pdf_utils.MAX_PAGE_PIXELS * 1.001
    doc = pdfium.PdfDocument.new(); doc.new_page(14400, 14400); doc.new_page(595, 842)
    import io
    buf = io.BytesIO(); doc.save(buf)
    big, a4 = pdf_utils.pdf_to_images(buf.getvalue())
    assert big.width * big.height <= pdf_utils.MAX_PAGE_PIXELS * 1.01 and a4.size == (1190, 1684)


def test_a_direct_caller_cannot_pick_its_own_identity():
    from my_essay_tutor import guard as g
    assert g.client_of("203.0.113.50", ["1.2.3.4"]) == "203.0.113.50"          # public peer: the header is ignored
    assert g.client_of("203.0.113.50", []) == "203.0.113.50"
    assert g.client_of("10.0.0.2", ["9.9.9.9", "198.51.100.7"]) == "198.51.100.7"   # our proxy: its own (right-most) entry
    assert g.client_of("127.0.0.1", []) == "127.0.0.1" and g.client_of("", []) == "unknown"
    assert g.peer_is_proxy("172.18.0.3") and g.peer_is_proxy("::1") and not g.peer_is_proxy("8.8.8.8") and not g.peer_is_proxy("testclient")
