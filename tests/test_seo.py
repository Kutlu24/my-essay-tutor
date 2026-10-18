from fastapi.testclient import TestClient

from my_essay_tutor.api.app import app

client = TestClient(app)


def test_robots_sitemap_llms():
    assert "Sitemap: http://testserver/sitemap.xml" in client.get("/robots.txt").text
    assert "<loc>http://testserver/ui/</loc>" in client.get("/sitemap.xml").text
    assert "Essay Tutor" in client.get("/llms.txt").text


def test_ui_has_canonical_jsonld_and_about_once():
    t = client.get("/ui/", headers={"x-forwarded-host": "demo.onrender.com", "x-forwarded-proto": "http"}).text
    assert t.count('rel="canonical"') == 1 and 'href="https://demo.onrender.com/ui/"' in t
    assert t.count("application/ld+json") == 1 and t.count('id="about"') == 1
