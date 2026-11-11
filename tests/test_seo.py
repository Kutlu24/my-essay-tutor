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

PATHS = ["/ui/", "/robots.txt", "/sitemap.xml"]


def test_hostile_host_headers_never_reach_the_page(monkeypatch):
    monkeypatch.delenv("SITE_URL", raising=False); monkeypatch.delenv("RENDER_EXTERNAL_URL", raising=False)
    evil = 'x"><script>alert(1)</script>'
    for path in PATHS:
        for headers in ({"x-forwarded-host": evil}, {"x-forwarded-host": "a.example</script><script>alert(1)//"}, {"host": "evil.example\"onload=x"}):
            r = client.get(path, headers=headers)
            if r.status_code != 200:
                continue                      # a refused request shows nothing
            assert "<script>alert(1)" not in r.text and "onload=x" not in r.text, (path, headers)
    assert "demo.onrender.com" in client.get(PATHS[0], headers={"x-forwarded-host": "demo.onrender.com"}).text


def test_on_render_the_platform_origin_wins_over_request_headers(monkeypatch):
    monkeypatch.delenv("SITE_URL", raising=False)
    monkeypatch.setenv("RENDER_EXTERNAL_URL", "https://real-site.onrender.com")
    t = client.get(PATHS[0], headers={"x-forwarded-host": "attacker.example"}).text
    assert "real-site.onrender.com" in t and "attacker.example" not in t
