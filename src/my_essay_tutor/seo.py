"""GEO/SEO: robots.txt, sitemap.xml, llms.txt and head/about injection for the UI page.

Most AI crawlers do not run JavaScript, so canonical, JSON-LD and a static about/FAQ section
are added to the served HTML. Set SEO_DISABLED=1 when the app is mounted inside another site
(that site owns the SEO then). SITE_URL pins the public origin; otherwise it is derived
from the request (https off localhost).
"""
from __future__ import annotations

import html
import json
import os
import re

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, Response

TITLE = "My Essay Tutor: AI Feedback on German, English and French Compositions"
DESC = ("Free writing tutor: upload or write a composition in German, English or French, pick an exam "
        "(telc, Goethe, DELF/DALF) and level, and get grammar feedback with an exam-style score.")
FAQ = [
    ("Which exams and languages are supported?",
     "German, English and French compositions, graded against exam formats such as telc, Goethe and DELF/DALF, with a CEFR level."),
    ("Does an AI decide the grammar errors?",
     "Grammar rules are checked by deterministic code; a language model supports the grading and feedback text."),
    ("Can I upload a handwritten text?",
     "Yes. Photos and PDFs are read with OCR first and you can correct the extracted text before grading."),
]
_LOCAL = ("localhost", "127.0.0.1", "testserver")


_HOST = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?(?::\d{1,5})?$")


def _safe_host(request: Request) -> str:
    """The Host / X-Forwarded-Host headers are text from the client and end up inside HTML, the sitemap and robots.txt:
    only a plain host name (optionally with a port) is accepted, anything else falls back."""
    for cand in (request.headers.get("x-forwarded-host", "").split(",")[0].strip(), request.url.netloc):
        if cand and _HOST.match(cand):
            return cand
    return "localhost"


def site_url(request: Request) -> str:
    env = os.environ.get("SITE_URL")
    if env:
        return env.rstrip("/")
    render = os.environ.get("RENDER_EXTERNAL_URL")  # set by Render itself: on Render no request header decides the origin
    if render and re.match(r"^https?://[A-Za-z0-9.-]+(:\d+)?/?$", render):
        return render.rstrip("/")
    host = _safe_host(request)
    proto = "http"  # local development; everything else is https (set just below)
    if host.split(":")[0] not in _LOCAL:
        proto = "https"
    return f"{proto}://{host}"


def _about() -> str:
    e = html.escape
    body = "".join(f"<h3>{e(q)}</h3><p>{e(a)}</p>" for q, a in FAQ)
    style = ("<style>.seo-about{max-width:860px;margin:2rem auto;padding:1rem 1.25rem;border-top:1px solid rgba(128,128,128,.35);"
             "font:15px/1.6 system-ui,sans-serif;opacity:.9}.seo-about h2{font-size:1.1rem;margin:0 0 .5rem}"
             ".seo-about h3{font-size:1rem;margin:.9rem 0 .2rem}.seo-about p{margin:.25rem 0}</style>")
    return f'{style}<section class="seo-about" id="about"><h2>About My Essay Tutor</h2><p>{e(DESC)}</p>{body}</section>'


def install(app: FastAPI) -> None:
    @app.middleware("http")
    async def _seo_head(request: Request, call_next):
        response = await call_next(request)
        if (os.environ.get("SEO_DISABLED") or request.method != "GET" or request.url.path != "/ui/"
                or response.status_code != 200 or "text/html" not in response.headers.get("content-type", "")):
            return response
        page = b"".join([chunk async for chunk in response.body_iterator]).decode("utf-8")
        url = site_url(request) + "/ui/"
        ld = [{"@context": "https://schema.org", "@type": "WebApplication", "name": "My Essay Tutor", "description": DESC,
               "url": url, "applicationCategory": "EducationalApplication", "operatingSystem": "Any (web browser)",
               "isAccessibleForFree": True, "offers": {"@type": "Offer", "price": "0", "priceCurrency": "USD"}},
              {"@context": "https://schema.org", "@type": "FAQPage",
               "mainEntity": [{"@type": "Question", "name": q, "acceptedAnswer": {"@type": "Answer", "text": a}} for q, a in FAQ]}]
        url_attr = html.escape(url, quote=True)
        head = (f'<link rel="canonical" href="{url_attr}">\n<meta property="og:title" content="{html.escape(TITLE)}">\n'
                f'<meta property="og:description" content="{html.escape(DESC)}">\n<meta property="og:url" content="{url_attr}">\n'
                f'<script type="application/ld+json">{json.dumps(ld, ensure_ascii=False).replace("<", "\\u003c")}</script>\n')
        page = page.replace("</head>", head + "</head>", 1).replace("</body>", _about() + "\n</body>", 1)
        headers = {k: v for k, v in response.headers.items() if k.lower() not in ("content-length", "content-encoding")}
        return Response(content=page, status_code=200, headers=headers, media_type="text/html")

    @app.get("/robots.txt", include_in_schema=False)
    def robots(request: Request) -> PlainTextResponse:
        lines = ["User-agent: *", "Allow: /", "Disallow: /api/", "Disallow: /docs", "Disallow: /openapi.json", ""]
        for bot in ("OAI-SearchBot", "ChatGPT-User", "PerplexityBot", "Claude-SearchBot", "Claude-User"):
            lines += [f"User-agent: {bot}", "Allow: /", ""]
        for bot in ("GPTBot", "ClaudeBot", "Google-Extended", "CCBot"):
            lines += [f"User-agent: {bot}", "Disallow: /", ""]
        lines.append(f"Sitemap: {site_url(request)}/sitemap.xml")
        return PlainTextResponse("\n".join(lines) + "\n")

    @app.get("/sitemap.xml", include_in_schema=False)
    def sitemap(request: Request) -> Response:
        return Response('<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
                        f"<url><loc>{site_url(request)}/ui/</loc></url></urlset>", media_type="application/xml")

    @app.get("/llms.txt", include_in_schema=False)
    def llms(request: Request) -> PlainTextResponse:
        return PlainTextResponse(f"# My Essay Tutor\n\n> {DESC}\n\n- [Essay Tutor]({site_url(request)}/ui/): the tool\n")
