"""Standalone MCP server exposing the two essay-tutor capabilities as tools:
extraction (HTR/OCR) and grading (grammar + CEFR).

Run directly (`python -m my_essay_tutor.mcp_server`) to attach this to
Claude Desktop or any other MCP client over stdio. The web app in
api/app.py calls the same underlying functions in-process for the
student-facing upload flow -- see the README for why (avoiding a fragile
subprocess-per-request MCP client hop on Render's free tier) -- so this
server and the web app share one implementation, exposed through two
different consumption paths.
"""

from __future__ import annotations

import base64
import io

from mcp.server.fastmcp import FastMCP
from PIL import Image

from . import grading, ocr, pdf_utils

mcp = FastMCP("my-essay-tutor")


@mcp.tool()
def extract_essay_text(file_base64: str, is_pdf: bool = False) -> str:
    """Extract handwritten or typed text from a student essay page, via a
    vision LLM (see ocr.py).

    file_base64: base64-encoded JPEG/PNG image, or PDF bytes if is_pdf=True.
    is_pdf: set True when file_base64 decodes to a PDF rather than an image.
    """
    raw = base64.b64decode(file_base64)
    if is_pdf:
        pages = pdf_utils.pdf_to_images(raw)
    else:
        pages = [Image.open(io.BytesIO(raw)).convert("RGB")]

    texts = [ocr.extract_text_from_page(page) for page in pages]
    return "\n\n".join(t for t in texts if t)


@mcp.tool()
def grade_essay(text: str, language_code: str, target_level: str) -> dict:
    """Grade a student essay for grammar and CEFR level.

    language_code: "de" | "en" | "fr"
    target_level: "A1" | "A2" | "B1" | "B2" | "C1"
    """
    result = grading.grade_essay(text, language_code, target_level)
    return result.model_dump()


if __name__ == "__main__":
    mcp.run()
