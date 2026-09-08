"""Handwriting/print text extraction via a vision-capable LLM.

v1 targeted TrOCR through Hugging Face's Inference API. That path closed:
HF's free serverless inference tier was cut down to a $0.10/month credit
allowance in 2026, not viable for real traffic. This module now uses a
vision LLM instead (GLM's glm-4.5v by default, Gemini as an alternative),
which:
  - needs no locally-loaded model weights -- keeps the same OOM-avoidance
    property TrOCR was originally chosen for on Render's free tier
  - reuses the same GLM/Gemini accounts already relied on for grading
  - reads a full essay page directly, rather than needing a separate
    line-segmentation pass first -- TrOCR was a line-level recognizer;
    a vision LLM handles a whole page at once, which also benchmarks
    more accurately in practice (frontier multimodal models now
    outperform dedicated open-source HTR models like TrOCR on
    handwriting recognition).

Known limitation: handwriting transcription is never perfect in any of
these approaches -- the frontend has a "review transcription" step before
grading for exactly this reason.
"""

from __future__ import annotations

import base64
import io
import time

from PIL import Image

from .config import get_settings

_TRANSCRIBE_PROMPT = (
    "Transcribe the handwritten or printed text in this image exactly as written, "
    "including any spelling or grammar mistakes -- do not correct anything. "
    "Return only the transcribed text, with line breaks matching the original. "
    "No commentary, no markdown formatting, no quotation marks around the text."
)


class OCRError(RuntimeError):
    pass


def _image_to_jpeg_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _call_glm(image_bytes: bytes) -> str:
    from openai import OpenAI

    settings = get_settings()
    if not settings.glm_api_key:
        raise OCRError("GLM_API_KEY is not configured")

    client = OpenAI(api_key=settings.glm_api_key, base_url="https://api.z.ai/api/paas/v4/")
    data_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")

    last_err: Exception | None = None
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=settings.ocr_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": _TRANSCRIBE_PROMPT},
                        ],
                    }
                ],
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            time.sleep(5 * (attempt + 1))
    raise OCRError(f"GLM OCR failed after retries: {last_err}")


def _call_gemini(image_bytes: bytes) -> str:
    from google import genai
    from google.genai import types

    settings = get_settings()
    if not settings.gemini_api_key:
        raise OCRError("GEMINI_API_KEY is not configured")

    client = genai.Client(api_key=settings.gemini_api_key)

    last_err: Exception | None = None
    for attempt in range(4):
        try:
            resp = client.models.generate_content(
                model=settings.gemini_model,
                contents=[
                    _TRANSCRIBE_PROMPT,
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                ],
            )
            return resp.text.strip()
        except Exception as e:
            last_err = e
            time.sleep(5 * (attempt + 1))
    raise OCRError(f"Gemini OCR failed after retries: {last_err}")


_CALLERS = {"glm": _call_glm, "gemini": _call_gemini}


def extract_text_from_page(image: Image.Image) -> str:
    settings = get_settings()
    caller = _CALLERS.get(settings.ocr_provider)
    if caller is None:
        raise OCRError(f"Unknown ocr_provider: {settings.ocr_provider}")
    return caller(_image_to_jpeg_bytes(image))
