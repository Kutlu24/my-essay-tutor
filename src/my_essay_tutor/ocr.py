"""Handwriting/print text extraction via TrOCR, hosted on Hugging Face's
free Inference API.

Of the three open-source HTR models considered (Kraken, PyLaia, TrOCR),
TrOCR is the one deployed here: it is the only one with a ready, free
hosted-inference path (no local model weights, no local torch process),
which matters because this app targets Render's 512MB free tier -- a
budget that has already failed once this session under a locally-loaded
ML model. Kraken and PyLaia have no equivalent hosted endpoint and expect
a model checkpoint fine-tuned for the target handwriting/script, so they
are left as a documented upgrade path (see segmentation.py) rather than
built in for v1.

Known limitation, stated plainly: TrOCR's public "handwritten" checkpoint
is trained on the IAM dataset, which is English. Accuracy on German/French
handwriting will be visibly weaker than on English until a multilingual or
per-language fine-tuned checkpoint is swapped in via HF_TROCR_MODEL.
"""

from __future__ import annotations

import io
import time

import httpx
from PIL import Image

from .config import get_settings
from .segmentation import segment_lines

_HF_URL = "https://api-inference.huggingface.co/models/{model}"
_MAX_LINES_PER_PAGE = 60


class OCRError(RuntimeError):
    pass


def _call_trocr(image_bytes: bytes) -> str:
    settings = get_settings()
    if not settings.hf_api_token:
        raise OCRError(
            "HF_API_TOKEN is not configured. Create a free token at "
            "https://huggingface.co/settings/tokens and set it in .env"
        )

    url = _HF_URL.format(model=settings.hf_trocr_model)
    headers = {"Authorization": f"Bearer {settings.hf_api_token}"}

    for attempt in range(4):
        resp = httpx.post(url, headers=headers, content=image_bytes, timeout=60.0)
        if resp.status_code == 503:
            # Model is cold-starting on HF's side; this is expected on first use.
            time.sleep(min(5 * (attempt + 1), 20))
            continue
        if resp.status_code == 429:
            time.sleep(min(5 * (attempt + 1), 20))
            continue
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        raise OCRError(f"Unexpected OCR response shape: {data}")

    raise OCRError("OCR model did not become ready in time, please retry")


def extract_text_from_page(image: Image.Image) -> str:
    lines = segment_lines(image)[:_MAX_LINES_PER_PAGE]

    texts: list[str] = []
    for i, line_img in enumerate(lines):
        buf = io.BytesIO()
        line_img.convert("RGB").save(buf, format="JPEG", quality=92)
        text = _call_trocr(buf.getvalue())
        if text:
            texts.append(text)
        if i < len(lines) - 1:
            time.sleep(0.2)  # stay well under HF's free-tier rate cap

    return "\n".join(texts)
