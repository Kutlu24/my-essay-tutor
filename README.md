# My Essay Tutor

Students upload a composition (photo or PDF, handwritten or typed) in German, English or French, pick a target CEFR level (A1&ndash;C1), and get back a transcription, a list of grammar/spelling errors with corrections, and a level score.

## Architecture

```
frontend/index.html  (upload -> review transcription -> results)
        |
        v
FastAPI backend (src/my_essay_tutor/api/app.py)
        |
        +--> ocr.py        text extraction (TrOCR via Hugging Face's free Inference API)
        |      segmentation.py   classical line-splitting so multi-line pages don't
        |                        get fed to TrOCR as one blob (see docstring for why)
        |      pdf_utils.py      PDF page -> image (pypdfium2)
        |
        +--> grading.py     grammar-error + CEFR scoring via an LLM (GLM by default)
```

The same extraction/grading logic is also exposed as a standalone **MCP server**
(`src/my_essay_tutor/mcp_server.py`, `python -m my_essay_tutor.mcp_server`), so any
MCP client (Claude Desktop, another agent) can call `extract_essay_text` and
`grade_essay` directly. The web backend calls the same functions in-process for the
student-facing flow, rather than round-tripping through a spawned MCP subprocess per
request &mdash; two consumption paths, one implementation.

### Why TrOCR (not Kraken or PyLaia)

Of the three open-source HTR models considered, only TrOCR has a free **hosted**
inference path (Hugging Face's Inference API) with no local model weights. Kraken and
PyLaia would need a resident pytorch process on the web server &mdash; the same kind of
local ML load that already OOM-killed the DSG Compliance project on Render's 512MB
free tier. TrOCR's tradeoff: its public checkpoint is trained on English handwriting
(IAM), so accuracy on German/French will be visibly weaker until a multilingual or
per-language fine-tuned checkpoint is swapped in via `HF_TROCR_MODEL`.

## Setup

```
pip install -e .
cp .env.example .env
```

Fill in `.env`:
- `GLM_API_KEY` &mdash; grading LLM (z.ai)
- `HF_API_TOKEN` &mdash; free token from https://huggingface.co/settings/tokens, used for TrOCR extraction

## Run locally

```
uvicorn my_essay_tutor.api.app:app --reload
```

Open http://127.0.0.1:8000/

## Deploy

`render.yaml` is a Render Blueprint. Push to GitHub, create a Blueprint instance on
Render pointing at the repo, then set `GLM_API_KEY` and `HF_API_TOKEN` in the
service's Environment tab (marked `sync: false` so they aren't committed).

## Known limitations (v1)

- Line segmentation is a classical ink-density heuristic, not a learned segmenter
  (see `segmentation.py`). Works well for neatly-spaced handwriting; degrades on
  heavily slanted or overlapping lines.
- TrOCR's handwriting accuracy is English-centric; German/French handwritten input
  should be reviewed carefully at the "review transcription" step before grading.
- Typed/printed compositions (scanned or exported PDF) generally transcribe far more
  reliably than handwriting, since they don't depend on cursive/print handwriting
  recognition at all.
