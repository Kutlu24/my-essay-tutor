# My Essay Tutor

Students upload a composition (photo or PDF, handwritten or typed) in German, English or French, pick a target CEFR level (A1&ndash;C1), and get back a transcription, a list of grammar/spelling errors with corrections, and a level score.

## Architecture

```
frontend/index.html  (upload -> review transcription -> results)
        |
        v
FastAPI backend (src/my_essay_tutor/api/app.py)
        |
        +--> ocr.py        text extraction via a vision LLM (GLM by default, Gemini as
        |                  an alternative -- see docstring for why this isn't TrOCR/Kraken/PyLaia)
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

### Why a vision LLM, not TrOCR/Kraken/PyLaia

v1 of this project used TrOCR through Hugging Face's free Inference API, chosen
because it was the only one of the three open-source HTR models under consideration
with a free **hosted** path (no local model weights on the web server &mdash; Kraken
and PyLaia would need a resident pytorch process, the same kind of local ML load that
already OOM-killed the DSG Compliance project on Render's 512MB free tier).

That path closed: HF's free serverless inference tier was cut down to a $0.10/month
credit allowance in 2026, not viable for real traffic. `ocr.py` now sends the essay
page directly to a vision-capable LLM instead of a dedicated HTR model. This keeps the
original property that mattered (no local model weights, no OOM risk), with no
separate line-segmentation step needed since a vision LLM reads a full page at once
rather than one line at a time.

Extraction defaults to **Gemini** (`OCR_PROVIDER=gemini`), not GLM: confirmed live
that GLM's `glm-4.5v` vision model needs a paid GLM resource package even on a key
where GLM's text models (grading) work fine on the free tier -- it fails with
`"Insufficient balance or no resource package"` (error 1113). Gemini's vision models
are covered by the same free-tier key already used elsewhere this session. Set
`OCR_PROVIDER=glm` instead if the GLM account is topped up.

## Validated grading output, and an independent grammar cross-check (opt-in)

`grading.py` used to parse each provider's grading response with a hand-rolled ```json-fence stripper and no schema validation - a malformed or incomplete response just crashed with a raw `json.loads` error. It now wraps every provider (GLM, Gemini, Anthropic) with [`instructor`](https://github.com/instructor-ai/instructor), so `response_model=_GradingResponse` returns an already-validated Pydantic object, with `instructor`'s own retry-with-the-validation-error-fed-back loop kicking in if a response doesn't match. Verified against the real GLM and Gemini APIs (2026-09-25). One real compatibility gap found and fixed along the way: GLM's default tool-calling mode returned the nested `criteria` field as a JSON-encoded *string* instead of an object, which failed validation outright - the GLM path now uses `instructor.Mode.JSON` instead of the default `Mode.TOOLS` to avoid it. The Gemini path needs the `instructor[google-genai]` extra (pulls in `jsonref`) for structured output to work at all.

A second, independent, non-LLM grammar cross-check via [`language_tool_python`](https://github.com/jxmorris12/language_tool_python) also now exists in `grammar_check.py`, since the grading pipeline is otherwise one LLM call asked to both find every error and score the essay - nothing else checks whether that error list is complete or correct. **Off by default** (`ENABLE_GRAMMAR_CROSSCHECK=true` to turn it on) - it's a ready-to-use capability, not yet wired to influence grading itself; when enabled, `GradingResult.grammar_crosscheck` carries LanguageTool's own match list alongside (not merged into) the LLM's `grammar_errors`. **Uses `language_tool_python`'s hosted Public API by default, never its local-server mode automatically** - the local mode spins up a resident Java process, exactly the kind of local ML/JVM load this project's own OCR redesign (see "Why a vision LLM" above) was chosen specifically to avoid after DSG Compliance was OOM-killed on Render's free tier. The Public API's shared free-tier rate limit means real-world use should expect occasional `RateLimitError`s - not wired to retry automatically yet.

A second, separate opt-in - `GRAMMAR_CROSSCHECK_LOCAL_SERVER=true` - switches to the local-server constructor instead, but only takes effect once `ENABLE_GRAMMAR_CROSSCHECK` is already on (flipping this alone does nothing). Included now, off by default, for a future deploy target with real memory headroom - **do not enable this on Render's free tier**, that's the exact failure mode the Public API default exists to avoid.

## Setup

```
pip install -e .
cp .env.example .env
```

Fill in `.env`:
- `GLM_API_KEY` &mdash; grading (z.ai); also works for extraction if `OCR_PROVIDER=glm` and the account has a resource package for vision
- `GEMINI_API_KEY` &mdash; extraction (default `OCR_PROVIDER=gemini`)

## Run locally

```
uvicorn my_essay_tutor.api.app:app --reload
```

Open http://127.0.0.1:8000/

## Deploy

`render.yaml` is a Render Blueprint. Push to GitHub, create a Blueprint instance on
Render pointing at the repo, then set `GLM_API_KEY` and `GEMINI_API_KEY` in the
service's Environment tab (marked `sync: false` so they aren't committed).

## Known limitations (v1)

- Handwriting transcription is never perfect from any engine; the frontend has a
  "review transcription" step before grading for exactly this reason.
- Typed/printed compositions (scanned or exported PDF) generally transcribe more
  reliably than handwriting.
