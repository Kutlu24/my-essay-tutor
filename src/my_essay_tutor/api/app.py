import io
import logging
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from .. import exams, grading, ocr, pdf_utils, seo
from ..config import get_settings
from ..guard import Guard
from ..models import GradingResult


def _allowed_origins() -> list[str]:
    """CORS allowlist: ALLOWED_ORIGINS env (comma-separated) plus this
    service's RENDER_EXTERNAL_URL; localhost only when not on Render."""
    origins = [
        o.strip()
        for o in os.environ.get("ALLOWED_ORIGINS", "").split(",")
        if o.strip()
    ]
    render_url = os.environ.get("RENDER_EXTERNAL_URL")
    if render_url and render_url not in origins:
        origins.append(render_url)
    if not render_url and not origins:
        origins = [
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:5173",
        ]
    return origins


app = FastAPI(title="my-essay-tutor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


seo.install(app)


app.add_middleware(Guard)  # rate limits + streamed body caps for /api/extract and /api/grade (see guard.py)
MAX_PIXELS = 40_000_000  # a phone photo is ~12 MP; a small file can still decode to a gigantic bitmap
log = logging.getLogger(__name__)


@app.middleware("http")
async def _security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault(
        "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
    )
    return response

# Frontend lives in the repo checkout during development and inside the
# installed package (my_essay_tutor.frontend) when pip-installed.
_DEV_FRONTEND = Path(__file__).resolve().parents[3] / "frontend"
_PKG_FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
_FRONTEND_DIR = _DEV_FRONTEND if _DEV_FRONTEND.exists() else _PKG_FRONTEND
if _FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="ui")


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse("/ui/")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


class ExtractResponse(BaseModel):
    extracted_text: str
    pages: int


@app.post("/api/extract", response_model=ExtractResponse)
async def extract(file: UploadFile = File(...)) -> ExtractResponse:
    settings = get_settings()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    content = await file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb}MB limit")

    is_pdf = (file.content_type == "application/pdf") or (file.filename or "").lower().endswith(".pdf")

    try:
        if is_pdf:
            pages = pdf_utils.pdf_to_images(content)
        else:
            img = Image.open(io.BytesIO(content))
            if img.width * img.height > MAX_PIXELS:
                raise HTTPException(413, "Image has too many pixels")
            pages = [img.convert("RGB")]
    except HTTPException:
        raise
    except Exception:
        log.exception("could not read upload")  # parser messages can carry paths and internals: log, do not send
        raise HTTPException(400, "Could not read the uploaded file (PDF, JPEG or PNG expected)")

    if not pages:
        raise HTTPException(400, "No pages found in upload")

    page_texts = []
    for page in pages:
        try:
            page_texts.append(ocr.extract_text_from_page(page))
        except ocr.OCRError:
            log.exception("OCR failed")
            raise HTTPException(502, "The text could not be read from the page right now. Please try again, or type the text in.")

    return ExtractResponse(extracted_text="\n\n".join(t for t in page_texts if t), pages=len(pages))


@app.get("/api/config")
def public_config() -> dict:
    """What this deployment offers: essay languages and the name it shows (see Settings)."""
    st = get_settings()
    kicker = st.app_kicker or ("A1 → C1 · telc · Goethe · DELF/DALF · " + " / ".join(x.upper() for x in st.language_list))
    return {"languages": st.language_list, "title": st.app_title, "kicker": kicker}


@app.get("/api/exams")
def list_exams() -> dict:
    """Exam catalog (language -> exams -> levels -> tasks) the frontend builds its pickers from."""
    allowed = get_settings().language_list
    return {lang: items for lang, items in exams.catalog().items() if lang in allowed}


class GradeRequest(BaseModel):
    text: str
    language_code: str = Field(max_length=10)
    target_level: str = Field(max_length=4)
    exam: str | None = Field(default=None, max_length=40)  # see /api/exams; omitted -> per-language default
    task: str | None = Field(default=None, max_length=80)


@app.post("/api/grade", response_model=GradingResult)
def grade(req: GradeRequest) -> GradingResult:
    allowed = get_settings().language_list
    if req.language_code not in allowed:
        raise HTTPException(400, "language_code must be one of: " + ", ".join(allowed))
    if req.target_level not in ("A1", "A2", "B1", "B2", "C1"):
        raise HTTPException(400, "target_level must be one of: A1, A2, B1, B2, C1")
    if not req.text.strip():
        raise HTTPException(400, "text is empty")
    if len(req.text) > 20_000:
        raise HTTPException(413, "text exceeds 20,000 characters")
    try:
        exams.resolve(req.language_code, req.exam, req.target_level, req.task)
    except ValueError as e:
        raise HTTPException(400, str(e))

    try:
        return grading.grade_essay(req.text, req.language_code, req.target_level, req.exam, req.task)
    except Exception as e:
        log.exception("grading failed")  # provider errors can echo request details or key fragments
        raise HTTPException(502, f"Grading failed ({type(e).__name__}). Please try again in a moment.")
