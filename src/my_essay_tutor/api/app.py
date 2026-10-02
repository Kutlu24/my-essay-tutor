import io
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from .. import grading, ocr, pdf_utils
from ..config import get_settings
from ..models import GradingResult

app = FastAPI(title="my-essay-tutor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    content = await file.read()

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb}MB limit")

    is_pdf = (file.content_type == "application/pdf") or (file.filename or "").lower().endswith(".pdf")

    try:
        if is_pdf:
            pages = pdf_utils.pdf_to_images(content)
        else:
            pages = [Image.open(io.BytesIO(content)).convert("RGB")]
    except Exception as e:
        raise HTTPException(400, f"Could not read uploaded file: {e}")

    if not pages:
        raise HTTPException(400, "No pages found in upload")

    page_texts = []
    for page in pages:
        try:
            page_texts.append(ocr.extract_text_from_page(page))
        except ocr.OCRError as e:
            raise HTTPException(502, str(e))

    return ExtractResponse(extracted_text="\n\n".join(t for t in page_texts if t), pages=len(pages))


class GradeRequest(BaseModel):
    text: str
    language_code: str
    target_level: str


@app.post("/api/grade", response_model=GradingResult)
def grade(req: GradeRequest) -> GradingResult:
    if req.language_code not in ("de", "en", "fr"):
        raise HTTPException(400, "language_code must be one of: de, en, fr")
    if req.target_level not in ("A1", "A2", "B1", "B2", "C1"):
        raise HTTPException(400, "target_level must be one of: A1, A2, B1, B2, C1")
    if not req.text.strip():
        raise HTTPException(400, "text is empty")

    try:
        return grading.grade_essay(req.text, req.language_code, req.target_level)
    except Exception as e:
        raise HTTPException(502, f"Grading failed: {e}")
