import pypdfium2 as pdfium
from PIL import Image

MAX_PAGE_PIXELS = 20_000_000  # per rendered page; an A4 page at scale 2 is about 2 MP


def page_scale(width_pt: float, height_pt: float, scale: float, max_pixels: int = MAX_PAGE_PIXELS) -> float:
    """A PDF page can declare any size (a page of 200 x 200 inches is a few bytes of text): the render scale is lowered so
    the bitmap never exceeds max_pixels, whatever the page claims."""
    area = max(1.0, width_pt * height_pt)
    return min(scale, (max_pixels / area) ** 0.5)


def pdf_to_images(pdf_bytes: bytes, max_pages: int = 5, scale: float = 2.0) -> list[Image.Image]:
    pdf = pdfium.PdfDocument(pdf_bytes)
    images: list[Image.Image] = []
    try:
        for i in range(min(len(pdf), max_pages)):
            page = pdf[i]
            w, h = page.get_size()
            bitmap = page.render(scale=page_scale(w, h, scale))
            images.append(bitmap.to_pil().convert("RGB"))
    finally:
        pdf.close()
    return images
