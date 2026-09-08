import pypdfium2 as pdfium
from PIL import Image


def pdf_to_images(pdf_bytes: bytes, max_pages: int = 5, scale: float = 2.0) -> list[Image.Image]:
    pdf = pdfium.PdfDocument(pdf_bytes)
    images: list[Image.Image] = []
    for i in range(min(len(pdf), max_pages)):
        bitmap = pdf[i].render(scale=scale)
        images.append(bitmap.to_pil().convert("RGB"))
    pdf.close()
    return images
