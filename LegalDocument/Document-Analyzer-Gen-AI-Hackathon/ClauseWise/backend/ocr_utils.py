# backend/ocr_utils.py
import easyocr
from pdf2image import convert_from_path
import tempfile
import os

reader = easyocr.Reader(['en'], gpu=False)  # we will reinitialize per-language at runtime

def ocr_image_pdf(pdf_path: str, langs=None) -> str:
    # Convert pdf -> images
    langs = langs or ['en']
    # reinit reader for languages if needed
    global reader
    try:
        reader = easyocr.Reader(langs, gpu=False)
    except Exception:
        # fallback: keep default
        pass

    pages = convert_from_path(pdf_path, dpi=200)
    text_chunks = []
    for p in pages:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            p.save(tmp.name, "PNG")
            res = reader.readtext(tmp.name, detail=0)
            text_chunks.append("\n".join(res))
            os.remove(tmp.name)
    return "\n\n".join(text_chunks)
