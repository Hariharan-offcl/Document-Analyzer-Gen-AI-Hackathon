# backend/file_handler.py
import io, os, tempfile
from typing import Tuple, Dict, Any
import pdfplumber
import fitz  # PyMuPDF
import docx
from PIL import Image
import easyocr
from utils import detect_language
from functools import lru_cache
import logging

logger = logging.getLogger("clausewise.file_handler")

# preload EasyOCR reader for multiple languages (CPU)
_EASY_READER = None

def _get_easy_reader(lang_list=None):
    global _EASY_READER
    if _EASY_READER is None:
        # default languages: add more as needed
        langs = lang_list or ['en', 'es', 'pt', 'fr', 'de', 'hi', 'ta', 'te']
        _EASY_READER = easyocr.Reader(langs, gpu=False)
    return _EASY_READER

def _extract_pdf_text(pdf_bytes: bytes) -> str:
    # Try text extraction first
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages_text = [p.extract_text() or "" for p in pdf.pages]
            text = "\n\n".join(pages_text).strip()
            if text:
                return text
    except Exception:
        logger.debug("pdfplumber extraction failed, will try OCR fallback", exc_info=True)
    # fallback to OCR via pymupdf images + easyocr
    text_chunks = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    reader = _get_easy_reader()
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            img.save(tmp.name)
            res = reader.readtext(tmp.name, detail=0, paragraph=True)
            text_chunks.append("\n".join(res))
        finally:
            tmp.close()
            try:
                os.remove(tmp.name)
            except Exception:
                pass
    return "\n\n".join(text_chunks).strip()

def _extract_docx_text(doc_bytes: bytes) -> str:
    f = io.BytesIO(doc_bytes)
    doc = docx.Document(f)
    return "\n\n".join([p.text for p in doc.paragraphs]).strip()

def _extract_txt_text(txt_bytes: bytes) -> str:
    try:
        return txt_bytes.decode("utf-8")
    except:
        return txt_bytes.decode("latin-1", errors="ignore")

# -------- ASR pipeline for audio transcription ----------
from transformers import pipeline as hf_pipeline

@lru_cache()
def get_asr_pipeline(model_name: str = "openai/whisper-small"):
    """
    Cache an ASR pipeline instance.
    Warning: downloads model on first call and uses CPU by default; adjust for GPU in production.
    """
    logger.info("Loading ASR pipeline: %s", model_name)
    return hf_pipeline("automatic-speech-recognition", model=model_name, chunk_length_s=30)

def _extract_audio_text(audio_bytes: bytes, filename_hint: str = "audio.wav") -> str:
    """
    Save incoming bytes to a temp file, run ASR pipeline and return transcription text.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename_hint)[1] or ".wav", delete=False)
    try:
        tmp.write(audio_bytes)
        tmp.flush()
        tmp.close()
        asr = get_asr_pipeline()
        # some pipelines accept file path or raw array; pass the file path for robustness
        res = asr(tmp.name)
        # res may be dict with 'text' field
        if isinstance(res, dict) and "text" in res:
            return res["text"]
        # if pipeline returns string
        if isinstance(res, str):
            return res
        # else try to join transcripts
        if isinstance(res, list):
            return " ".join((r.get("text") or str(r) for r in res))
        return str(res)
    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass

# ----------------------------------------

def extract_text_from_file(filename: str, content: bytes) -> Dict[str, Any]:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        text = _extract_pdf_text(content)
    elif lower.endswith(".docx"):
        text = _extract_docx_text(content)
    elif lower.endswith(".txt") or lower.endswith(".md"):
        text = _extract_txt_text(content)
    elif lower.endswith((".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus")):
        # treat audio files: transcribe to text
        try:
            text = _extract_audio_text(content, filename_hint=filename)
        except Exception as e:
            logger.exception("Audio transcription failed: %s", e)
            raise
    else:
        raise ValueError("Unsupported file type: " + filename)
    lang = detect_language((text or "")[:4000]) if text else "en"
    return {"text": text, "lang": lang}
