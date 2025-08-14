# backend/langdetect_utils.py
from langdetect import detect_langs
import fasttext
import os

_fasttext_model = None
FASTTEXT_MODEL_PATH = "lid.176.ftz"  # will download if needed

def _load_fasttext():
    global _fasttext_model
    if _fasttext_model is None:
        if not os.path.exists(FASTTEXT_MODEL_PATH):
            # user should download from https://fasttext.cc/docs/en/language-identification.html
            raise FileNotFoundError("FastText model not found. Download lid.176.ftz and place beside this file.")
        _fasttext_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
    return _fasttext_model

def detect_language(text: str) -> str:
    try:
        # try langdetect quick
        langs = detect_langs(text[:2000])
        if langs:
            return langs[0].lang
    except Exception:
        pass
    try:
        m = _load_fasttext()
        pred = m.predict(text.replace("\n", " "), k=1)
        lang = pred[0][0].replace("__label__", "")
        return lang
    except Exception:
        return "en"  # fallback
