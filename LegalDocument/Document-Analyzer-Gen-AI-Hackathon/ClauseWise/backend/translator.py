# backend/translator.py
import logging
from functools import lru_cache
from typing import List
from transformers import pipeline, M2M100ForConditionalGeneration, M2M100Tokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from utils import detect_language
import math

logger = logging.getLogger("clausewise.translator")
logger.setLevel(logging.INFO)

# Languages we consider "Indic" (prefer AI4Bharat Indic models)
_INDIC_LANGS = {"hi", "ta", "te", "bn", "ml", "kn", "mr", "gu", "or", "pa", "as", "sd", "ne"}

def _normalize_lang(code: str) -> str:
    if not code:
        return code
    code = code.lower()
    mapping = {
        "zh-cn": "zh", "zh-tw": "zh",
        "pt-br": "pt", "pt-pt": "pt",
        # some common aliases
        "telugu": "te", "tamil": "ta", "hindi": "hi",
        "english": "en", "spanish": "es", "portuguese": "pt"
    }
    if code in mapping:
        return mapping[code]
    return code.split("-")[0]

def _chunk_text(text: str, max_chars: int = 3000) -> List[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        return [text]
    chunks = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks

# ------------- Helsinki/Marian wrapper -------------
@lru_cache(maxsize=128)
def _get_helsinki_pipeline(src: str, tgt: str):
    model_id = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    logger.info("Loading Helsinki pipeline %s", model_id)
    return pipeline("translation", model=model_id, truncation=True)

def _translate_via_pipeline(pipeline_obj, text: str) -> str:
    chunks = _chunk_text(text, 3000)
    out = []
    for c in chunks:
        try:
            r = pipeline_obj(c)
            if isinstance(r, list) and r and "translation_text" in r[0]:
                out.append(r[0]["translation_text"])
            elif isinstance(r, str):
                out.append(r)
            else:
                out.append(str(r))
        except Exception as e:
            logger.exception("Pipeline chunk failed: %s", e)
            out.append(c)
    return "\n\n".join(out)

# ------------- M2M100 fallback -------------
@lru_cache(maxsize=2)
def _get_m2m100(model_name: str = "facebook/m2m100_418M"):
    logger.info("Loading M2M100 %s", model_name)
    tok = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    return tok, model

def _translate_with_m2m100(text: str, src: str, tgt: str) -> str:
    tok, model = _get_m2m100()
    # M2M100 expects language codes matching its tokenizer; tokenizer.lang_code_to_id shows supported codes
    # Try several common variants for tgt and src (e.g., 'te', 'tel', 'te_IN')
    def pick_lang_code(code):
        candidates = [code, code.split("-")[0]]
        # some tokenizers may use 2-letter only; try lower-case
        for c in candidates:
            if c in tok.lang_code_to_id:
                return c
        # try uppercase variants or FLORES-like codes
        for c in list(tok.lang_code_to_id.keys()):
            if c.lower().startswith(code.lower()):
                return c
        return None

    src_code = pick_lang_code(src)
    tgt_code = pick_lang_code(tgt)
    if src_code is None or tgt_code is None:
        raise ValueError(f"M2M100: unsupported codes src={src}({src_code}) tgt={tgt}({tgt_code})")

    logger.info("M2M100 using src=%s tgt=%s", src_code, tgt_code)
    tok.src_lang = src_code
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=2048)
    forced_bos_token_id = tok.get_lang_id(tgt_code)
    generated_tokens = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_new_tokens=1024)
    out = tok.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return out

# ------------- AI4Bharat IndicTrans2 helpers -------------
# NOTE: IndicTrans2 models are available on HF: ai4bharat/indictrans2-en-indic-1B and ai4bharat/indictrans2-indic-en-1B
# They use FLORES-like language codes (e.g. eng_Latn, hin_Deva, tam_Taml, tel_Telu)
# For a robust production integration follow AI4Bharat's repo and use IndicProcessor; here we implement a simple wrapper.

@lru_cache(maxsize=8)
def _load_ai4bharat_model(model_id: str):
    logger.info("Loading AI4Bharat model %s", model_id)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return tok, model

# helper mapping from simple code to IndicTrans2 FLORES-like codes used in some AI4Bharat models
_INDIC_FLORES = {
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "bn": "ben_Beng",
    "ml": "mal_Mlym",
    "kn": "kan_Knda",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "or": "ori_Orya",
    "pa": "pan_Guru",
    "as": "asm_Beng"
}

def _translate_via_ai4bharat(text: str, src: str, tgt: str) -> str:
    """
    Use ai4bharat/indictrans2 models where available.
    This function attempts to load en->indic or indic->en models and use tokenizer/model generate.
    """
    # only attempt if either src or tgt is in _INDIC_FLORES
    model_id = None
    src_norm = src
    tgt_norm = tgt
    # if src is en and tgt is indic: use en-indic model
    if src == "en" and tgt in _INDIC_FLORES:
        model_id = "ai4bharat/indictrans2-en-indic-1B"
        tgt_f = _INDIC_FLORES[tgt]
        src_f = "eng_Latn"
    # if tgt is en and src is indic: use indic-en
    elif tgt == "en" and src in _INDIC_FLORES:
        model_id = "ai4bharat/indictrans2-indic-en-1B"
        src_f = _INDIC_FLORES[src]
        tgt_f = "eng_Latn"
    else:
        raise ValueError("IndicTrans2 not applicable for given src/tgt")

    try:
        tok, model = _load_ai4bharat_model(model_id)
        # some AI4Bharat models require passing forced_bos_token_id or encoding the language tags in inputs
        # We attempt a simple approach: include a special token at start indicating languages if tokenizer has it,
        # otherwise rely on model defaults. This is heuristic-y; see AI4Bharat docs for robust approach.
        prompt = text
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
        # try to set forced_bos if token available
        forced_bos = None
        if hasattr(tok, "lang_code_to_id") and tgt_f in tok.lang_code_to_id:
            forced_bos = tok.lang_code_to_id[tgt_f]
        generated = model.generate(**inputs, forced_bos_token_id=forced_bos, max_new_tokens=1024) if forced_bos else model.generate(**inputs, max_new_tokens=1024)
        out = tok.batch_decode(generated, skip_special_tokens=True)[0]
        return out
    except Exception as e:
        logger.exception("AI4Bharat translation failed: %s", e)
        raise

# ------------- High level translate function -------------
def translate_text_if_requested(text: str, src_lang: str = "auto", tgt_lang: str = "en", plain_english: bool = False) -> str:
    """
    Multi-tier translation:
      1) detect source if auto
      2) If src==tgt return input
      3) Try Helsinki direct opus-mt-{src}-{tgt}
      4) If indic language involved, try AI4Bharat IndicTrans2 (en<->indic)
      5) Try Helsinki pivot via English (src->en then en->tgt)
      6) Try M2M100 fallback
      7) Try NLLB (if available)
      8) Fallback: return original text
    """
    if not text:
        return text

    # detect
    if src_lang == "auto":
        detected = detect_language(text)
        logger.info("Auto-detected language: %s", detected)
        src = _normalize_lang(detected)
    else:
        src = _normalize_lang(src_lang)
    tgt = _normalize_lang(tgt_lang)

    logger.info("Requested translation %s -> %s (plain_english=%s)", src, tgt, plain_english)

    if src == tgt:
        logger.info("Source and target equal -> returning original")
        return text

    # 1) Try Helsinki direct
    try:
        logger.info("Attempting direct Helsinki %s -> %s", src, tgt)
        pipe = _get_helsinki_pipeline(src, tgt)
        return _translate_via_pipeline(pipe, text)
    except Exception as e:
        logger.warning("Direct Helsinki %s->%s failed: %s", src, tgt, e)

    # 2) If Indic involvement, try AI4Bharat
    try:
        if (src in _INDIC_LANGS) or (tgt in _INDIC_LANGS):
            # prefer ai4bharat for en<->indic; otherwise skip
            if (src == "en" and tgt in _INDIC_FLORES) or (tgt == "en" and src in _INDIC_FLORES):
                logger.info("Attempting AI4Bharat IndicTrans2 fallback for %s<->%s", src, tgt)
                return _translate_via_ai4bharat(text, src, tgt)
    except Exception as e:
        logger.warning("AI4Bharat attempt failed: %s", e)

    # 3) Try pivot via English using Helsinki
    try:
        if src != "en":
            logger.info("Attempting Helsinki pivot: %s -> en", src)
            pipe1 = _get_helsinki_pipeline(src, "en")
            text_en = _translate_via_pipeline(pipe1, text)
        else:
            text_en = text

        if tgt == "en":
            return text_en

        logger.info("Attempting Helsinki pivot en -> %s", tgt)
        pipe2 = _get_helsinki_pipeline("en", tgt)
        return _translate_via_pipeline(pipe2, text_en)
    except Exception as e:
        logger.warning("Helsinki pivot failed: %s", e)

    # 4) M2M100 fallback
    try:
        logger.info("Attempting M2M100 fallback %s -> %s", src, tgt)
        return _translate_with_m2m100(text, src, tgt)
    except Exception as e:
        logger.warning("M2M100 fallback failed: %s", e)

    # 5) Optionally: NLLB fallback (heavy). We don't include full code here to avoid extra deps.
    #    If you want NLLB fallback, add code to call facebook/nllb-200 models with FLORES codes.

    logger.warning("All translation attempts failed for %s->%s. Returning original text.", src, tgt)
    return text

