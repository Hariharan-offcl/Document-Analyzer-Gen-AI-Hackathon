# backend/simplifier.py
import logging
from typing import Optional
from translator import translate_text_if_requested
from utils import detect_language
from functools import lru_cache

logger = logging.getLogger("clausewise.simplifier")
logger.setLevel(logging.INFO)

@lru_cache()
def _get_local_summarizer():
    try:
        from transformers import pipeline
        logger.info("Loading local summarizer (sshleifer/distilbart-cnn-12-6)")
        return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception as e:
        logger.warning("Local summarizer unavailable: %s", e)
        return None

def _chunk_text(text:str, max_chars:int=2000):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        return [text]
    chunks=[]
    buf=""
    for p in paras:
        if len(buf)+len(p)+2 <= max_chars:
            buf=(buf+"\n\n"+p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf=p
    if buf:
        chunks.append(buf)
    return chunks

def _local_summarize(text:str)->str:
    summ = _get_local_summarizer()
    if not summ:
        return (text[:1200]+"...") if len(text)>1200 else text
    out=[]
    for ch in _chunk_text(text, max_chars=2000):
        try:
            r = summ(ch, max_length=220, min_length=30, do_sample=False)
            if isinstance(r, list) and "summary_text" in r[0]:
                out.append(r[0]["summary_text"])
            else:
                out.append(str(r))
        except Exception as e:
            logger.warning("Local summarization chunk failed: %s", e)
            out.append(ch[:1000])
    return "\n\n".join(out)

def summarize_text(text: str, input_lang: Optional[str] = "auto", output_lang: str = "en", simplify: bool = True) -> str:
    """
    High-level summarization:
      - detect input language if auto
      - if simplify=True: translate -> summarize in English -> translate summary to output_lang
      - else: direct translate (text -> output_lang) or truncate
    """
    if not text:
        return ""

    if input_lang == "auto" or not input_lang:
        input_lang = detect_language(text)
    logger.info("summarize_text: input_lang=%s, output_lang=%s, simplify=%s", input_lang, output_lang, simplify)

    # If simplify (plain english), create English summary then translate out
    if simplify:
        # 1) translate -> English if needed
        if input_lang != "en":
            try:
                logger.info("Translating source -> en for summarization")
                en_text = translate_text_if_requested(text, src_lang=input_lang, tgt_lang="en", plain_english=False)
            except Exception as e:
                logger.warning("translate->en failed: %s", e)
                en_text = text
        else:
            en_text = text

        # 2) summarize in English
        try:
            summary_en = _local_summarize(en_text)
        except Exception as e:
            logger.warning("Local summarizer failed: %s", e)
            summary_en = en_text if len(en_text) < 1200 else en_text[:1200] + "..."

        # 3) translate summary to output_lang if needed
        if output_lang and output_lang != "en":
            try:
                logger.info("Translating summary en -> %s", output_lang)
                out_summary = translate_text_if_requested(summary_en, src_lang="en", tgt_lang=output_lang, plain_english=False)
                return out_summary
            except Exception as e:
                logger.warning("Translation of summary failed: %s", e)
                return summary_en
        return summary_en

    # Not simplify: direct translation or truncation
    if input_lang == output_lang:
        return text if len(text) < 4000 else text[:4000] + "..."
    try:
        return translate_text_if_requested(text, src_lang=input_lang, tgt_lang=output_lang, plain_english=False)
    except Exception as e:
        logger.warning("Direct translation failed: %s", e)
        return text if len(text) < 4000 else text[:4000] + "..."


