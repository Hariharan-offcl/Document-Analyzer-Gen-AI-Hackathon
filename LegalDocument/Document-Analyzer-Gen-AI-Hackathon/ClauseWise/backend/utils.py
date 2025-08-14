# backend/utils.py
from langdetect import detect, DetectorFactory
import re
from typing import List, Dict, Any

DetectorFactory.seed = 0

CLAUSE_HEADING_RE = re.compile(
    r'\b(WHEREAS|GOVERNING LAW|TERMINATION|TERMS? OF PAYMENT|CONFIDENTIALITY|INDEMNITY|LIMITATION|LIABILITY|FORCE MAJEURE|ASSIGNMENT|NOTICES|SIGNATURES|ENTIRE AGREEMENT)\b',
    flags=re.I
)

HF_TRANSLATION_MAP = {
    # self-contained mapping/ fallback; for most pairs we will build model ids dynamically
    # Helsinki models will be used as "Helsinki-NLP/opus-mt-{src}-{tgt}"
}

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"

def split_into_paragraphs(text: str) -> List[str]:
    # similar to chunking for long texts
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        paras = [line.strip() for line in text.splitlines() if line.strip()]
    return paras

def extract_candidate_clauses(text: str) -> List[Dict[str, Any]]:
    # Very simple heuristic: split on headings or paragraphs
    paras = split_into_paragraphs(text)
    clauses = []
    buffer = []
    for p in paras:
        if CLAUSE_HEADING_RE.search(p) and buffer:
            clauses.append({"title": buffer[0][:60], "text": "\n\n".join(buffer)})
            buffer = [p]
        else:
            buffer.append(p)
    if buffer:
        clauses.append({"title": buffer[0][:60], "text": "\n\n".join(buffer)})
    # fallback ensure at least one clause
    if not clauses:
        clauses = [{"title": "Document", "text": text}]
    return clauses
