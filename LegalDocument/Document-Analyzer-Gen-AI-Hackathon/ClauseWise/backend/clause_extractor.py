# backend/clause_extractor.py
from utils import extract_candidate_clauses
from simplifier import summarize_text
from classifier import get_zeroshot
from functools import lru_cache
from transformers import pipeline

@lru_cache()
def get_clause_classifier():
    # optional: small classifier; we'll use zero-shot approach for clause-type detection
    return pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")

CLAUSE_LABELS = ["Confidentiality", "Termination", "Payment", "Liability", "Governing Law", "Assignment", "Other"]

def extract_clauses(text: str, lang="en"):
    candidates = extract_candidate_clauses(text)
    # For each candidate, run a short summarization and zero-shot to label type
    cc = get_clause_classifier()
    out = []
    for c in candidates:
        short = c["text"][:1500]
        # summarise clause to produce short summary
        summary = summarize_text(short, input_lang=lang, output_lang=lang, simplify=True)
        try:
            z = cc(short, candidate_labels=CLAUSE_LABELS)
            clause_type = z["labels"][0]
        except Exception:
            clause_type = "Other"
        # simple risk heuristic
        risk = "low"
        lowtext = short.lower()
        if any(k in lowtext for k in ["penalty", "breach", "liable", "indemnify", "forfeit", "injunction"]):
            risk = "high"
        elif any(k in lowtext for k in ["notice", "30 days", "termination", "renew"]):
            risk = "medium"
        out.append({"title": c.get("title"), "text": c.get("text"), "summary": summary, "type": clause_type, "risk": risk})
    return out
