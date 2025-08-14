# backend/classifier.py
import logging
from functools import lru_cache
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict

logger = logging.getLogger("clausewise.classifier")

CANDIDATE_LABELS = [
    "Service Agreement", "NDA", "Employment Contract", "Lease Agreement",
    "Purchase Agreement", "Privacy Policy", "Terms of Service", "Loan Agreement", "Other"
]

@lru_cache()
def get_zeroshot():
    """
    Attempt to load the multilingual zero-shot model first (joeddav/xlm-roberta-large-xnli).
    If that fails due to tokenizer/model issues (SentencePiece, etc.), fallback to
    facebook/bart-large-mnli (English-only) to keep classification working.
    """
    try:
        model_id = "joeddav/xlm-roberta-large-xnli"
        logger.info("Loading zero-shot model '%s' (multilingual)...", model_id)
        # Load tokenizer and model explicitly (avoid transformers trying slow->fast conversion)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
            pipe = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)
            logger.info("Loaded multilingual zero-shot model.")
            return pipe
        except Exception as e:
            logger.warning("Explicit load of multilingual model failed: %s", e)
            # fall-through to english fallback
    except Exception as e:
        logger.exception("Unexpected error when preparing multilingual model: %s", e)

    # Fallback (English)
    fallback = "facebook/bart-large-mnli"
    logger.info("Falling back to %s for zero-shot classification (English only).", fallback)
    pipe = pipeline("zero-shot-classification", model=fallback)
    return pipe


def classify_document(text: str) -> Dict:
    try:
        pipe = get_zeroshot()
        out = pipe(text[:2000], candidate_labels=CANDIDATE_LABELS)
        return {"label": out["labels"][0], "scores": dict(zip(out["labels"], out["scores"]))}
    except Exception as e:
        logger.exception("Document classification failed: %s", e)
        return {"label": "Unknown", "scores": {}}
