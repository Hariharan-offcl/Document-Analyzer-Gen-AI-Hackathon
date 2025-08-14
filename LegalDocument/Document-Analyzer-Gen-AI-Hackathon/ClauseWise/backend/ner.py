# backend/ner.py
from transformers import pipeline
from functools import lru_cache

@lru_cache()
def get_ner_pipeline():
    # multilingual NER (XLM-R based) - Davlan model is lightweight and multilingual
    return pipeline("token-classification", model="Davlan/xlm-roberta-base-ner-hrl", aggregation_strategy="simple")

def run_ner(text: str):
    try:
        pipe = get_ner_pipeline()
        results = pipe(text[:4000])
        out = [{"text": r["word"], "type": r.get("entity_group") or r.get("entity"), "score": float(r.get("score", 0)), "start": r.get("start"), "end": r.get("end")} for r in results]
        return out
    except Exception as e:
        return []
