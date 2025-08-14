# backend/download_models.py
from translator import _load_pipeline_for_model, _get_m2m100
from classifier import get_zeroshot
from simplifier import _get_local_summarizer
import logging

logging.basicConfig(level=logging.INFO)
models = [
    "Helsinki-NLP/opus-mt-en-es",
    "Helsinki-NLP/opus-mt-es-en",
    "Helsinki-NLP/opus-mt-en-pt",
    "Helsinki-NLP/opus-mt-pt-en",
    "Helsinki-NLP/opus-mt-en-hi",
    "Helsinki-NLP/opus-mt-hi-en",
    "Helsinki-NLP/opus-mt-en-ta",
    "Helsinki-NLP/opus-mt-ta-en",
    "Helsinki-NLP/opus-mt-en-te",
    "Helsinki-NLP/opus-mt-te-en",
]

for m in models:
    try:
        print("Loading", m)
        _load_pipeline_for_model(m)
    except Exception as e:
        print("Could not load", m, e)

print("Loading m2m100 fallback (may take a while)")
try:
    _get_m2m100()
except Exception as e:
    print("m2m100 fail:", e)

print("Loading zero-shot classifier and summarizer")
get_zeroshot()
_get_local_summarizer()
print("Done.")
