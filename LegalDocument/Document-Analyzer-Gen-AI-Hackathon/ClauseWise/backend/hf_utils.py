# backend/hf_utils.py
from transformers import pipeline
from functools import lru_cache

@lru_cache()
def get_summarizer(model_name="facebook/bart-large-cnn"):
    return pipeline("summarization", model=model_name)

@lru_cache()
def get_ner(model_name="xlm-roberta-large-finetuned-conll03-english"):
    return pipeline("ner", model=model_name, aggregation_strategy="simple")

@lru_cache()
def get_translation(model_name):
    return pipeline("translation", model=model_name)

def summarize(text):
    s = get_summarizer()
    out = s(text, max_length=150, min_length=30, do_sample=False)
    return out[0]['summary_text']

def ner(text):
    n = get_ner()
    return n(text)
