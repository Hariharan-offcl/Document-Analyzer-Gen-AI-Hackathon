# backend/test_translations.py
from translator import translate_text_if_requested
tests = [
    ("This is a short test summary of a contract. It contains clauses about payment, termination and liability.", "en", "es"),
    ("This is a short test summary of a contract. It contains clauses about payment, termination and liability.", "en", "pt"),
    ("This is a short test summary of a contract. It contains clauses about payment, termination and liability.", "en", "hi"),
    ("This is a short test summary of a contract. It contains clauses about payment, termination and liability.", "en", "ta"),
    ("This is a short test summary of a contract. It contains clauses about payment, termination and liability.", "en", "te"),
]
for text, s, t in tests:
    print("========================================")
    print("to", t)
    try:
        out = translate_text_if_requested(text, src_lang=s, tgt_lang=t, plain_english=False)
    except Exception as e:
        out = f"Error: {e}"
    print(out)
