# ClauseWise — Legal Document Analyzer

## Overview
ClauseWise ingests legal documents (PDF/DOCX/TXT), detects language, applies OCR for scanned PDFs, extracts clause-level segments, performs multilingual NER, zero-shot document classification, and produces a multilingual summary (text + optional audio). Implementation uses Hugging Face pipelines, EasyOCR and an optional IBM Granite wrapper.

---

## Quick local setup (recommended Python 3.10 / 3.11)

1. Install Python 3.10 (or 3.11). Do NOT use 3.13 for this stack.
2. Create a virtual environment:
   ```bash
   # Windows
   py -3.10 -m venv .clausevenv
   .clausevenv\Scripts\activate
   
