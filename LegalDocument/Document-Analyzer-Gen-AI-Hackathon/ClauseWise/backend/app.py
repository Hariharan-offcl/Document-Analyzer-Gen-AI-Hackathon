# backend/app.py
import base64
import os
import io
import tempfile
import logging
import asyncio
import uuid
import time

from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# Import your project modules (these should exist in backend/)
from file_handler import extract_text_from_file
from utils import detect_language
from clause_extractor import extract_clauses
from ner import run_ner
from classifier import classify_document
from simplifier import summarize_text
from ibm_watson_utils import granite_analyze

# TTS
from gtts import gTTS

# ----- logging -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)
logger = logging.getLogger("clausewise")

# ----- FastAPI app -----
app = FastAPI(title="ClauseWise API (background jobs)")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ----- Job store + concurrency config -----
_jobs: Dict[str, Dict[str, Any]] = {}
MAX_CONCURRENT_JOBS = int(os.getenv("CLAUSEWISE_MAX_WORKERS", "2"))
_worker_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
JOB_TTL_SECONDS = int(os.getenv("CLAUSEWISE_JOB_TTL", str(60 * 60)))  # default 1 hour

# ----- Pydantic response model (for familiarity) -----
class AnalysisResponse(BaseModel):
    lang_detected: str
    doc_type: Dict[str, Any]
    entities: List[Dict[str, Any]]
    clauses: List[Dict[str, Any]]
    summary: str

# ----- small helpers -----
def _coerce_bool(v) -> bool:
    """Robustly coerce form values (which may be 'true'/'false' strings) to bool."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "on")

# ----- gTTS language mapping (add more codes if needed) -----
_GTT_LANGS = {
    "en": "en", "es": "es", "pt": "pt", "hi": "hi", "ta": "ta", "te": "te", "fr": "fr", "de": "de"
}
def _coerce_tts_lang(code: str) -> str:
    """
    Normalize TTS language code to something gTTS supports.
    Fallback to two-letter code and ultimately to 'en'.
    """
    if not code:
        return "en"
    code = code.lower()
    if code in _GTT_LANGS:
        return _GTT_LANGS[code]
    # fallback to two-letter code
    c = code.split("-")[0]
    return _GTT_LANGS.get(c, "en")

# ----- health endpoint -----
@app.get("/health")
async def health():
    return {"status": "ok"}

# ----- warmup at startup (optional) -----
@app.on_event("startup")
async def startup_event():
    """
    Warm up Hugging Face pipelines at server start to reduce first-request latency.
    This runs in executor to avoid blocking the event loop.
    """
    logger.info("Startup: warming up ML pipelines (this may take a while on first run)...")
    loop = asyncio.get_running_loop()
    warmup_tasks = []

    try:
        from classifier import get_zeroshot
        warmup_tasks.append(loop.run_in_executor(None, get_zeroshot))
    except Exception as e:
        logger.warning("Could not queue get_zeroshot warmup: %s", e)

    try:
        from ner import get_ner_pipeline
        warmup_tasks.append(loop.run_in_executor(None, get_ner_pipeline))
    except Exception as e:
        logger.warning("Could not queue get_ner_pipeline warmup: %s", e)

    try:
        from simplifier import _get_local_summarizer, get_summarizer  # supports both approaches
        # attempt to warm up both summarizer entrypoints if present
        if "get_summarizer" in globals():
            warmup_tasks.append(loop.run_in_executor(None, get_summarizer))
        # local summarizer function
        warmup_tasks.append(loop.run_in_executor(None, _get_local_summarizer))
    except Exception:
        pass

    try:
        from clause_extractor import get_clause_classifier
        warmup_tasks.append(loop.run_in_executor(None, get_clause_classifier))
    except Exception:
        pass

    if warmup_tasks:
        try:
            await asyncio.wait_for(asyncio.gather(*warmup_tasks, return_exceptions=True), timeout=300)
            logger.info("Warmup complete (models loaded or attempted).")
        except asyncio.TimeoutError:
            logger.warning("Warmup timed out (some models may still be loading on first request).")
        except Exception as e:
            logger.exception("Unexpected error during warmup: %s", e)
    else:
        logger.info("No warmup tasks queued.")

# ----- Async background worker function -----
async def _process_job(job_id: str, filename: str, content: bytes, output_lang: str, plain_english: bool, use_granite: bool, want_tts: bool = False, tts_lang: str = "en"):
    """
    Process a job in the background:
      - extract text (PDF/DOCX/TXT/Audio)
      - optional Granite analysis (best-effort)
      - classify, NER, clause extraction, summarization
      - optional TTS generation (base64) in requested tts_lang
    """
    logger.info("Job %s: queued, waiting for worker slot...", job_id)
    async with _worker_semaphore:
        logger.info("Job %s: starting processing.", job_id)
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()
        start_time = time.time()
        try:
            # 1) Extract text
            doc = await asyncio.to_thread(extract_text_from_file, filename, content)
            text = doc.get("text", "") or ""
            lang = doc.get("lang", detect_language(text))

            # 2) Optional Granite
            granite_out = None
            if use_granite:
                try:
                    prompt = f"Analyze the following legal document (lang={lang}) and summarize clauses, entities and risk:\n\n{text[:5000]}"
                    granite_out = await asyncio.to_thread(granite_analyze, prompt)
                except Exception as e:
                    logger.warning("Job %s: Granite call failed: %s", job_id, e)

            # 3) HF pipelines (offload to threads)
            doc_type = await asyncio.to_thread(classify_document, text)
            entities = await asyncio.to_thread(run_ner, text)
            clauses = await asyncio.to_thread(extract_clauses, text, lang)

            # ensure summarizer creates summary in the requested output_lang
            summary = await asyncio.to_thread(summarize_text, text, input_lang=lang, output_lang=output_lang, simplify=plain_english)

            result_obj = {
                "lang_detected": lang,
                "doc_type": doc_type,
                "entities": entities,
                "clauses": clauses,
                "summary": summary
            }

            # 4) Generate TTS if requested
            if want_tts:
                try:
                    # choose TTS language: prefer explicit tts_lang, fallback to output_lang
                    chosen_tts_lang = tts_lang or output_lang
                    tts_code = _coerce_tts_lang(chosen_tts_lang)
                    # gTTS supports many languages - ensure provided code is acceptable
                    tts = gTTS(text=summary, lang=tts_code)
                    tmpf = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                    tmpf.close()
                    tts.save(tmpf.name)
                    with open(tmpf.name, "rb") as af:
                        audio_bytes = af.read()
                    # remove tmp file
                    try:
                        os.remove(tmpf.name)
                    except Exception:
                        pass
                    # encode result for JSON transfer
                    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                    result_obj["audio_base64"] = audio_b64
                    result_obj["audio_mime"] = "audio/mpeg"
                    result_obj["audio_filename"] = f"summary_{job_id}.mp3"
                except Exception as e:
                    logger.exception("Job %s: TTS generation failed: %s", job_id, e)
                    result_obj["audio_error"] = str(e)

            # 5) Save results
            _jobs[job_id]["result"] = result_obj
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()
            _jobs[job_id]["expires_at"] = (datetime.utcnow() + timedelta(seconds=JOB_TTL_SECONDS)).isoformat()
            elapsed = time.time() - start_time
            logger.info("Job %s: completed in %.2fs", job_id, elapsed)

        except Exception as e:
            logger.exception("Job %s: processing failed: %s", job_id, e)
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)
            _jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()

# ----- Analyze endpoint (enqueue job) -----
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    output_lang: Optional[str] = Form("en"),
    plain_english: Optional[str] = Form("true"),
    use_granite: Optional[str] = Form("false"),
    want_tts: Optional[str] = Form("false"),
    tts_lang: Optional[str] = Form("en")
):
    """
    Accepts a multipart form file and options; enqueues background job and returns job_id.
    NOTE: boolean form fields may arrive as strings from the frontend; we coerce them.
    """
    content = await file.read()

    # coerce booleans robustly
    plain_english_bool = _coerce_bool(plain_english)
    use_granite_bool = _coerce_bool(use_granite)
    want_tts_bool = _coerce_bool(want_tts)
    tts_lang = tts_lang or "en"
    output_lang = output_lang or "en"

    # create job entry
    job_id = str(uuid.uuid4())
    now = datetime.utcnow()
    _jobs[job_id] = {
        "status": "queued",
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "result": None,
        "error": None,
        "expires_at": (now + timedelta(seconds=JOB_TTL_SECONDS)).isoformat()
    }

    # schedule background processing
    asyncio.create_task(_process_job(job_id, file.filename, content, output_lang, plain_english_bool, use_granite_bool, want_tts_bool, tts_lang))

    return {"job_id": job_id, "status": "queued"}

# ----- Result polling endpoint -----
@app.get("/result/{job_id}")
async def get_result(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    return job

# ----- Jobs listing (convenience) -----
@app.get("/jobs")
async def list_jobs():
    return {"count": len(_jobs), "jobs": {k: {"status": v["status"], "created_at": v["created_at"]} for k, v in _jobs.items()}}

# ----- background cleanup loop -----
async def _cleanup_jobs_loop():
    while True:
        now = datetime.utcnow()
        to_delete = []
        for jid, job in list(_jobs.items()):
            exp = job.get("expires_at")
            if exp:
                try:
                    exp_dt = datetime.fromisoformat(exp)
                    if exp_dt < now:
                        to_delete.append(jid)
                except Exception:
                    pass
        for jid in to_delete:
            logger.info("Cleaning up expired job %s", jid)
            _jobs.pop(jid, None)
        await asyncio.sleep(60)

@app.on_event("startup")
async def on_startup_background_cleanup():
    asyncio.create_task(_cleanup_jobs_loop())

# optional run for local debugging
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
