# frontend/streamlit_app.py  (relevant excerpts; you can replace the entire file with this)
import streamlit as st
import requests, base64, io, tempfile, os, time, json, pathlib
from pydub import AudioSegment
from gtts import gTTS

ROOT = pathlib.Path(__file__).resolve().parents[1]
st.set_page_config(page_title="ClauseWise — Legal Analyzer", layout="wide", initial_sidebar_state="expanded")

# Load styles if exist
css_file = pathlib.Path(__file__).parent / "style.css"
if css_file.exists():
    st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)

# Sidebar preferences
API_URL = st.sidebar.text_input("API URL", value="http://localhost:8000/analyze", help="Point to your backend analyze endpoint")

st.title("ClauseWise — Legal Document Analyzer")

with st.sidebar.form("prefs"):
    st.subheader("Preferences")
    out_lang = st.selectbox("Summary language", options=["en","es","pt","hi","ta","te","fr","de"], index=0, format_func=lambda x: x)
    tts_lang = st.selectbox("TTS language (for audio)", options=["en","es","pt","hi","ta","te","fr","de"], index=0, format_func=lambda x: x)
    want_audio = st.checkbox("Generate audio summary (MP3)", value=True)
    use_granite = st.checkbox("Use IBM Granite (optional)", value=False)
    plain_english = st.checkbox("Simplify to Plain Language", value=True)
    submitted = st.form_submit_button("Save preferences")

st.info("Upload a document or audio file. Supported: PDF, DOCX, TXT, MP3, WAV, M4A, OGG.")

col1, col2 = st.columns([2,1])
with col1:
    uploaded = st.file_uploader("Upload document or audio", type=["pdf","docx","txt","mp3","wav","m4a","ogg"], help="PDF / DOCX / TXT / MP3 / WAV; Scanned PDFs supported via OCR.")
    analyze_btn = st.button("Analyze")

with col2:
    st.markdown("### Quick Actions")
    st.write("Use sample files in the `data/samples` folder for quick testing.")
    st.markdown("---")

if analyze_btn and uploaded:
    # Prepare form data & files
    files = {"file": (uploaded.name, uploaded.read())}
    data = {
        "output_lang": out_lang,
        "plain_english": json.dumps(plain_english),
        "use_granite": json.dumps(use_granite),
        "want_tts": json.dumps(want_audio),
        "tts_lang": tts_lang
    }

    # Submit job
    try:
        # Short timeout for job submission
        resp = requests.post(API_URL, files=files, data=data, timeout=30)
        resp.raise_for_status()
        job = resp.json()
        job_id = job.get("job_id")
        if not job_id:
            st.error("No job_id returned from server.")
            st.stop()
    except Exception as e:
        st.error(f"Failed to submit job: {e}")
        st.stop()

    # Poll for result
    poll_url = API_URL.replace("/analyze", f"/result/{job_id}")
    max_wait = 600  # seconds
    interval = 2
    waited = 0
    result = None

    with st.spinner("Processing document (this may take a moment)..."):
        while waited < max_wait:
            try:
                r = requests.get(poll_url, timeout=30)
                r.raise_for_status()
                j = r.json()
                status = j.get("status")
                if status == "completed":
                    result = j.get("result")
                    break
                if status == "failed":
                    st.error(f"Analysis failed: {j.get('error')}")
                    st.stop()
                # else queued/running
            except Exception:
                # Ignore transient errors, try again
                pass
            time.sleep(interval)
            waited += interval
        else:
            st.error("Processing timed out. Try again or increase max_wait.")
            st.stop()

    # Show results
    st.subheader("Document Summary")
    st.info(f"Detected language: **{result.get('lang_detected','unknown')}** — Document Type: **{result.get('doc_type',{}).get('label','-')}**")
    st.markdown(f"**Summary ({out_lang}):**")
    st.write(result.get("summary", ""))

    # Audio play & download
    if want_audio and result.get("audio_base64"):
        try:
            audio_b64 = result.get("audio_base64")
            audio_bytes = base64.b64decode(audio_b64)
            st.audio(audio_bytes, format="audio/mp3")
            st.download_button("Download audio summary (MP3)", data=audio_bytes, file_name=result.get("audio_filename", f"summary_{job_id}.mp3"), mime="audio/mpeg")
        except Exception as e:
            st.warning(f"Could not show/download audio: {e}")

    # Show clauses
    st.subheader("Clauses")
    clauses = result.get("clauses", [])
    for idx, c in enumerate(clauses, 1):
        risk = c.get("risk", "low")
        risk_class = "risk-low" if risk == "low" else ("risk-med" if risk == "medium" else "risk-high")
        st.markdown(f"**Clause {idx}: {c.get('title','')}** — {risk.upper()}")
        st.write(c.get("summary", ""))
        with st.expander("Show clause text"):
            st.write(c.get("text",""))

    # Named entities
    st.subheader("Named Entities")
    st.dataframe(result.get("entities", []))

    st.download_button("Download summary (TXT)", result.get("summary",""), file_name="clausewise_summary.txt")
