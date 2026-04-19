import streamlit as st
from faster_whisper import WhisperModel
import tempfile
import os
import subprocess
import re
from deep_translator import GoogleTranslator

# ================= MODEL LOADING =================
@st.cache_resource
def load_models(model_size: str):
    st.info("⏳ AI Models load ho rahe hain... (Pehli baar 2-4 min lag sakta hai)")
    whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return whisper_model

LANGUAGE_OPTIONS = [
    "en", "hi", "ur", "ar", "fr", "de", "es", "pt",
    "it", "ru", "zh", "ja", "ko", "tr", "bn", "ta",
    "te", "ml", "kn", "gu", "pa", "mr", "id", "ms",
    "th", "vi", "nl", "sv", "pl", "uk", "he", "fa",
    "sw", "am", "yo", "ig",
]

# ================= CORE FUNCTIONS =================
def extract_audio_ffmpeg(video_path, audio_path):
    """FFmpeg se video se clean 16kHz mono WAV extract karein"""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", audio_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr.strip()}")

def transcribe_video_audio(audio_path, model, lang=None):
    """Whisper transcription + auto language detect"""
    kwargs = {}
    if lang and lang != "Auto-Detect":
        kwargs["language"] = lang

    segments, info = model.transcribe(audio_path, **kwargs)
    text = " ".join(segment.text for segment in segments)
    return info.language, text, [
        {"start": segment.start, "end": segment.end, "text": segment.text}
        for segment in segments
    ]

def translate_to_urdu(text, src_lang=None):
    """Long text ko sentence-level chunks mein tod kar Urdu translate karein"""
    if not src_lang or src_lang == "Auto-Detect":
        translator = GoogleTranslator(source="auto", target="ur")
    else:
        translator = GoogleTranslator(source=src_lang, target="ur")

    sentences = re.split(r'(?<=[.!?۔؟۔،\n])\s+', text.strip())
    chunks, current = [], ""

    for sent in sentences:
        if len(current) + len(sent) < 450:
            current += sent + " "
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sent + " "
    if current.strip():
        chunks.append(current.strip())

    translated = [translator.translate(chunk) for chunk in chunks]
    return " ".join(translated)

def generate_srt(segments, urdu_segments, output_path):
    """SRT file generate karein timestamps ke sath"""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, (seg, urdu) in enumerate(zip(segments, urdu_segments), 1):
            start = format_time(seg["start"])
            end = format_time(seg["end"])
            f.write(f"{i}\n{start} --> {end}\n{urdu.strip()}\n\n")

def format_time(seconds):
    hh = int(seconds // 3600)
    mm = int((seconds % 3600) // 60)
    ss = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

# ================= STREAMLIT UI =================
st.set_page_config(page_title="🎥 MP4 → Urdu Transcription", layout="centered")
st.title("🎥 MP4 Video → Urdu Transcription")
st.caption("Upload MP4 → Extract Audio → Transcribe → Translate to Urdu")

uploaded_video = st.file_uploader("📂 MP4 Video Upload Karein", type=["mp4", "mov", "avi", "mkv"])

with st.expander("🔧 Advanced Settings"):
    force_lang = st.selectbox(
        "Force Source Language (Auto-Detect recommended)",
        ["Auto-Detect"] + LANGUAGE_OPTIONS,
        index=0,
    )
    model_size = st.selectbox("Whisper Model Size", ["tiny", "base", "small"], index=1)

if uploaded_video is not None:
    if st.button("🚀 Transcription Start Karein", type="primary"):
        with st.spinner("⚙️ Pipeline chal raha hai..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as vid_f:
                vid_f.write(uploaded_video.read())
                vid_path = vid_f.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as aud_f:
                aud_path = aud_f.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as txt_f:
                txt_path = txt_f.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".srt") as srt_f:
                srt_path = srt_f.name

            try:
                whisper_model = load_models(model_size)

                # 1️⃣ Audio Extraction
                st.info("🎬 Step 1: Video se audio extract ho raha hai...")
                extract_audio_ffmpeg(vid_path, aud_path)

                # 2️⃣ Transcription
                st.info("🎤 Step 2: Audio transcribe ho raha hai (Whisper)...")
                src_lang = force_lang if force_lang != "Auto-Detect" else None
                detected_lang, eng_text, segments = transcribe_video_audio(aud_path, whisper_model, src_lang)
                st.success(f"🔍 Detected: `{detected_lang.upper()}`")

                # 3️⃣ Translation to Urdu
                st.info("🌍 Step 3: Text Urdu mein translate ho raha hai...")
                urdu_text = translate_to_urdu(eng_text, src_lang)
                
                # SRT ke liye segment-level translation
                urdu_segments = []
                for seg in segments:
                    seg_urdu = translate_to_urdu(seg["text"].strip(), src_lang)
                    urdu_segments.append(seg_urdu)

                # Save outputs
                with open(txt_path, "w", encoding="utf-8") as f: f.write(urdu_text)
                generate_srt(segments, urdu_segments, srt_path)

                # ✅ UI Output
                st.success("✅ Transcription Complete!")
                st.subheader("📖 Urdu Transcription")
                st.text_area("Transcribed Text", urdu_text, height=200)

                col1, col2 = st.columns(2)
                with col1:
                    with open(txt_path, "rb") as f:
                        st.download_button("⬇️ Download Urdu Text (.txt)", f, "urdu_transcript.txt", "text/plain")
                with col2:
                    with open(srt_path, "rb") as f:
                        st.download_button("⬇️ Download Subtitles (.srt)", f, "urdu_subtitles.srt", "text/plain")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.code(str(e))
            finally:
                for p in [vid_path, aud_path, txt_path, srt_path]:
                    if os.path.exists(p):
                        os.remove(p)