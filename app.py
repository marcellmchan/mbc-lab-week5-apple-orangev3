import os
import streamlit as st
import numpy as np
import pickle
import re
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import onnxruntime as ort

# ============================================================
# KONFIGURASI PATH
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
H5_MODEL_PATH = os.path.join(BASE_DIR, "sentiment_GRU_cfg1_seq100.h5")
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "sentiment_gru.onnx")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
MAX_LEN = 100

st.set_page_config(page_title="Cinematic Sentiment Analysis", page_icon="🍿", layout="wide")

# ============================================================
# CUSTOM CSS: MODERN DARK THEME DENGAN AKSEN KUNING
# ============================================================
st.markdown("""
<style>
    /* Background & Teks Umum */
    .stApp {
        background-color: #0E0E10;
        color: #E0E0E0;
    }
    
    /* Header/Title dengan gradasi emas/kuning */
    h1 {
        background: -webkit-linear-gradient(45deg, #F5C518, #E2B616);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        font-size: 3rem !important;
        margin-bottom: 0px !important;
    }
    h2, h3 { color: #F5C518 !important; font-weight: 700 !important; }

    /* Text Area Styling */
    .stTextArea textarea {
        background-color: #1A1A1D !important;
        color: #FFFFFF !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
        padding: 15px !important;
        font-size: 1.1rem !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .stTextArea textarea:focus {
        border-color: #F5C518 !important;
        box-shadow: 0 0 0 1px #F5C518 !important;
    }

    /* Primary Button (Kuning) */
    .stButton>button {
        background-color: #F5C518 !important;
        color: #0E0E10 !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FFD700 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(245, 197, 24, 0.4);
    }

    /* Card Box / Wadah Hasil */
    .glass-card {
        background: rgba(26, 26, 29, 0.95);
        border: 1px solid rgba(245, 197, 24, 0.2);
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    }

    /* Metric Override */
    [data-testid="stMetricValue"] {
        color: #F5C518 !important;
        font-size: 2rem !important;
        font-weight: bold;
    }
    [data-testid="stMetricLabel"] { color: #AAAAAA !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL & UTILS
# ============================================================
@st.cache_resource
def load_keras_model():
    return load_model(H5_MODEL_PATH)

@st.cache_resource
def load_onnx_model():
    return ort.InferenceSession(ONNX_MODEL_PATH)

@st.cache_resource
def get_tokenizer():
    with open(TOKENIZER_PATH, 'rb') as f:
        return pickle.load(f)

def preprocess_text(text, tokenizer):
    text = text.lower()
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    sequences = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

def predict_onnx(session, processed):
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: processed.astype(np.float32)})
    return result[0][0][0]

if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# MAIN LAYOUT
# ============================================================
st.markdown("<h1>🎬 Cinematic Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #888; font-size: 1.1rem; margin-top: 5px; margin-bottom: 30px;'>Ketahui apakah ulasan film Anda terdeteksi Positif atau Negatif oleh AI.</p>", unsafe_allow_html=True)

col_main, col_side = st.columns([2, 1], gap="large")

with col_main:
    # -------------------------
    # INPUT SECTION
    # -------------------------
    st.markdown("### 📝 Tulis Ulasan Anda (B. Inggris)")
    
    user_input = st.text_area(
        label="Review text",
        value="The cinematography was absolutely breathtaking. Every scene felt like a painting. I would highly recommend this movie to anyone!",
        height=140,
        label_visibility="collapsed"
    )

    col_btn, col_radio = st.columns([1, 2])
    with col_radio:
        model_choice = st.radio(
            "Mesin Inferensi:",
            ["H5 (Original)", "ONNX (Optimized)"],
            horizontal=True,
            label_visibility="collapsed"
        )
    with col_btn:
        analyze_clicked = st.button("✨ Analisis Sekarang")

    # -------------------------
    # HASIL PREDIKSI
    # -------------------------
    if analyze_clicked:
        if user_input.strip() == "":
            st.warning("Mohon isi ulasan film terlebih dahulu.")
        else:
            with st.spinner("🤖 AI sedang membaca ulasan Anda..."):
                tokenizer = get_tokenizer()
                processed = preprocess_text(user_input, tokenizer)
                start = time.time()

                if model_choice.startswith("H5"):
                    keras_model = load_keras_model()
                    pred = keras_model.predict(processed, verbose=0)[0][0]
                    model_label = "Keras H5"
                else:
                    onnx_session = load_onnx_model()
                    pred = predict_onnx(onnx_session, processed)
                    model_label = "ONNX Runtime"

                elapsed = time.time() - start

            prob_positive = float(pred)
            prob_negative = 1 - prob_positive
            pred_class = "Positive" if prob_positive > 0.5 else "Negative"
            score_10 = round(prob_positive * 10, 1) if prob_positive > 0.5 else round((prob_negative * 10), 1)

            st.markdown("<br>", unsafe_allow_html=True)
            
            # UI Card Hasil
            emoji = "🌟" if pred_class == "Positive" else "💔"
            color = "#F5C518" if pred_class == "Positive" else "#FF4B4B"
            
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="color: {color}; margin-top:0; font-size: 1.5rem;">{emoji} {pred_class.upper()} SENTIMENT</h4>
                <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 15px;">
                    <div>
                        <p style="margin: 0; color: #AAA; font-size: 0.9rem;">Confidence Score</p>
                        <h2 style="margin: 0; color: #FFF;">{max(prob_positive, prob_negative)*100:.1f}%</h2>
                    </div>
                    <div style="text-align: right;">
                        <p style="margin: 0; color: #AAA; font-size: 0.9rem;">Rating Setara</p>
                        <h2 style="margin: 0; color: {color};">★ {score_10}<span style="font-size:1rem;color:#777;">/10</span></h2>
                    </div>
                </div>
                <hr style="border-color: #333; margin: 15px 0;">
                <p style="margin: 0; color: #888; font-size: 0.85rem;">⏱ Waktu Inferensi: <b>{elapsed*1000:.1f}ms</b> menggunakan mesin <b>{model_label}</b></p>
            </div>
            """, unsafe_allow_html=True)

            # Simpan ke riwayat
            st.session_state.history.append({
                "Review (Snippet)": user_input[:40] + "...",
                "Rating": f"★ {score_10}/10",
                "Sentiment": pred_class,
                "Model": model_label
            })


with col_side:
    # -------------------------
    # RIGHT SIDEBAR (Info & History)
    # -------------------------
    st.markdown("### 📊 Info Optimasi")
    
    if os.path.exists(H5_MODEL_PATH) and os.path.exists(ONNX_MODEL_PATH):
        h5_mb = os.path.getsize(H5_MODEL_PATH) / (1024 * 1024)
        onnx_mb = os.path.getsize(ONNX_MODEL_PATH) / (1024 * 1024)
        saving = (1 - onnx_mb / h5_mb) * 100
        
        st.metric("Model Asli (.h5)", f"{h5_mb:.2f} MB")
        st.metric("Model Optimasi (.onnx)", f"{onnx_mb:.2f} MB", delta=f"-{h5_mb-onnx_mb:.2f} MB", delta_color="inverse")
    else:
        st.info("Metrics akan muncul setelah model di-load.")
    
    st.markdown("<br>### 🕒 Riwayat Terakhir", unsafe_allow_html=True)
    if st.session_state.history:
        for idx, item in enumerate(reversed(st.session_state.history[-4:])): # Tampilkan 4 terakhir
            color = "#F5C518" if item['Sentiment'] == "Positive" else "#FF4B4B"
            st.markdown(f"""
            <div style="background-color: #1A1A1D; padding: 10px 15px; border-radius: 6px; margin-bottom: 10px; border-left: 3px solid {color};">
                <div style="display:flex; justify-content:space-between;">
                    <span style="font-weight:bold; color:{color}; font-size:0.9rem;">{item['Sentiment']}</span>
                    <span style="color:#777; font-size:0.8rem;">{item['Model']}</span>
                </div>
                <div style="color:#AAA; font-size:0.85rem; margin-top:5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                    "{item['Review (Snippet)']}"
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: #555; font-size: 0.9rem;'>Belum ada prediksi.</p>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #555; font-size: 0.9rem;'>Final Project Week 5 • Menggunakan Model Keras GRU dengan Optimasi ONNX Runtime</div>", unsafe_allow_html=True)
