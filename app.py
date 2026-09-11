import os
import time
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# ============================================================
# KONFIGURASI
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
H5_MODEL_PATH = os.path.join(BASE_DIR, "model_pretrained_mobilenetv2.h5")
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, "model_appleorange_quant.tflite")
IMG_SIZE = (128, 128)
CLASS_NAMES = ["Apel", "Jeruk"]  # index 0 -> Apel, index 1 -> Jeruk
LOW_CONF_THRESHOLD = 0.60

VERSION_HISTORY = [
    {
        "versi": "v1",
        "perubahan": "Model dasar klasifikasi Apel vs Jeruk memakai transfer learning "
                     "MobileNetV2. Antarmuka satu halaman: unggah gambar, lihat hasil.",
    },
    {
        "versi": "v2",
        "perubahan": "Menambahkan model teroptimasi (TFLite hasil quantization) sebagai "
                     "pilihan selain model asli, breakdown probabilitas tiap kelas.",
    },
    {
        "versi": "v3",
        "perubahan": "Desain ulang antarmuka, navigasi multi-halaman, dan fitur Uji "
                     "Performa yang membandingkan ukuran berkas & kecepatan inferensi "
                     "kedua model secara langsung.",
    },
]

st.set_page_config(page_title="Stasiun Sortir Apel & Jeruk", page_icon="🧺", layout="wide")

# ============================================================
# GAYA (CSS) — disederhanakan, kontras diperbaiki lewat accent-color
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bitter:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
    --bg: #F6F7F1; --panel: #FFFFFF; --ink: #1E2420; --ink-soft: #55604F;
    --line: #DCE3D0; --forest: #2F4B3C; --forest-dark: #22362A;
    --apple: #A8402A; --orange: #B9821F;
}

.stApp { background: var(--bg); color: var(--ink); }
.block-container { max-width: 800px; padding-top: 2.5rem; }
h1, h2, h3 { font-family: 'Bitter', serif; color: var(--ink); }
p, span, label, .stMarkdown { font-family: 'IBM Plex Sans', sans-serif; color: var(--ink); }
footer { visibility: hidden; }

/* Radio di halaman utama (latar terang): pill putih, dot pakai warna tema */
div[role="radiogroup"] label {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 4px; padding: 0.3rem 0.8rem; margin-right: 0.4rem;
}
.main input[type="radio"] { accent-color: var(--forest); }

/* Sidebar: latar gelap, teks terang. Pill putih DIHAPUS di sini supaya tidak
   jadi teks putih di atas kotak putih (bug kontras) - dibuat transparan saja. */
[data-testid="stSidebar"] { background: var(--forest); }
[data-testid="stSidebar"] * { color: #EFF3EA !important; }
[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: transparent; border: none; padding: 0.4rem 0; margin-right: 0;
}
[data-testid="stSidebar"] input[type="radio"] { accent-color: #EFF3EA; }

.stButton>button {
    background: var(--forest); color: #FFFFFF; border: none;
    border-radius: 4px; padding: 0.55rem 1.4rem; font-weight: 500;
}
.stButton>button:hover { background: var(--forest-dark); color: #FFFFFF; }
.stButton>button p, .stButton>button span, .stButton>button div {
    color: #FFFFFF !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: var(--panel); border: 1.5px dashed var(--line); border-radius: 6px;
}
.stProgress > div > div { background: var(--forest); }

.result-card {
    background: var(--panel); border: 1px solid var(--line); border-left: 6px solid var(--forest);
    border-radius: 6px; padding: 1.1rem 1.4rem; margin-top: 0.8rem;
}
.result-card.apel { border-left-color: var(--apple); }
.result-card.jeruk { border-left-color: var(--orange); }
.result-card.low-conf { border-left-color: #8A8F82; background: #FBFBF7; }
.result-title { font-family: 'Bitter', serif; font-size: 1.4rem; margin: 0 0 0.2rem 0; }
.result-meta { color: var(--ink-soft); font-size: 0.9rem; }

.spec-card {
    background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 1rem 1.2rem;
}
.spec-card .label { color: var(--ink-soft); font-size: 0.85rem; }
.spec-card .value { font-family: 'Bitter', serif; font-size: 1.6rem; margin-top: 0.1rem; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL
# ============================================================
@st.cache_resource
def load_keras_model():
    return tf.keras.models.load_model(H5_MODEL_PATH)


@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0).astype(np.float32)


def predict_keras(processed):
    return float(load_keras_model().predict(processed, verbose=0)[0][0])


def predict_tflite(processed):
    interpreter = load_tflite_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], processed)
    interpreter.invoke()
    return float(interpreter.get_tensor(output_details[0]["index"])[0][0])


def classify(prob_jeruk):
    prob_apel = 1 - prob_jeruk
    label = CLASS_NAMES[1] if prob_jeruk > 0.5 else CLASS_NAMES[0]
    confidence = max(prob_apel, prob_jeruk)
    return label, confidence, prob_apel, prob_jeruk


def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


# ============================================================
# STATE
# ============================================================
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "current_name" not in st.session_state:
    st.session_state.current_name = None


# ============================================================
# SIDEBAR — NAVIGASI
# ============================================================
with st.sidebar:
    st.markdown("### 🍎 Apple vs Orange Classifier")
    st.caption("managed by marcellmchan")
    page = st.radio(
        "Navigasi",
        ["Sortir Gambar", "Uji Performa Model", "Tentang & Versi"],
        label_visibility="collapsed",
    )
    st.markdown("---")
  


# ============================================================
# HALAMAN 1 — SORTIR GAMBAR
# ============================================================
if page == "Sortir Gambar":
    st.markdown('<p class="hero-eyebrow">Klasifikasi gambar buah</p>', unsafe_allow_html=True)
    st.title("Apel atau jeruk?")
    st.write(
        "Unggah satu foto buah, pilih model yang ingin dipakai, lalu jalankan klasifikasi."
    )

    model_choice = st.radio(
        "Pilih model inferensi",
        ["Model asli (.h5)", "Model teroptimasi (.tflite)"],
        horizontal=True,
        help="Model teroptimasi hasil kuantisasi dari model asli, berukuran lebih kecil — "
             "bandingkan keduanya di halaman 'Uji Performa Model'.",
    )

    uploaded_file = st.file_uploader(
        "Unggah gambar (JPG/PNG)", type=["jpg", "jpeg", "png"],
        help="Gunakan foto satu buah dengan latar belakang polos untuk hasil terbaik.",
    )

    if uploaded_file is None:
        camera_file = st.camera_input(
            "Atau ambil foto langsung dari kamera",
            help="Pastikan buah terlihat jelas dan pencahayaan cukup.",
        )
    else:
        camera_file = None

    input_file = uploaded_file or camera_file

    if input_file is not None:
        image = Image.open(input_file)
        image_name = getattr(input_file, "name", None) or f"kamera_{int(time.time())}.jpg"
        st.session_state.current_image = image
        st.session_state.current_name = image_name
        st.image(image, caption=f"Gambar: {image_name}", width=320)

        if st.button("Klasifikasikan"):
            with st.spinner("Memproses gambar..."):
                processed = preprocess_image(image)
                if model_choice.startswith("Model asli"):
                    prob_jeruk = predict_keras(processed)
                else:
                    prob_jeruk = predict_tflite(processed)
                label, confidence, prob_apel, prob_jeruk = classify(prob_jeruk)

            low_conf = confidence < LOW_CONF_THRESHOLD
            css_class = "low-conf" if low_conf else ("jeruk" if label == "Jeruk" else "apel")
            title_text = f"Kurang yakin — kemungkinan {label}" if low_conf else label

            st.markdown(f"""
            <div class="result-card {css_class}">
                <p class="result-title">{title_text}</p>
                <p class="result-meta">Tingkat keyakinan: {confidence*100:.1f}% — {model_choice}</p>
            </div>
            """, unsafe_allow_html=True)

            if low_conf:
                st.caption(
                    "Keyakinan di bawah 60% — coba foto dengan pencahayaan lebih baik, "
                    "latar belakang polos, dan satu buah saja dalam bingkai."
                )

            st.write("")
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"Apel — {prob_apel*100:.1f}%")
                st.progress(prob_apel)
            with c2:
                st.write(f"Jeruk — {prob_jeruk*100:.1f}%")
                st.progress(prob_jeruk)
    else:
        st.info("Silakan unggah gambar terlebih dahulu.")


# ============================================================
# HALAMAN 2 — UJI PERFORMA MODEL (poin optimasi/bonus)
# ============================================================
elif page == "Uji Performa Model":
    st.markdown('<p class="hero-eyebrow">Perbandingan model</p>', unsafe_allow_html=True)
    st.title("Model asli vs. model teroptimasi")
    st.write(
        "Model teroptimasi dibuat lewat post-training quantization dari model asli. "
        "Halaman ini mengukur langsung selisih ukuran berkas dan kecepatan inferensi "
        "keduanya, memakai gambar yang sama dari halaman 'Sortir Gambar'."
    )

    if st.session_state.current_image is None:
        st.warning("Unggah gambar dulu di halaman 'Sortir Gambar', lalu kembali ke sini.")
    else:
        st.image(st.session_state.current_image, caption=st.session_state.current_name, width=240)
        n_runs = st.slider("Jumlah pengulangan inferensi", min_value=3, max_value=15, value=5)

        if st.button("Jalankan benchmark"):
            processed = preprocess_image(st.session_state.current_image)
            with st.spinner("Menjalankan kedua model..."):
                predict_keras(processed)   # warm-up, tidak dihitung
                predict_tflite(processed)

                keras_times, tflite_times = [], []
                for _ in range(n_runs):
                    t0 = time.perf_counter()
                    prob_h5 = predict_keras(processed)
                    keras_times.append((time.perf_counter() - t0) * 1000)

                    t0 = time.perf_counter()
                    prob_tflite = predict_tflite(processed)
                    tflite_times.append((time.perf_counter() - t0) * 1000)

            label_h5, conf_h5, _, _ = classify(prob_h5)
            label_tflite, conf_tflite, _, _ = classify(prob_tflite)
            avg_h5 = sum(keras_times) / len(keras_times)
            avg_tflite = sum(tflite_times) / len(tflite_times)
            size_h5 = file_size_mb(H5_MODEL_PATH)
            size_tflite = file_size_mb(TFLITE_MODEL_PATH)

            st.write("")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="spec-card">
                    <p class="label">Model asli (.h5)</p>
                    <p class="value">{size_h5:.1f} MB</p>
                    <p class="result-meta">{avg_h5:.1f} ms/gambar — prediksi: {label_h5} ({conf_h5*100:.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="spec-card">
                    <p class="label">Model teroptimasi (.tflite)</p>
                    <p class="value">{size_tflite:.1f} MB</p>
                    <p class="result-meta">{avg_tflite:.1f} ms/gambar — prediksi: {label_tflite} ({conf_tflite*100:.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            reduction = (1 - size_tflite / size_h5) * 100
            speed_note = "lebih cepat" if avg_tflite < avg_h5 else "lebih lambat"
            speed_diff = abs(avg_h5 - avg_tflite)
            agree = "sama" if label_h5 == label_tflite else "berbeda"
            st.write(
                f"Model teroptimasi {reduction:.0f}% lebih kecil dan {speed_diff:.1f} ms "
                f"{speed_note} per gambar dibanding model asli. Hasil klasifikasi kedua "
                f"model {agree} untuk gambar ini."
            )


# ============================================================
# HALAMAN 3 — TENTANG & VERSI
# ============================================================
else:
    st.markdown('<p class="hero-eyebrow">Dokumentasi proyek</p>', unsafe_allow_html=True)
    st.title("Tentang aplikasi ini")
    st.write(
        "Aplikasi ini mengklasifikasikan foto buah ke dalam dua kelas, apel atau jeruk, "
        "memakai transfer learning dari MobileNetV2. Tersedia juga versi teroptimasi "
        "(TFLite hasil post-training quantization) untuk perbandingan performa."
    )

    with st.expander("Cara pakai"):
        st.markdown("""
        1. Buka halaman **Sortir Gambar**, lalu unggah foto.
        2. Pilih model yang ingin dipakai, lalu klik **Klasifikasikan**.
        3. Lihat hasil dan tingkat keyakinan pada kartu hasil.
        4. Untuk membandingkan kecepatan kedua model, buka halaman **Uji Performa Model**.
        """)

    st.subheader("Riwayat versi")
    for v in VERSION_HISTORY:
        st.markdown(f"""
        <div class="spec-card" style="margin-bottom:0.6rem;">
            <p class="label">{v['versi']}</p>
            <p style="margin-top:0.3rem;">{v['perubahan']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "Arsitektur: Transfer learning MobileNetV2, input 128×128, klasifikasi biner "
        "(sigmoid). Model teroptimasi dihasilkan lewat TensorFlow Lite post-training "
        "quantization dari model asli."
    )
