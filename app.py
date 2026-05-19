import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Classic Car Price Predictor",
    page_icon="🚘",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');

:root {
    --cream: #f7efe2;
    --brown: #4b2e1f;
    --gold: #b7894b;
    --dark: #1f1712;
}

.stApp {
    background: linear-gradient(135deg, #f8efe0 0%, #ead8bd 45%, #d6b88d 100%);
    color: var(--dark);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 2.2rem;
    border-radius: 28px;
    background: rgba(75, 46, 31, 0.92);
    color: #fff7e8;
    border: 2px solid rgba(183, 137, 75, 0.65);
    box-shadow: 0 18px 40px rgba(31, 23, 18, 0.25);
    margin-bottom: 1.5rem;
}

.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    margin-bottom: 0.3rem;
}

.hero p {
    font-family: 'Inter', sans-serif;
    font-size: 1.05rem;
    opacity: 0.9;
}

.vintage-card {
    background: rgba(255, 248, 235, 0.78);
    border: 1.5px solid rgba(75, 46, 31, 0.18);
    border-radius: 22px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 10px 26px rgba(75, 46, 31, 0.12);
}

.result-box {
    text-align: center;
    background: linear-gradient(145deg, #4b2e1f, #241711);
    color: #fff2d8;
    border-radius: 24px;
    padding: 2rem;
    border: 2px solid #b7894b;
}

.result-box h2 {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
}

.result-price {
    font-size: 3rem;
    font-weight: 800;
    color: #f6c979;
}

[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif;
}

.stButton > button {
    background: #4b2e1f;
    color: #fff7e8;
    border: 1px solid #b7894b;
    border-radius: 14px;
    padding: 0.7rem 1.2rem;
    font-weight: 700;
}

.stButton > button:hover {
    background: #6a422c;
    color: white;
    border-color: #f1c27d;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("car_price_model.pkl")

@st.cache_data
def load_info():
    with open("model_info.json", "r") as f:
        return json.load(f)

model = load_model()
info = load_info()

st.markdown("""
<div class="hero">
    <h1>Classic Car Price Predictor</h1>
    <p>Prediksi harga mobil dengan nuansa klasik, elegan, dan vintage berdasarkan spesifikasi kendaraan.</p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.35, 0.85], gap="large")

with left:
    st.markdown('<div class="vintage-card">', unsafe_allow_html=True)
    st.subheader("⚙️ Masukkan Spesifikasi Mobil")

    cat_inputs = {}
    num_inputs = {}

    c1, c2, c3 = st.columns(3)
    cat_cols = info["categorical_features"]
    for idx, col in enumerate(cat_cols):
        opts = info["options"].get(col, [])
        container = [c1, c2, c3][idx % 3]
        with container:
            cat_inputs[col] = st.selectbox(col.replace("_", " "), opts, index=0 if opts else None)

    st.divider()
    st.caption("Nilai default diambil dari median dataset. Silakan ubah sesuai mobil yang ingin diprediksi.")

    numeric_labels = {
        "Sales_in_thousands": "Sales in Thousands",
        "__year_resale_value": "Year Resale Value",
        "Engine_size": "Engine Size",
        "Horsepower": "Horsepower",
        "Wheelbase": "Wheelbase",
        "Width": "Width",
        "Length": "Length",
        "Curb_weight": "Curb Weight",
        "Fuel_capacity": "Fuel Capacity",
        "Fuel_efficiency": "Fuel Efficiency",
        "Power_perf_factor": "Power Performance Factor",
        "Launch_Year": "Launch Year",
        "Launch_Month": "Launch Month"
    }

    cols = st.columns(3)
    for idx, col in enumerate(info["numeric_features"]):
        default = info["defaults"].get(col)
        if default is None or np.isnan(default):
            default = 0.0
        with cols[idx % 3]:
            step = 1.0 if col in ["Horsepower", "Launch_Year", "Launch_Month"] else 0.1
            num_inputs[col] = st.number_input(
                numeric_labels.get(col, col.replace("_", " ")),
                value=float(default),
                step=step,
                format="%.3f" if step < 1 else "%.0f"
            )

    predict_clicked = st.button("Prediksi Harga Mobil", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="vintage-card">', unsafe_allow_html=True)
    st.subheader("📊 Info Model")
    m = info["metrics"]
    st.metric("R² Test", m.get("r2", "-"))
    st.metric("MAE", f"{m.get('mae', 0):,.3f}")
    st.caption("MAE dan R² dihitung dari pembagian train-test sederhana. Dataset kecil, jadi hasil bisa berubah kalau model dilatih ulang.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    placeholder = st.empty()

input_data = {**cat_inputs, **num_inputs}
input_df = pd.DataFrame([input_data])
for col in info["categorical_features"]:
    input_df[col] = input_df[col].astype(str)

# Pastikan kolom sesuai urutan training pipeline
ordered_cols = info["numeric_features"] + info["categorical_features"]
input_df = input_df[ordered_cols]

if predict_clicked:
    prediction = float(model.predict(input_df)[0])
    usd_estimate = prediction * 1000
    with placeholder.container():
        st.markdown(f"""
        <div class="result-box">
            <h2>Estimasi Harga</h2>
            <div class="result-price">${usd_estimate:,.2f}</div>
            <p>{prediction:,.3f} dalam satuan <b>Price_in_thousands</b></p>
        </div>
        """, unsafe_allow_html=True)
else:
    with placeholder.container():
        st.markdown("""
        <div class="result-box">
            <h2>Siap Memprediksi</h2>
            <p>Isi spesifikasi mobil, lalu klik tombol prediksi.</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("© Classic Car Price Predictor • Dibuat dengan Streamlit")
