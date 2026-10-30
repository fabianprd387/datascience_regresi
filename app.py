import os
import streamlit as st
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_house_price all.pkl')

models = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Prediksi Harga Rumah Multi-Model", layout="wide")
st.title("Aplikasi Prediksi Harga Rumah")
st.write("Pilih algoritma regresi dan masukkan parameter properti untuk menghitung estimasi harga.")

selected_model = st.selectbox(
    "Pilih Algoritma Regresi:",
    options=list(models.keys())
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    area = st.number_input("Luas Area (m²)", min_value=100, max_value=5000, value=2000)
    bedrooms = st.number_input("Jumlah Kamar Tidur", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Jumlah Kamar Mandi", min_value=1, max_value=10, value=2)
    floors = st.number_input("Jumlah Lantai", min_value=1, max_value=5, value=1)
    age = st.number_input("Usia Bangunan (Tahun)", min_value=0, max_value=100, value=10)

with col2:
    distance = st.number_input("Jarak ke Pusat Kota (km)", min_value=0, max_value=100, value=10)
    garage = st.selectbox("Memiliki Garasi?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    parking = st.selectbox("Memiliki Area Parkir?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    garden = st.selectbox("Memiliki Taman?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    security = st.selectbox("Sistem Keamanan?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")

with col3:
    school = st.selectbox("Akses Sekolah Dekat?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    hospital = st.selectbox("Akses Rumah Sakit Dekat?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    mall = st.selectbox("Akses Mal Dekat?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    transport = st.selectbox("Akses Transportasi Umum Dekat?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    crime_rate = st.number_input("Tingkat Kejahatan (Crime Rate)", min_value=0.0, max_value=100.0, value=5.0)
    population_density = st.number_input("Kepadatan Penduduk", min_value=100, max_value=50000, value=5000)
    location = st.selectbox("Kualitas Lokasi", ["low", "medium", "premium"])
    income_level = st.selectbox("Tingkat Pendapatan Area", ["low", "mid", "high"])

if st.button("Hitung Prediksi Harga", type="primary"):
    input_data = pd.DataFrame([{
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'floors': floors,
        'age': age,
        'distance': distance,
        'garage': garage,
        'parking': parking,
        'garden': garden,
        'security': security,
        'school_nearby': school,
        'hospital_nearby': hospital,
        'shopping_mall_nearby': mall,
        'public_transport': transport,
        'crime_rate': crime_rate,
        'population_density': population_density,
        'location': location,
        'income_level': income_level
    }])
    
    chosen_pipeline = models[selected_model]
    prediction = chosen_pipeline.predict(input_data)[0]
    
    val_usd = prediction * 1000
    val_idr = val_usd * 15000
    
    st.success(f"Indeks Prediksi Harga ({selected_model}): **{prediction:.2f}**")
    st.info(f"**Estimasi Nominal USD:** ${val_usd:,.2f} USD\n\n**Estimasi Nominal IDR:** Rp {val_idr:,.0f} (Kurs $1 = Rp 15.000)")
