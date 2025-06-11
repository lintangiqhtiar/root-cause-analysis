import streamlit as st
import pandas as pd
import os
from utils.recommendation import generate_recommendation


# st.set_page_config(page_title="Upload data", layout="wide")
st.title("❄ Prototype Aplikasi Rekomendasi akar permasalahan pengguna Aplikasi Digital Korlantas POLRI")
st.subheader("Hasil analisis pemodelan topik LDA dan Guided LDA")

# import file excel
df = pd.read_excel("data/rekap_lda_glda.xlsx")
# Memberi fitur untuk seleksi isi kolom LDA atau GLDA
lda_option = st.selectbox("Pilih model LDA atau GLDA", ["LDA", "Guided LDA", "LDA & GLDA"])
if lda_option == "LDA":
    st.subheader("Hasil analisis LDA")
    df = df[df["model"] == "LDA"]
elif lda_option == "GLDA":
    st.subheader("Hasil analisis GLDA")
    df = df[df["model"] == "GLDA"]
elif lda_option == "LDA & GLDA":
    st.subheader("Hasil analisis LDA & GLDA")
    df = df[df["model"].isin(["LDA", "GLDA"])]

st.dataframe(df)

# Generate Recommendation
st.subheader("🔧 Sistem Rekomendasi Perbaikan")
st.markdown("Masukkan keluhan pengguna, dan sistem akan memberikan rekomendasi perbaikannya.")

# Input dari pengguna
keluhan_input = st.text_area("Keluhan Pengguna", placeholder="Contoh: Aplikasi crash saat dibuka")

if st.button("🔍 Generate Rekomendasi") and keluhan_input:
    with st.spinner("Menghasilkan rekomendasi..."):
        hasil = generate_recommendation(keluhan_input)
        st.success("✅ Rekomendasi:")
        st.write(hasil)
