"""
H1 : input file yang sudah melalui preprocessing (yakni hingga data telah ber sentimen negatif)
H2 : input berupa file json

"""

import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Upload data", layout="wide")
st.title("❄ Upload data komentar google playstore")

#Upload file
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    file_path = os.path.join("data", "uploaded_data-test.csv")
    df = pd.read_csv(uploaded_file)
    df.to_csv(file_path, index= False)

    st.success("File berhasil diupload silahkan lanjut ke halaman analisis")
    st.page_link("pages/analisis.py", label="Lihat hasil analisis")