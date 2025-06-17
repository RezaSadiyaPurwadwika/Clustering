import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ========== Judul Halaman ==========
st.set_page_config(page_title="GUI Ensemble Rock Clustering", layout="wide")
st.title("🧩 GUI Ensemble Rock Clustering")

# ========== Buat Dua Kolom ==========
col1, col2 = st.columns(2)

with col1:
    st.header("📂 Upload Data")
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

with col2:
    st.header("⚙️ Data Preprocessing")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Awal")
        st.dataframe(df)

        # Contoh preprocessing awal (bisa dikembangkan lagi)
        df_clean = df.dropna()
        st.write("### Setelah Menghapus Missing Values")
        st.dataframe(df_clean)
    else:
        st.info("Silakan upload file terlebih dahulu untuk preprocessing.")
