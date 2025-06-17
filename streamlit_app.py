import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ========== Judul Halaman ==========
st.set_page_config(page_title="Template GUI Clustering", layout="wide")
st.title("🧩 Template GUI Clustering Data")

# ========== 1. Upload Data ==========
st.header("📂 Upload Data")
uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Awal")
    st.dataframe(df)

    # ========== 2. Preprocessing ==========
    st.header("⚙️ Data Preprocessing")
