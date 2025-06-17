import streamlit as st
import pandas as pd

st.set_page_config(page_title="Ensemble Rock Clustering", layout="wide")

# Inisialisasi halaman
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Custom navbar pakai tombol teks
col1, col2, col3 = st.columns([1,1,1])
with col1:
    if st.button("🏠 Home"):
        st.session_state.page = "Home"
with col2:
    if st.button("📘 About"):
        st.session_state.page = "About"
with col3:
    if st.button("📜 Rules"):
        st.session_state.page = "Rules"

# CSS untuk mengganti warna teks tombol jadi putih
st.markdown("""
    <style>
    .stButton > button {
        background-color: transparent;
        color: white;
        border: none;
        font-size: 18px;
        font-weight: bold;
        padding: 10px;
    }
    .stButton > button:hover {
        background-color: #87CEFA;
        color: #003366;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Background navbar
st.markdown("""
    <div style='position: absolute; top: 0; left: 0; width: 100%; height: 80px; background-color: #003366; z-index: -1; border-radius: 0 0 20px 20px;'></div>
    <br><br>
""", unsafe_allow_html=True)

# Tampilkan halaman
if st.session_state.page == "Home":
    st.markdown("<h1 style='text-align: center;'>Better Solutions for Your Clustering</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Please read the About & Rules menu first.</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your CSV data", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)

elif st.session_state.page == "About":
    st.subheader("About this App")
    st.markdown("Aplikasi ini digunakan untuk membantu clustering UMKM...")

elif st.session_state.page == "Rules":
    st.subheader("Aturan Penggunaan")
    st.markdown("Mohon pastikan file Anda sudah sesuai format...")
