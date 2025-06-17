import streamlit as st
import pandas as pd

st.set_page_config(page_title="Ensemble ROCK Clustering", layout="wide")

# Inisialisasi halaman
if "page" not in st.session_state:
    st.session_state.page = "Home"

# CSS untuk styling navbar dan judul
st.markdown("""
    <style>
    .main-header {
        background-color: #003366;
        padding: 20px 0;
        border-radius: 0 0 20px 20px;
        text-align: center;
    }
    .main-header button {
        background-color: transparent;
        color: white;
        font-weight: bold;
        font-size: 18px;
        border: none;
        margin: 0 20px;
    }
    .main-header button:hover {
        background-color: #87CEFA;
        color: #003366;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .hero {
        background-color: #87CEFA;
        padding: 50px 0;
        text-align: center;
        border-radius: 10px;
        margin-top: 20px;
    }
    .hero h1 {
        font-size: 42px;
        font-weight: bold;
        color: #003366;
    }
    .hero p {
        font-size: 18px;
        color: #003366;
    }
    </style>
""", unsafe_allow_html=True)

# Navbar
st.markdown("<div class='main-header'>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("🏠 Home"):
        st.session_state.page = "Home"
with col2:
    if st.button("📘 About"):
        st.session_state.page = "About"
with col3:
    if st.button("📜 Rules"):
        st.session_state.page = "Rules"
st.markdown("</div>", unsafe_allow_html=True)

# Halaman Home
if st.session_state.page == "Home":
    st.markdown("""
        <div class="hero">
            <h1>Better Solutions for Your Clustering</h1>
            <p>Please read the About & Rules menu first.</p>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload your CSV data", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)

# Halaman About
elif st.session_state.page == "About":
    st.subheader("About This App")
    st.markdown("Aplikasi ini dirancang untuk membantu analisis klaster UMKM menggunakan pendekatan Ensemble ROCK...")

# Halaman Rules
elif st.session_state.page == "Rules":
    st.subheader("Usage Rules")
    st.markdown("Silakan unggah file dalam format .csv dengan struktur data yang benar...")
