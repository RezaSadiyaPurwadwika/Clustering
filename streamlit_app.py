import streamlit as st
import pandas as pd

st.set_page_config(page_title="Clustering App", layout="wide")

# Session state untuk halaman aktif
if "page" not in st.session_state:
    st.session_state.page = "home"

# CSS untuk Navbar dan Hero Section
st.markdown("""
    <style>
    .navbar {
        background-color: #003366;
        padding: 1rem;
        border-radius: 0 0 15px 15px;
        display: flex;
        justify-content: center;
        gap: 40px;
    }
    .nav-item {
        color: white;
        font-weight: bold;
        font-size: 18px;
        text-decoration: none;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        transition: 0.3s;
    }
    .nav-item:hover {
        background-color: #1E90FF;
        color: #fff;
    }
    .active {
        background-color: red;
    }

    .hero {
        background-color: #E0F0FF;
        text-align: center;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 20px;
    }
    .hero h1 {
        color: #003366;
        font-size: 36px;
        margin-bottom: 0.5rem;
    }
    .hero p {
        color: #003366;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Navbar HTML
page = st.session_state.page
st.markdown(f"""
<div class="navbar">
    <a href="/?page=home" class="nav-item {'active' if page == 'home' else ''}">🏠 Home</a>
    <a href="/?page=about" class="nav-item {'active' if page == 'about' else ''}">📘 About</a>
    <a href="/?page=rules" class="nav-item {'active' if page == 'rules' else ''}">📜 Rules</a>
</div>
""", unsafe_allow_html=True)

# Ambil parameter dari URL
query_params = st.experimental_get_query_params()
if "page" in query_params:
    st.session_state.page = query_params["page"][0]

# Konten halaman
if st.session_state.page == "home":
    st.markdown("""
    <div class="hero">
        <h1>Better Solutions for Your Clustering</h1>
        <p>Please read the About & Rules menu first.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📤 Upload Your CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)

elif st.session_state.page == "about":
    st.subheader("📘 About")
    st.markdown("Aplikasi ini digunakan untuk clustering data UMKM...")

elif st.session_state.page == "rules":
    st.subheader("📜 Rules")
    st.markdown("Silakan upload data yang sesuai format, tidak ada missing values, dan variabel kategorikal jelas...")
