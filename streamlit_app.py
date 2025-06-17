import streamlit as st
import pandas as pd

st.set_page_config(page_title="Ensemble ROCK Clustering", layout="wide")

# Simpan state halaman
if "page" not in st.session_state:
    st.session_state.page = "home"

# Gaya CSS khusus
st.markdown("""
    <style>
    /* Navbar container */
    .nav-container {
        background-color: #003366;
        padding: 20px;
        border-radius: 0 0 20px 20px;
        display: flex;
        justify-content: center;
        gap: 60px;
    }

    /* Tautan menu */
    .nav-link {
        color: white;
        text-decoration: none;
        font-weight: bold;
        font-size: 20px;
        padding: 10px 20px;
        border-radius: 10px;
        transition: background-color 0.3s ease;
    }

    /* Saat hover */
    .nav-link:hover {
        background-color: #87CEFA;
        color: #003366;
    }

    /* Aktif */
    .active {
        background-color: red;
        color: white;
    }

    /* Judul bagian */
    .hero {
        background-color: #87CEFA;
        padding: 50px;
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
page = st.session_state.page
st.markdown(f"""
<div class="nav-container">
    <a href="?page=home" class="nav-link {'active' if page == 'home' else ''}">🏠 Home</a>
    <a href="?page=about" class="nav-link {'active' if page == 'about' else ''}">📘 About</a>
    <a href="?page=rules" class="nav-link {'active' if page == 'rules' else ''}">📜 Rules</a>
</div>
""", unsafe_allow_html=True)

# Baca URL param
query_params = st.experimental_get_query_params()
if "page" in query_params:
    st.session_state.page = query_params["page"][0]

# Konten per halaman
if st.session_state.page == "home":
    st.markdown("""
    <div class="hero">
        <h1>Better Solutions for Your Clustering</h1>
        <p>Please read the About & Rules menu first.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Upload your CSV data")
    uploaded_file = st.file_uploader("Upload file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)

elif st.session_state.page == "about":
    st.subheader("📘 About This App")
    st.markdown("Aplikasi ini dirancang untuk membantu proses clustering data UMKM dengan metode Ensemble ROCK...")

elif st.session_state.page == "rules":
    st.subheader("📜 Rules")
    st.markdown("Silakan upload file CSV dengan format yang sesuai. Data harus mencakup kolom kategori dan numerik...")
