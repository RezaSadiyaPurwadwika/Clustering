import streamlit as st
import pandas as pd

st.set_page_config(page_title="Ensemble Rock Clustering", layout="wide")

# Inisialisasi halaman
if "page" not in st.session_state:
    st.session_state.page = "home"

# =========================
# CSS untuk Navbar dan Tombol
# =========================
st.markdown("""
<style>
    .navbar {
        background-color: #003366;
        padding: 1.2rem 2rem;
        color: white;
        border-radius: 0 0 10px 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .navbar-title {
        font-size: 1.5rem;
        font-weight: bold;
        display: flex;
        align-items: center;
    }
    .navbar-buttons {
        display: flex;
        gap: 1rem;
    }
    .navbar-buttons button {
        background-color: white;
        color: #003366;
        border: 2px solid #003366;
        padding: 0.4rem 1.2rem;
        border-radius: 8px;
        font-weight: bold;
        cursor: pointer;
    }
    .navbar-buttons button:hover {
        background-color: #003366;
        color: white;
    }
</style>

<div class="navbar">
    <div class="navbar-title">🌐 Ensemble Rock Clustering</div>
    <div class="navbar-buttons">
        <form action="" method="post">
            <button name="page" value="home">Home</button>
            <button name="page" value="about">About</button>
            <button name="page" value="rules">Rules</button>
        </form>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# Tangkap Event dari Tombol HTML
# =========================
if st.experimental_get_query_params().get("page"):
    st.session_state.page = st.experimental_get_query_params()["page"][0]

# =========================
# Konten Halaman
# =========================
def show_home():
    st.markdown("""
        <div style='padding: 4rem 2rem; background-color: #87CEFA; color: #003366; text-align: center; border-radius: 10px'>
            <h1><strong>Better Solutions for Your Clustering</strong></h1>
            <p>Please read the About menu before uploading data.</p>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload your CSV data", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data")
        st.dataframe(df)

def show_about():
    st.markdown("""
    <div style='padding: 2rem; background-color: #f0f2f6; border-radius: 10px;'>
        <h2>About This App</h2>
        <p>
            This application helps cluster MSMEs based on characteristics such as business type, capital, revenue, and workforce.
            Using <strong>Agglomerative Hierarchical Clustering</strong> and <strong>Robust Clustering using Links (Ensemble ROCK)</strong>,
            it supports the government in formulating targeted policies to help MSMEs thrive and prosper.
        </p>
    </div>
    """, unsafe_allow_html=True)

def show_rules():
    st.markdown("""
    <div style='padding: 2rem; background-color: #f8f9fa; border-radius: 10px;'>
        <h2>⚠️ Important Guidelines</h2>
        <ul>
            <li>Upload files in <strong>.csv</strong> format (max 200MB).</li>
            <li>Ensure the following columns exist:
                <ul>
                    <li><code>modal</code>, <code>omset</code>, <code>tenaga_kerja</code>: numbers only (no dots or commas).</li>
                    <li><code>ojol</code>: values must be <strong>Ya</strong> or <strong>Tidak</strong>.</li>
                    <li><code>jenis</code>: values must be <strong>mamin</strong> or <strong>oleh</strong>.</li>
                </ul>
            </li>
            <li>Clean and format your data properly before uploading.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# =========================
# Tampilkan Halaman Aktif
# =========================
if st.session_state.page == "home":
    show_home()
elif st.session_state.page == "about":
    show_about()
elif st.session_state.page == "rules":
    show_rules()
