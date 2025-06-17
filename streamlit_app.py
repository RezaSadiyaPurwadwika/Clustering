import streamlit as st
import pandas as pd

# Konfigurasi halaman
st.set_page_config(page_title="Ensemble Rock Clustering", layout="wide")

# ==============================
# State Awal untuk Navigasi
# ==============================
if "page" not in st.session_state:
    st.session_state.page = "home"

# ==============================
# Fungsi Navigasi
# ==============================
def go_home():
    st.session_state.page = "home"

def go_about():
    st.session_state.page = "about"

# ==============================
# CSS dan Navbar HTML
# ==============================
st.markdown("""
<style>
    .navbar {
        background-color: #37517e;
        padding: 1rem 2rem;
        color: white;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .navbar a {
        color: white;
        margin-left: 2rem;
        text-decoration: none;
        font-weight: bold;
        cursor: pointer;
    }
    .navbar a:hover {
        text-decoration: underline;
    }
    .hero {
        padding: 4rem 2rem 2rem;
        background-color: #87CEFA;
        color: #003366;
        text-align: center;
    }
    .file-upload-clean {
        margin-top: 2rem;
        width: 50%;
        margin-left: auto;
        margin-right: auto;
    }
</style>

<!-- Navbar -->
<div class="navbar">
    <div><strong>🌐 Ensemble Rock Clustering</strong></div>
    <div>
        <a onClick="window.location.reload()">Home</a>
        <a href="#" onclick="window.parent.postMessage('about','*')">About</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================
# Komponen Halaman
# ==============================
def show_home():
    st.markdown("""
        <div class="hero">
            <h1><strong>Better Solutions for Your Clustering</strong></h1>
            <p>before uploading your data, it is recommended to read the information in the About menu first</p>
            <div class="file-upload-clean">
        """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data")
        st.dataframe(df)

    st.markdown("</div></div>", unsafe_allow_html=True)


def show_about():
    st.markdown("""
        <div style='padding: 3rem 2rem; background-color: #f0f2f6; border-radius: 10px;'>
            <h2>About This App</h2>
            <p>
                This application is designed to assist in clustering analysis using the Ensemble ROCK method.
                It allows you to upload CSV data, preprocess, and explore advanced clustering capabilities.
            </p>
            <ul>
                <li>Built with Streamlit</li>
                <li>Supports CSV uploads up to 200MB</li>
                <li>Optimized for mixed data types</li>
            </ul>
            <button onclick="window.location.reload()">🔙 Back to Home</button>
        </div>
        """, unsafe_allow_html=True)

# ==============================
# Tampilan Berdasarkan State
# ==============================
query_params = st.query_params
if "about" in query_params:
    show_about()
else:
    show_home()
