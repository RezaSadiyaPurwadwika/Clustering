import streamlit as st
import pandas as pd

# Konfigurasi halaman
st.set_page_config(page_title="ArshaClust Landing", layout="wide")

# HTML + CSS
st.markdown(
    """
    <style>
        /* Navbar: warna biru gelap */
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
        }
        .navbar a:hover {
            text-decoration: underline;
        }

        /* Hero section: warna baby blue */
        .hero {
            padding: 4rem 2rem 2rem;
            background-color: #87CEFA;
            color: #003366;
            text-align: center;
        }

        /* File uploader */
        .file-upload-clean {
            margin-top: 2rem;
            width: 50%;
            margin-left: auto;
            margin-right: auto;
        }

        .stFileUploader > label {
            display: none;
        }
    </style>

    <!-- Navbar -->
    <div class="navbar">
        <div><strong>Ensemble Rock Clustering</strong></div>
        <div>
            <a href="#">Home</a>
            <a href="#">Rules</a>
        </div>
    </div>

    <!-- Hero Section -->
    <div class="hero">
        <h1><strong>Better Solutions for Your Clustering</strong></h1>
        <p>Upload your data and get started with powerful clustering tools</p>
        <div class="file-upload-clean">
    """,
    unsafe_allow_html=True
)

# Upload file
uploaded_file = st.file_uploader("", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(df)

# Tutup div
st.markdown(
    """
        </div> <!-- end upload clean -->
    </div> <!-- end hero -->
    """,
    unsafe_allow_html=True
)
