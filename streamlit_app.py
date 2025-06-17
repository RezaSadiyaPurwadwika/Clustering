import streamlit as st
import pandas as pd

# ========== Konfigurasi Halaman ==========
st.set_page_config(page_title="ArshaClust Landing", layout="wide")

# ========== CSS + HTML ==========
st.markdown(
    """
    <style>
        /* Navbar style */
        .navbar {
            background-color: #007bff;
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

        /* Hero section */
        .hero {
            padding: 4rem 2rem 2rem;
            background-color: #37517e;
            color: white;
            text-align: center;
        }

        /* Hapus kotak hitam */
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
        <div><strong>ARSHACLUST</strong></div>
        <div>
            <a href="#">Home</a>
            <a href="#">About</a>
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

# ========== Upload File ==========
uploaded_file = st.file_uploader("", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(df)

# ========== Tutup Div ==========
st.markdown(
    """
        </div> <!-- end upload clean -->
    </div> <!-- end hero -->
    """,
    unsafe_allow_html=True
)
