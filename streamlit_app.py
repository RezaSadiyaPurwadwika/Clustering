import streamlit as st
import pandas as pd

# ========== Konfigurasi Halaman ==========
st.set_page_config(page_title="ArshaClust Landing", layout="wide")

# ========== Custom CSS & Layout ==========
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
            padding: 6rem 2rem 4rem;
            background-color: #37517e;
            color: white;
            text-align: center;
        }

        /* Upload box center */
        .upload-box {
            text-align: center;
            margin-top: 2rem;
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
    </div>
    """,
    unsafe_allow_html=True
)

# ========== Upload File Section ==========
with st.container():
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.subheader("📂 Upload Your CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded Data")
        st.dataframe(df)
    
    st.markdown('</div>', unsafe_allow_html=True)
