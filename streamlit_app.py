import streamlit as st
import pandas as pd

# ===== Page Config =====
st.set_page_config(page_title="Arsha Style Clustering", layout="wide")

# ===== Simulasi Navbar =====
st.markdown(
    """
    <style>
        .navbar {
            background-color: #0d6efd;
            padding: 1rem;
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
        .hero {
            padding: 4rem 2rem;
            background-color: #37517e;
            color: white;
            text-align: center;
        }
        .upload-box {
            margin-top: 2rem;
            padding: 2rem;
            background-color: #f8f9fa;
            border-radius: 10px;
        }
    </style>
    <div class="navbar">
        <div><strong>ARSHACLUST</strong></div>
        <div>
            <a href="#">Home</a>
            <a href="#">About</a>
            <a href="#">Services</a>
            <a href="#">Contact</a>
        </div>
    </div>
    <div class="hero">
        <h1>Better Solutions for Your Clustering</h1>
        <p>Upload your data and get started with powerful clustering tools</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ===== Upload Section =====
with st.container():
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    st.subheader("📂 Upload Your CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview Data")
        st.dataframe(df)
    
    st.markdown('</div>', unsafe_allow_html=True)
