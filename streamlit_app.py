import streamlit as st
import pandas as pd

st.set_page_config(page_title="Ensemble Rock Clustering", layout="wide")

# === Navigasi radio button ===
page = st.radio(
    label="Navigation",
    options=["Home", "About", "Rules"],
    horizontal=True,
    label_visibility="collapsed",
)

# === CSS Kustom untuk radio ===
st.markdown("""
    <style>
        /* Container radio */
        .stRadio > div {
            background-color: #003366;
            padding: 1rem 2rem;
            border-radius: 0 0 10px 10px;
            display: flex;
            justify-content: center;
            gap: 2rem;
        }

        /* Label radio (semua opsi) */
        .stRadio label {
            color: white !important;
            font-weight: bold;
            font-size: 1.1rem;
            background-color: transparent !important;
            padding: 0.3rem 1rem;
            border-radius: 8px;
            transition: background-color 0.3s ease;
        }

        /* Label yang aktif (dipilih) */
        .stRadio div[data-baseweb="radio"] > div > div > label[data-selected="true"] {
            background-color: #87CEFA !important;
            color: #003366 !important;
        }
    </style>
""", unsafe_allow_html=True)

# === Fungsi Halaman ===
def show_home():
    st.markdown("""
        <div style='padding: 4rem 2rem; background-color: #87CEFA; color: #003366; text-align: center; border-radius: 10px'>
            <h1><strong>Better Solutions for Your Clustering</strong></h1>
            <p>Please read the About & Rules menu first.</p>
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

# === Halaman sesuai navigasi ===
if page == "Home":
    show_home()
elif page == "About":
    show_about()
elif page == "Rules":
    show_rules()
