import streamlit as st
import pandas as pd

# Konfigurasi halaman
st.set_page_config(page_title="Ensemble Rock Clustering", layout="wide")

# ==========================
# Navigasi State
# ==========================
if "page" not in st.session_state:
    st.session_state.page = "home"

# ==========================
# Fungsi Navigasi
# ==========================
def go_home():
    st.session_state.page = "home"

def go_about():
    st.session_state.page = "about"

def go_rules():
    st.session_state.page = "rules"

# ==========================
# CSS dan Navbar
# ==========================
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
""", unsafe_allow_html=True)

# ==========================
# Navbar
# ==========================
st.markdown("""
<div class="navbar">
    <div><strong>🌐 Ensemble Rock Clustering</strong></div>
    <div>
        <a href="#" onclick="window.location.reload()">Home</a>
        <a href="#" onclick="window.location.search='?page=about'">About</a>
        <a href="#" onclick="window.location.search='?page=rules'">Rules</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================
# Komponen Halaman
# ==========================
def show_home():
    st.markdown("""
        <div class="hero">
            <h1><strong>Better Solutions for Your Clustering</strong></h1>
            <p>Read the About menu before uploading data</p>
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
            This application helps cluster MSMEs based on characteristics such as business type, capital, revenue, and workforce.
            Using <strong>Agglomerative Hierarchical Clustering</strong> and <strong>Robust Clustering using Links (Ensemble ROCK)</strong>,
            it supports the government in formulating targeted policies to help MSMEs thrive and prosper.
        </p>
    </div>
    """, unsafe_allow_html=True)


def show_rules():
    st.markdown("""
    <div style='padding: 3rem 2rem; background-color: #f8f9fa; border-radius: 10px;'>
        <h2>⚠️ Important Guidelines</h2>
        <ul>
            <li>Upload files in <strong>.csv</strong> format (max 200MB).</li>
            <li>Ensure the following columns exist:
                <ul>
                    <li><code>modal</code>, <code>omset</code>, <code>tenaga_kerja</code>: whole numbers (no dots or commas).</li>
                    <li><code>ojol</code>: values must be <strong>Ya</strong> or <strong>Tidak</strong>.</li>
                    <li><code>jenis</code>: values must be <strong>mamin</strong> or <strong>oleh</strong>.</li>
                </ul>
            </li>
            <li>Clean and format your data properly before uploading.</li>
            <li>It is recommended to read the <strong>About</strong> page before uploading your data.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ==========================
# Routing Berdasarkan Query
# ==========================
query_params = st.query_params
page = query_params.get("page", ["home"])[0]

if page == "about":
    show_about()
elif page == "rules":
    show_rules()
else:
    show_home()
