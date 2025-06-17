import streamlit as st
import pandas as pd

# Set page config
st.set_page_config(page_title="Ensemble Rock Clustering", layout="wide")

# Inisialisasi page state
if "page" not in st.session_state:
    st.session_state.page = "home"

# =========================
# Navbar (dengan tombol Streamlit)
# =========================
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    st.markdown("### 🌐 Ensemble Rock Clustering")
with col2:
    if st.button("Home"):
        st.session_state.page = "home"
with col3:
    if st.button("About"):
        st.session_state.page = "about"
with col4:
    if st.button("Rules"):
        st.session_state.page = "rules"

# =========================
# Tampilan Halaman
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

# =========================
# Render halaman sesuai state
# =========================
if st.session_state.page == "home":
    show_home()
elif st.session_state.page == "about":
    show_about()
elif st.session_state.page == "rules":
    show_rules()
