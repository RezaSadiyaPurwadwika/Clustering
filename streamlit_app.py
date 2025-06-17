import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
import io

st.set_page_config(page_title="Clustering App", layout="wide")

# Atur halaman awal
if "page" not in st.session_state:
    st.session_state.page = "home"

query_params = st.query_params
if "page" in query_params:
    st.session_state.page = query_params["page"]

# Styling CSS
st.markdown("""
    <style>
    .navbar {
        background-color: #002B5B;
        padding: 1rem;
        border-radius: 0 0 15px 15px;
        display: flex;
        justify-content: center;
        gap: 40px;
    }
    .nav-item {
        color: #ADD8E6;
        font-weight: bold;
        font-size: 18px;
        text-decoration: none;
        padding: 0.6rem 1.2rem;
        border-radius: 12px;
        transition: all 0.3s ease-in-out;
    }
    .nav-item:hover {
        background-color: #1E90FF;
        color: white !important;
    }
    .active {
        background-color: #87CEFA;
        color: #002B5B !important;
    }
    .hero {
        background-color: #E6F2FF;
        text-align: center;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 20px;
    }
    .hero h1 {
        color: #003366;
        font-size: 36px;
        margin-bottom: 0.5rem;
    }
    .hero p {
        color: #003366;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Navbar
page = st.session_state.page
st.markdown(f"""
<div class="navbar">
    <a href="/?page=home" class="nav-item {'active' if page == 'home' else ''}">🏠 Home</a>
    <a href="/?page=about" class="nav-item {'active' if page == 'about' else ''}">📋 About</a>
    <a href="/?page=rules" class="nav-item {'active' if page == 'rules' else ''}">📜 Rules</a>
</div>
""", unsafe_allow_html=True)

# =====================
# HOME PAGE
# =====================
if page == "home":
    st.markdown("""
    <div class="hero">
        <h1>Ensemble Clustering Using Links (ROCK)</h1>
        <p>Please read the About & Rules menu first.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📂 Upload Your CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        st.success("✅ File berhasil diunggah!")
        st.write("📌 Kolom terbaca:", df.columns.tolist())
        st.dataframe(df)

        run_preprocessing = st.button("🔧 Jalankan Preprocessing")

        if run_preprocessing:
            try:
                # Bersihkan kolom kategorikal
                if 'jenis' in df.columns:
                    df['jenis'] = df['jenis'].astype(str).str.strip().str.lower()
                if 'ojol' in df.columns:
                    df['ojol'] = df['ojol'].astype(str).str.strip().str.lower()

                # Distribusi kategori
if 'jenis' in df.columns or 'ojol' in df.columns:
    st.subheader("✅ Distribusi Kategori")
    col1, col2 = st.columns(2)

    if 'jenis' in df.columns:
        with col1:
            st.caption("Distribusi 'jenis'")
            fig1, ax1 = plt.subplots(figsize=(3.2, 2.5))
            sns.countplot(data=df, x='jenis', ax=ax1)
            ax1.set_title("Jenis", fontsize=10)
            ax1.tick_params(axis='x', labelsize=8)
            ax1.tick_params(axis='y', labelsize=8)
            st.pyplot(fig1)

    if 'ojol' in df.columns:
        with col2:
            st.caption("Distribusi 'ojol'")
            fig2, ax2 = plt.subplots(figsize=(3.2, 2.5))
            sns.countplot(data=df, x='ojol', ax=ax2)
            ax2.set_title("Ojol", fontsize=10)
            ax2.tick_params(axis='x', labelsize=8)
            ax2.tick_params(axis='y', labelsize=8)
            st.pyplot(fig2)

# Boxplot
st.subheader("📦 Boxplot Sebelum & Sesudah Penanganan Outlier")
col3, col4 = st.columns(2)

with col3:
    st.caption("Sebelum")
    fig3, ax3 = plt.subplots(figsize=(3.2, 2.5))
    sns.boxplot(data=df[num_cols], ax=ax3)
    ax3.set_title("Sebelum", fontsize=10)
    ax3.tick_params(axis='x', labelsize=8)
    ax3.tick_params(axis='y', labelsize=8)
    st.pyplot(fig3)

# Tangani outlier
for col in ['omset', 'modal']:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower, upper=upper)

with col4:
    st.caption("Sesudah")
    fig4, ax4 = plt.subplots(figsize=(3.2, 2.5))
    sns.boxplot(data=df[num_cols], ax=ax4)
    ax4.set_title("Sesudah", fontsize=10)
    ax4.tick_params(axis='x', labelsize=8)
    ax4.tick_params(axis='y', labelsize=8)
    st.pyplot(fig4)

    # Normalisasi Z-Score
    st.subheader("📈 Data Setelah Normalisasi Z-Score")
    df_zscore = df.copy()
    df_zscore[num_cols] = df_zscore[num_cols].apply(zscore)
    st.dataframe(df_zscore[num_cols].head())

            except Exception as e:
                st.error(f"🚨 Terjadi error saat preprocessing: {e}")

# =====================
# ABOUT PAGE
# =====================
elif page == "about":
    st.subheader("📋 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dirancang untuk mengelompokkan UMKM berdasarkan karakteristik usaha seperti jenis, modal, omset, dan tenaga kerja.  
    Dengan metode **Agglomerative Hierarchical Clustering** dan **Robust Clustering using Links (Ensemble ROCK)**, aplikasi ini membantu pemerintah dalam merumuskan kebijakan yang tepat sasaran sehingga UMKM dapat berkembang dan sejahtera.
    """)

# =====================
# RULES PAGE
# =====================
elif page == "rules":
    st.subheader("📜 Hal yang Perlu Diperhatikan")
    st.markdown("""
    File yang diunggah harus berformat **`.csv`** dan maksimal **200MB**.

    **Data harus memiliki kolom berikut:**
    - `modal`, `omset`, `tenaga_kerja`: isi dengan **angka bulat** tanpa titik atau koma.
    - `ojol`: isi dengan **"Ya"** atau **"Tidak"**.
    - `jenis`: isi dengan **"mamin"** (makanan/minuman) atau **"oleh"** (oleh-oleh).
    """)
