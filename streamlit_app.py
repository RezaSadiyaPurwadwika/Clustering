import streamlit as st
import pandas as pd

st.set_page_config(page_title="Clustering App", layout="wide")

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
        color: #ADD8E6; /* baby blue */
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
        background-color: #87CEFA; /* light sky blue */
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

# Tampilkan navbar
page = st.session_state.page
st.markdown(f"""
<div class="navbar">
    <a href="/?page=home" class="nav-item {'active' if page == 'home' else ''}">🏠 Home</a>
    <a href="/?page=about" class="nav-item {'active' if page == 'about' else ''}">📋 About</a>
    <a href="/?page=rules" class="nav-item {'active' if page == 'rules' else ''}">📜 Rules</a>
</div>
""", unsafe_allow_html=True)

# Konten halaman
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
    st.success("✅ File berhasil diunggah!")
    st.dataframe(df)

    # Tampilkan tombol di luar blok proses berat
    run_preprocessing = st.button("🔧 Jalankan Preprocessing")

    if run_preprocessing:
        # Mulai PREPROCESSING
        df['jenis'] = df['jenis'].str.strip().str.lower()
        df['ojol'] = df['ojol'].str.strip().str.lower()

        import seaborn as sns
        import matplotlib.pyplot as plt
        import io

        st.subheader("✅ Distribusi Kategori 'jenis'")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df, x='jenis', ax=ax1)
        ax1.set_title("Distribusi Kategori: jenis")
        st.pyplot(fig1)

        st.subheader("✅ Distribusi Kategori 'ojol'")
        fig2, ax2 = plt.subplots()
        sns.countplot(data=df, x='ojol', ax=ax2)
        ax2.set_title("Distribusi Kategori: ojol")
        st.pyplot(fig2)

        st.subheader("ℹ️ Info Dataset")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

        st.subheader("📊 Statistik Deskriptif (Numerik)")
        st.dataframe(df[['omset', 'tenaga_kerja', 'modal']].describe())

        st.subheader("🔍 Missing Values")
        st.dataframe(df.isnull().sum())

        # Boxplot sebelum outlier handling
        st.subheader("📦 Boxplot Sebelum Penanganan Outlier")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df[['omset', 'tenaga_kerja', 'modal']], ax=ax3)
        ax3.set_title('Boxplot Kolom Omset, Tenaga Kerja, dan Modal')
        st.pyplot(fig3)

        # Outlier handling
        for col in ['omset', 'modal']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        # Boxplot setelah outlier handling
        st.subheader("📦 Boxplot Setelah Penanganan Outlier")
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df[['omset', 'tenaga_kerja', 'modal']], ax=ax4)
        ax4.set_title('Boxplot Setelah Penanganan Outlier')
        st.pyplot(fig4)

        # Normalisasi Z-Score
        from scipy.stats import zscore
        df_zscore = df.copy()
        cols_to_normalize = ['omset', 'tenaga_kerja', 'modal']
        df_zscore[cols_to_normalize] = df_zscore[cols_to_normalize].apply(zscore)

        st.subheader("📈 Data Setelah Normalisasi Z-Score")
        st.dataframe(df_zscore[cols_to_normalize].head())

elif page == "about":
    st.subheader("📋 Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dirancang untuk mengelompokkan UMKM berdasarkan karakteristik usaha seperti jenis, modal, omset, dan tenaga kerja.  
    Dengan metode **Agglomerative Hierarchical Clustering** dan **Robust Clustering using Links (Ensemble ROCK)**, aplikasi ini membantu pemerintah dalam merumuskan kebijakan yang tepat sasaran sehingga UMKM dapat berkembang dan sejahtera.
    """)

elif page == "rules":
    st.subheader("📜 Hal yang Perlu Diperhatikan")

    st.markdown("""
    File yang diunggah harus berformat **`.csv`** dan maksimal **200MB**.

    **Data harus memiliki kolom berikut:**
    - `modal`, `omset`, `tenaga_kerja`: isi dengan **angka bulat** tanpa titik atau koma.
    - `ojol`: isi dengan **"Ya"** atau **"Tidak"**.
    - `jenis`: isi dengan **"mamin"** (makanan/minuman) atau **"oleh"** (oleh-oleh).
    """)

