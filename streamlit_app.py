import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import io

# Konfigurasi layout
st.set_page_config(page_title="Clustering App", layout="wide")

# Sidebar Menu
st.sidebar.title("🧭 Menu Navigasi")
menu = st.sidebar.radio("Pilih halaman:", [
    "🏠 Home",
    "📂 Upload Data",
    "⚙️ Data Preprocessing",
    "📊 Clustering Numerik",
    "🧮 Clustering Kategorik",
    "🔗 Clustering Ensemble"
])

# Inisialisasi variabel global
if "df" not in st.session_state:
    st.session_state.df = None
if "df_zscore" not in st.session_state:
    st.session_state.df_zscore = None

# ================= HOME =================
if menu == "🏠 Home":
    st.title("📘 Selamat Datang di Aplikasi Clustering UMKM")
    
    tab1, tab2 = st.tabs(["📋 About", "📜 Rules"])
    
    with tab1:
        st.markdown("""
        ### Tentang Aplikasi
        Aplikasi ini dirancang untuk mengelompokkan data UMKM di Kabupaten Malang menggunakan metode:
        - Agglomerative Hierarchical Clustering (AHC)
        - Robust Clustering using Links (Ensemble ROCK)
        
        Aplikasi ini membantu analisis dan pemetaan kelompok usaha berdasarkan karakteristik numerik dan kategorikal.
        """)
    
    with tab2:
        st.markdown("""
        ### Aturan Penggunaan
        **Format data CSV wajib memuat kolom berikut:**
        - `modal`, `omset`, `tenaga_kerja`: berupa **angka bulat** tanpa titik/koma.
        - `ojol`: hanya berisi nilai `"ya"` atau `"tidak"`.
        - `jenis`: hanya berisi nilai `"mamin"` (makanan/minuman) atau `"oleh"` (oleh-oleh).
        
        **Ukuran file maksimal:** 200MB
        """)

# ================= UPLOAD =================
elif menu == "📂 Upload Data":
    st.title("📂 Upload Dataset UMKM")
    uploaded_file = st.file_uploader("Unggah file CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("✅ File berhasil diunggah!")
        st.dataframe(df)

# ================= PREPROCESSING =================
elif menu == "⚙️ Data Preprocessing":
    st.title("⚙️ Tahap Preprocessing Data")
    if st.session_state.df is None:
        st.warning("⚠️ Silakan unggah data terlebih dahulu di menu 'Upload Data'.")
    else:
        df = st.session_state.df.copy()
        st.subheader("1. Pembersihan Data Kategorikal")
        df['jenis'] = df['jenis'].str.strip().str.lower()
        df['ojol'] = df['ojol'].str.strip().str.lower()

        st.subheader("2. Distribusi Kategori")
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots()
            sns.countplot(data=df, x='jenis', ax=ax1)
            ax1.set_title("Distribusi 'jenis'")
            st.pyplot(fig1)
        with col2:
            fig2, ax2 = plt.subplots()
            sns.countplot(data=df, x='ojol', ax=ax2)
            ax2.set_title("Distribusi 'ojol'")
            st.pyplot(fig2)

        st.subheader("3. Info Dataset & Statistik")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

        st.dataframe(df[['omset', 'tenaga_kerja', 'modal']].describe())

        st.subheader("4. Missing Values")
        st.dataframe(df.isnull().sum())

        st.subheader("5. Boxplot Sebelum Penanganan Outlier")
        fig3, ax3 = plt.subplots()
        sns.boxplot(data=df[['omset', 'tenaga_kerja', 'modal']], ax=ax3)
        st.pyplot(fig3)

        for col in ['omset', 'modal']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower, upper=upper)

        st.subheader("6. Boxplot Setelah Penanganan Outlier")
        fig4, ax4 = plt.subplots()
        sns.boxplot(data=df[['omset', 'tenaga_kerja', 'modal']], ax=ax4)
        st.pyplot(fig4)

        st.subheader("7. Normalisasi Z-Score")
        df_zscore = df.copy()
        df_zscore[['omset', 'tenaga_kerja', 'modal']] = df_zscore[['omset', 'tenaga_kerja', 'modal']].apply(zscore)
        st.session_state.df_zscore = df_zscore
        st.dataframe(df_zscore.head())

# ================= CLUSTERING NUMERIK =================
elif menu == "📊 Clustering Numerik":
    st.title("📊 Clustering Data Numerik")
    st.info("💡 Fitur ini akan memproses clustering berdasarkan nilai `omset`, `tenaga_kerja`, dan `modal`.")
    st.warning("Fitur ini akan diisi setelah preprocessing dan implementasi algoritma clustering.")

# ================= CLUSTERING KATEGORIK =================
elif menu == "🧮 Clustering Kategorik":
    st.title("🧮 Clustering Data Kategorik")
    st.info("💡 Fitur ini akan mengelompokkan data berdasarkan `jenis` dan `ojol`.")
    st.warning("Fitur sedang dalam pengembangan.")

# ================= CLUSTERING ENSEMBLE =================
elif menu == "🔗 Clustering Ensemble":
    st.title("🔗 Clustering Ensemble (ROCK)")
    st.info("💡 Clustering Ensemble menggabungkan hasil numerik dan kategorik menggunakan pendekatan ROCK.")
    st.warning("Fitur sedang dalam pengembangan.")

