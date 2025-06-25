import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import io

# Konfigurasi halaman
st.set_page_config(page_title="Clustering UMKM", layout="wide")

# Sidebar Navigasi
st.sidebar.title("🛍️ Menu Navigasi")
menu = st.sidebar.radio("Pilih halaman:", [
    "🏠 Home",
    "📂 Upload Data",
    "⚙️ Data Preprocessing",
    "📊 Clustering Numerik",
    "🧮 Clustering Kategorik",
    "🔗 Clustering Ensemble"
])

# Inisialisasi
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
        Aplikasi ini dirancang untuk mengelompokkan data UMKM di Kabupaten Malang menggunakan:
        - Agglomerative Hierarchical Clustering (AHC)
        - Robust Clustering using Links (Ensemble ROCK)

        Aplikasi membantu analisis dan pemetaan kelompok usaha berdasarkan karakteristik numerik dan kategorikal.
        """)
    
    with tab2:
        st.markdown("""
        ### Aturan Penggunaan
        **Format CSV wajib memuat kolom:**
        - `modal`, `omset`, `tenaga_kerja`: angka bulat
        - `ojol`: "ya" / "tidak"
        - `jenis`: "mamin" / "oleh"

        **Ukuran file maksimal:** 200MB
        """)

# ================= UPLOAD =================
elif menu == "📂 Upload Data":
    st.title("📂 Upload Dataset UMKM")
    uploaded_file = st.file_uploader("Unggah file CSV", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success("✅ File berhasil diunggah!")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")

# ================= PREPROCESSING =================
elif menu == "⚙️ Data Preprocessing":
    st.title("⚙️ Tahap Preprocessing Data")
    df = st.session_state.df
    if df is None:
        st.warning("⚠️ Silakan unggah data terlebih dahulu.")
    else:
        try:
            df['jenis'] = df['jenis'].str.strip().str.lower()
            df['ojol'] = df['ojol'].str.strip().str.lower()

            st.subheader("1. Distribusi Kategori")
            col1, col2 = st.columns(2)
            with col1:
                fig1, ax1 = plt.subplots()
                sns.countplot(data=df, x='jenis', ax=ax1)
                st.pyplot(fig1)
            with col2:
                fig2, ax2 = plt.subplots()
                sns.countplot(data=df, x='ojol', ax=ax2)
                st.pyplot(fig2)

            st.subheader("2. Info Dataset & Statistik Deskriptif")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())

            print("Info Dataset:")
            print(df.info())

            cols_num = ['omset', 'tenaga kerja', 'modal']
            cols_ada = [col for col in cols_num if col in df.columns]
            if cols_ada:
                st.dataframe(df[cols_ada].describe())

                print("\nStatistik Deskriptif Variabel Numerik:")
                print(df[cols_ada].describe())
            else:
                st.error("❌ Tidak ditemukan kolom numerik yang sesuai.")

            st.subheader("3. Cek Missing Values")
            st.dataframe(df.isnull().sum())
            print("\nMissing Values:\n", df.isnull().sum())

            if cols_ada:
                st.subheader("4. Boxplot Sebelum Outlier Handling")
                fig3, ax3 = plt.subplots()
                sns.boxplot(data=df[cols_ada], ax=ax3)
                st.pyplot(fig3)

                for col in ['omset', 'tenaga kerja', 'modal']:
                    if col in df.columns:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        df[col] = df[col].clip(lower=lower, upper=upper)

                st.subheader("5. Boxplot Setelah Outlier Handling")
                fig4, ax4 = plt.subplots()
                sns.boxplot(data=df[cols_ada], ax=ax4)
                st.pyplot(fig4)

                st.subheader("6. Normalisasi Z-Score")
                df_zscore = df.copy()
                df_zscore[cols_ada] = df_zscore[cols_ada].apply(zscore)
                st.session_state.df_zscore = df_zscore
                st.dataframe(df_zscore[cols_ada].head())
            else:
                st.info("Data numerik tidak ditemukan, normalisasi dilewati.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat preprocessing: {e}")

# ================= CLUSTERING NUMERIK =================
elif menu == "📊 Clustering Numerik":
    st.title("📊 Clustering Data Numerik")
    st.info("💡 Akan dilakukan clustering terhadap `modal`, `omset`, dan `tenaga_kerja`.")
    st.warning("Fitur ini akan dikembangkan setelah preprocessing selesai.")

# ================= CLUSTERING KATEGORIK =================
elif menu == "🧮 Clustering Kategorik":
    st.title("🧮 Clustering Data Kategorik")
    st.info("💡 Akan dilakukan clustering terhadap `jenis` dan `ojol`.")
    st.warning("Fitur dalam tahap pengembangan.")

# ================= CLUSTERING ENSEMBLE =================
elif menu == "🔗 Clustering Ensemble":
    st.title("🔗 Clustering Ensemble (ROCK)")
    st.info("💡 Menggabungkan hasil clustering numerik dan kategorik.")
    st.warning("Fitur dalam tahap pengembangan.")
