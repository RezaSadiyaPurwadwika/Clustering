import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import io

st.set_page_config(page_title="Clustering App", layout="wide")

# Sidebar dengan menu navigasi
menu = st.sidebar.radio("🔧 Menu Navigasi", [
    "Upload Data",
    "Data Preprocessing",
    "Clustering Numerik",
    "Clustering Kategorik",
    "Clustering Ensemble"
])

# Global storage
if "df" not in st.session_state:
    st.session_state.df = None
if "df_zscore" not in st.session_state:
    st.session_state.df_zscore = None

# ========== MENU 1: UPLOAD ==========
if menu == "Upload Data":
    st.title("📂 Upload Dataset UMKM")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("✅ File berhasil diunggah!")
        st.dataframe(df)

# ========== MENU 2: PREPROCESSING ==========
elif menu == "Data Preprocessing":
    st.title("⚙️ Data Preprocessing")
    if st.session_state.df is None:
        st.warning("⚠️ Silakan upload data terlebih dahulu.")
    else:
        df = st.session_state.df.copy()
        st.subheader("1. Membersihkan Data Kategorikal")
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
        st.write("Missing Values:")
        st.dataframe(df.isnull().sum())

        st.subheader("4. Boxplot Sebelum Penanganan Outlier")
        fig3, ax3 = plt.subplots()
        sns.boxplot(data=df[['omset', 'tenaga_kerja', 'modal']], ax=ax3)
        st.pyplot(fig3)

        # Tangani outlier
        for col in ['omset', 'modal']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        st.subheader("5. Boxplot Setelah Penanganan Outlier")
        fig4, ax4 = plt.subplots()
        sns.boxplot(data=df[['omset', 'tenaga_kerja', 'modal']], ax=ax4)
        st.pyplot(fig4)

        st.subheader("6. Normalisasi Z-Score")
        df_zscore = df.copy()
        for col in ['omset', 'tenaga_kerja', 'modal']:
            df_zscore[col] = zscore(df[col])
        st.session_state.df_zscore = df_zscore
        st.dataframe(df_zscore[['omset', 'tenaga_kerja', 'modal']].head())

# ========== MENU 3: CLUSTERING NUMERIK ==========
elif menu == "Clustering Numerik":
    st.title("📊 Clustering Numerik")
    st.info("Fitur clustering numerik akan dikembangkan...")

# ========== MENU 4: CLUSTERING KATEGORIK ==========
elif menu == "Clustering Kategorik":
    st.title("🧮 Clustering Kategorik")
    st.info("Fitur clustering kategorik akan dikembangkan...")

# ========== MENU 5: ENSEMBLE CLUSTERING ==========
elif menu == "Clustering Ensemble":
    st.title("🔗 Clustering Ensemble (ROCK)")
    st.info("Fitur ensemble clustering akan dikembangkan...")
