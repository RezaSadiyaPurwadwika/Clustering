import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import io
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import numpy as np

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
    st.info("💡 Akan dilakukan clustering terhadap `modal`, `omset`, dan `tenaga kerja`.")
    df_zscore = st.session_state.df_zscore
    df = st.session_state.df

    if df_zscore is None:
        st.warning("⚠️ Data belum tersedia. Lakukan preprocessing terlebih dahulu.")
    else:
        st.subheader("🔎 Validasi Clustering (Pseudo-F dan ICD Rate)")

        X = df_zscore[['omset', 'tenaga kerja', 'modal']]
        X_scaled = StandardScaler().fit_transform(X)

        n = len(X_scaled)
        global_mean = np.mean(X_scaled, axis=0)

        linkage_types = ['single', 'complete', 'average']
        best_result = {'k': None, 'link': None, 'PseudoF': -np.inf, 'ICD': np.inf}
        results = []

        for link in linkage_types:
            for k in range(2, 7):
                model = AgglomerativeClustering(n_clusters=k, linkage=link)
                labels = model.fit_predict(X_scaled)

                SW = 0
                SB = 0
                for cl in np.unique(labels):
                    cluster_data = X_scaled[labels == cl]
                    mean_cl = np.mean(cluster_data, axis=0)
                    SW += np.sum((cluster_data - mean_cl) ** 2)
                    SB += len(cluster_data) * np.sum((mean_cl - global_mean) ** 2)

                pseudoF = (SB / (k - 1)) / (SW / (n - k)) if SW != 0 else np.inf
                ICD = SW / n
                results.append((link, k, pseudoF, ICD))

                if pseudoF > best_result['PseudoF']:
                    best_result = {'k': k, 'link': link, 'PseudoF': pseudoF, 'ICD': ICD}

        # Tampilkan hasil validasi
        result_df = pd.DataFrame(results, columns=['Linkage', 'K', 'Pseudo-F', 'ICD'])
        st.dataframe(result_df.style.format({'Pseudo-F': '{:.4f}', 'ICD': '{:.4f}'}))

        # Tampilkan hasil terbaik
        st.subheader("🏆 Hasil Clustering Terbaik")
        st.markdown(f"""
        - Jumlah klaster optimum: **{best_result['k']}**
        - Metode linkage terbaik: **{best_result['link'].capitalize()}**
        - Nilai Pseudo-F tertinggi: **{best_result['PseudoF']:.4f}**
        - Nilai ICD terkecil: **{best_result['ICD']:.4f}**
        """)

        # ========== Visualisasi hasil clustering ==========
        st.subheader("🎨 Visualisasi Clustering")

        # 1. Clustering dengan parameter terbaik
        best_model = AgglomerativeClustering(n_clusters=best_result['k'], linkage=best_result['link'])
        best_labels = best_model.fit_predict(X_scaled)

        # Simpan ke dataframe
        df['cluster_numerik'] = best_labels
        st.session_state.df = df  # Simpan kembali ke session_state

        # 2. Reduksi dimensi dengan t-SNE
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        X_reduced = tsne.fit_transform(X_scaled)

        # 3. Plot t-SNE
        fig_tsne, ax_tsne = plt.subplots(figsize=(8, 6))
        for cl in np.unique(best_labels):
            ax_tsne.scatter(
                X_reduced[best_labels == cl, 0],
                X_reduced[best_labels == cl, 1],
                label=f'Cluster {cl+1}'
            )
        ax_tsne.set_title(f't-SNE Clustering\nLinkage={best_result["link"].upper()}, k={best_result["k"]}')
        ax_tsne.set_xlabel('t-SNE 1')
        ax_tsne.set_ylabel('t-SNE 2')
        ax_tsne.legend()
        ax_tsne.grid(True)
        st.pyplot(fig_tsne)

        # 4. Plot dendrogram
        st.subheader("🧬 Dendrogram Hierarki")
        from scipy.cluster.hierarchy import dendrogram, linkage
        linked = linkage(X_scaled, method=best_result['link'])
        fig_dendro, ax_dendro = plt.subplots(figsize=(10, 6))
        dendrogram(linked, ax=ax_dendro, orientation='top', distance_sort='descending', show_leaf_counts=False)
        ax_dendro.set_title(f'Dendrogram Linkage={best_result["link"].upper()}')
        ax_dendro.set_xlabel('Data')
        ax_dendro.set_ylabel('Jarak (Distance)')
        st.pyplot(fig_dendro)

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
