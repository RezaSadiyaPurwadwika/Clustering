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
        """)
    with tab2:
        st.markdown("""
        ### Aturan Penggunaan
        **Format CSV wajib memuat kolom:**
        - `modal`, `omset`, `tenaga_kerja`
        - `ojol`: "ya" / "tidak"
        - `jenis`: "mamin" / "oleh"
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
            cols_num = ['omset', 'tenaga kerja', 'modal']
            cols_ada = [col for col in cols_num if col in df.columns]
            if cols_ada:
                for col in cols_ada:
                    Q1, Q3 = df[col].quantile([0.25, 0.75])
                    IQR = Q3 - Q1
                    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                    df[col] = df[col].clip(lower, upper)
                df_zscore = df.copy()
                df_zscore[cols_ada] = df_zscore[cols_ada].apply(zscore)
                st.session_state.df_zscore = df_zscore
                st.success("✅ Preprocessing selesai.")
            else:
                st.warning("Tidak ada kolom numerik yang ditemukan.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat preprocessing: {e}")

# ================= CLUSTERING NUMERIK =================
elif menu == "📊 Clustering Numerik":
    st.title("📊 Clustering Data Numerik")
    df_zscore = st.session_state.df_zscore
    df = st.session_state.df
    if df_zscore is None:
        st.warning("⚠️ Data belum tersedia. Lakukan preprocessing terlebih dahulu.")
    else:
        try:
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
                    SW, SB = 0, 0
                    for cl in np.unique(labels):
                        cluster_data = X_scaled[labels == cl]
                        mean_cl = np.mean(cluster_data, axis=0)
                        SW += np.sum((cluster_data - mean_cl)**2)
                        SB += len(cluster_data) * np.sum((mean_cl - global_mean)**2)
                    pseudoF = (SB / (k - 1)) / (SW / (n - k)) if SW != 0 else np.inf
                    ICD = SW / n
                    results.append((link, k, pseudoF, ICD))
                    if pseudoF > best_result['PseudoF']:
                        best_result = {'k': k, 'link': link, 'PseudoF': pseudoF, 'ICD': ICD}
            st.subheader("🏆 Hasil Clustering Terbaik")
            st.markdown(
                f"- Jumlah klaster optimum: **{best_result['k']}**\n"
                f"- Metode linkage terbaik: **{best_result['link'].capitalize()}**\n"
                f"- Pseudo-F: **{best_result['PseudoF']:.4f}**\n"
                f"- ICD: **{best_result['ICD']:.4f}**"
            )

            best_model = AgglomerativeClustering(n_clusters=best_result['k'], linkage=best_result['link'])
            best_labels = best_model.fit_predict(X_scaled)
            df['cluster_numerik'] = best_labels
            st.session_state.df = df

            # Visualisasi t-SNE
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=5, n_iter=1000)
            X_reduced = tsne.fit_transform(X_scaled)
            st.subheader("🔸 Visualisasi t-SNE")
            fig_tsne, ax_tsne = plt.subplots(figsize=(8, 6))
            for cl in np.unique(best_labels):
                ax_tsne.scatter(
                    X_reduced[best_labels == cl, 0],
                    X_reduced[best_labels == cl, 1],
                    label=f'Cluster {cl+1}'
                )
            ax_tsne.set_title(f't-SNE Clustering\\nLinkage={best_result["link"].upper()}, k={best_result["k"]}')
            ax_tsne.set_xlabel('t-SNE 1')
            ax_tsne.set_ylabel('t-SNE 2')
            ax_tsne.legend()
            ax_tsne.grid(True)
            st.pyplot(fig_tsne)

            # Visualisasi dendrogram
            st.subheader("🧬 Dendrogram Hierarki")
            from scipy.cluster.hierarchy import dendrogram, linkage
            linked = linkage(X_scaled, method=best_result['link'])
            fig_dendro, ax_dendro = plt.subplots(figsize=(10, 6))
            dendrogram(linked, ax=ax_dendro, orientation='top', distance_sort='descending', show_leaf_counts=False)
            ax_dendro.set_title(f'Dendrogram Linkage={best_result["link"].upper()}')
            ax_dendro.set_xlabel('Data')
            ax_dendro.set_ylabel('Jarak (Distance)')
            st.pyplot(fig_dendro)

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {e}")

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
