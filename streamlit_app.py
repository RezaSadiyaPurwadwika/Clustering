import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io

from scipy.stats import zscore
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder  # ⬅️ Tambahkan LabelEncoder di sini
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from itertools import combinations
from sklearn.manifold import TSNE  # ⬅️ Diperlukan untuk t-SNE di ROCK

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
    "🔗 Clustering Ensemble",
    "📏 Evaluasi Clustering",
    "🧾 Interpretasi Hasil"
])

# Inisialisasi session state
if "df" not in st.session_state:
    st.session_state.df = None
if "df_zscore" not in st.session_state:
    st.session_state.df_zscore = None

# =============== HOME ===============
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
        - `modal`, `omset`, `tenaga_kerja`: angka bulat
        - `ojol`: "ya" / "tidak"
        - `jenis`: "mamin" / "oleh"
        """)

# =============== UPLOAD ===============
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

# =============== PREPROCESSING ===============
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

            st.subheader("2. Statistik Deskriptif")
            cols_num = ['omset', 'tenaga kerja', 'modal']
            st.dataframe(df[cols_num].describe())

            st.subheader("3. Missing Values")
            st.dataframe(df.isnull().sum())

            st.subheader("4. Boxplot Sebelum Outlier Handling")
            fig3, ax3 = plt.subplots()
            sns.boxplot(data=df[cols_num], ax=ax3)
            st.pyplot(fig3)

            for col in cols_num:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower=lower, upper=upper)

            st.subheader("5. Boxplot Setelah Outlier Handling")
            fig4, ax4 = plt.subplots()
            sns.boxplot(data=df[cols_num], ax=ax4)
            st.pyplot(fig4)

            df_zscore = df.copy()
            df_zscore[cols_num] = df_zscore[cols_num].apply(zscore)
            st.session_state.df_zscore = df_zscore
            st.dataframe(df_zscore[cols_num].head())
        except Exception as e:
            st.error(f"Terjadi kesalahan saat preprocessing: {e}")

# =============== CLUSTERING NUMERIK ===============
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

            result_df = pd.DataFrame(results, columns=['Linkage', 'K', 'Pseudo-F', 'ICD'])
            st.dataframe(result_df.style.format({'Pseudo-F': '{:.4f}', 'ICD': '{:.4f}'}))

            st.subheader("🏆 Hasil Clustering Terbaik")
            st.markdown(f"""
            - Jumlah klaster optimum: **{best_result['k']}**
            - Metode linkage terbaik: **{best_result['link'].capitalize()}**
            - Nilai Pseudo-F tertinggi: **{best_result['PseudoF']:.4f}**
            - Nilai ICD terkecil: **{best_result['ICD']:.4f}**
            """)

            # Clustering & visualisasi t-SNE dan dendrogram
            from sklearn.manifold import TSNE
            from scipy.cluster.hierarchy import dendrogram, linkage

            best_model = AgglomerativeClustering(n_clusters=best_result['k'], linkage=best_result['link'])
            best_labels = best_model.fit_predict(X_scaled)
            df['cluster_numerik'] = best_labels
            st.session_state.df = df

            # t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            X_reduced = tsne.fit_transform(X_scaled)

            st.subheader("🔸 Visualisasi t-SNE")
            fig_tsne = plt.figure(figsize=(8, 6))
            for cl in np.unique(best_labels):
                plt.scatter(
                    X_reduced[best_labels == cl, 0],
                    X_reduced[best_labels == cl, 1],
                    label=f'Cluster {cl+1}'
                )
            plt.title(f'Visualisasi Clustering dengan t-SNE\nLinkage={best_result["link"].upper()}, k={best_result["k"]}')
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.legend()
            plt.grid(True)
            st.pyplot(fig_tsne)

            # Dendrogram
            st.subheader("🧬 Dendrogram Hierarki")
            linked = linkage(X_scaled, method=best_result['link'])
            fig_dendro = plt.figure(figsize=(10, 6))
            dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False)
            plt.title(f'Dendrogram Linkage={best_result["link"].upper()}')
            plt.xlabel("Data")
            plt.ylabel("Jarak (distance)")
            st.pyplot(fig_dendro)

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {e}")

# =============== CLUSTERING KATEGORIK ===============
elif menu == "🧮 Clustering Kategorik":
    st.title("🧮 Clustering Data Kategorik")
    df = st.session_state.df

    if df is None:
        st.warning("⚠️ Silakan unggah dan preprocessing data terlebih dahulu.")
    else:
        try:
            st.subheader("🔢 Clustering dengan ROCK (jenis & ojol)")

            # 1. Fungsi bantu
            def jaccard_similarity_matrix(encoded):
                return 1 - pairwise_distances(encoded, metric="hamming")

            def get_neighbors(sim_matrix, theta):
                n = sim_matrix.shape[0]
                neighbors = [set(np.where(sim_matrix[i] >= theta)[0]) - {i} for i in range(n)]
                return neighbors

            def calculate_links(neighbors):
                n = len(neighbors)
                links = np.zeros((n, n), dtype=int)
                for i, j in combinations(range(n), 2):
                    common = neighbors[i].intersection(neighbors[j])
                    links[i, j] = links[j, i] = len(common)
                return links

            def rock_clustering(df_cat, theta, k_opt):
                encoded = pd.DataFrame()
                for col in df_cat.columns:
                    le = LabelEncoder()
                    encoded[col] = le.fit_transform(df_cat[col])
                sim_matrix = jaccard_similarity_matrix(encoded)
                neighbors = get_neighbors(sim_matrix, theta)
                links = calculate_links(neighbors)

                dist = 1 / (links + 1e-5)
                np.fill_diagonal(dist, 0)
                condensed_dist = squareform(dist, checks=False)
                linkage_matrix = linkage(condensed_dist, method='average')
                labels = fcluster(linkage_matrix, t=k_opt, criterion='maxclust')
                return labels, encoded

            def compute_cp_star(encoded, labels):
                sim_matrix = jaccard_similarity_matrix(encoded)
                N = len(labels)
                cp_total = 0
                unique_labels = np.unique(labels)

                for lbl in unique_labels:
                    indices = np.where(labels == lbl)[0]
                    n_k = len(indices)
                    if n_k <= 1:
                        continue
                    sim_sum = 0
                    for i, j in combinations(indices, 2):
                        sim_sum += sim_matrix[i, j]
                    sim_avg = sim_sum / (n_k * (n_k - 1) / 2)
                    cp_total += n_k * sim_avg

                cp_star = cp_total / N
                return cp_star

            # 2. Siapkan data kategorikal
            df_cat = df[['ojol', 'jenis']].copy()
            theta_list = [0.05, 0.1, 0.12, 0.15, 0.17, 0.2, 0.22, 0.25, 0.27, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            k_range = range(2, 5)

            best_cp = -np.inf
            best_labels = None
            best_theta = None
            best_k = None

            for theta in theta_list:
                for k_opt in k_range:
                    labels, encoded = rock_clustering(df_cat, theta, k_opt)
                    cp_star = compute_cp_star(encoded, labels)
                    if cp_star > best_cp:
                        best_cp = cp_star
                        best_labels = labels
                        best_theta = theta
                        best_k = k_opt

            # 3. Simpan hasil ke dataframe
            df['cluster_kategorik'] = best_labels
            st.session_state.df = df

            st.success(f"✅ Clustering selesai! Theta terbaik = {best_theta}, k = {best_k}, CP* = {best_cp:.4f}")
            st.subheader("📋 Preview Hasil Clustering")
            st.dataframe(df[['ojol', 'jenis', 'cluster_kategorik']])

            st.subheader("📈 Distribusi Cluster")
            fig, ax = plt.subplots()
            sns.countplot(x=df['cluster_kategorik'], palette='viridis', ax=ax)
            ax.set_title("Distribusi Hasil Clustering Kategorik (ROCK)")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Jumlah Data")
            st.pyplot(fig)

            # 4. Visualisasi t-SNE Hasil Clustering ROCK
            st.subheader("🔍 Visualisasi t-SNE Hasil Clustering ROCK")

            # Gunakan encoded dari hasil clustering terbaik
            sim_matrix = jaccard_similarity_matrix(encoded)
            dist_matrix = 1 - sim_matrix

            # Jalankan t-SNE
            tsne = TSNE(n_components=2, metric='precomputed', init='random', random_state=42)
            X_tsne = tsne.fit_transform(dist_matrix)

            # Visualisasi
            fig_tsne = plt.figure(figsize=(8, 6))
            for cl in np.unique(best_labels):
                idx = np.where(best_labels == cl)
                plt.scatter(
                    X_tsne[idx, 0],
                    X_tsne[idx, 1],
                    label=f'Cluster {cl}'
                )
            plt.title(f'Visualisasi ROCK Clustering\nTheta={best_theta}, k={best_k}, CP*={best_cp:.4f}')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.legend()
            plt.grid(True)
            st.pyplot(fig_tsne)

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat melakukan clustering ROCK: {e}")

# =============== CLUSTERING ENSEMBLE ===============
# =============== CLUSTERING ENSEMBLE ===============
elif menu == "🔗 Clustering Ensemble":
    st.title("🔗 Clustering Ensemble (ROCK)")

    df = st.session_state.df

    if df is None or 'cluster_numerik' not in df or 'cluster_kategorik' not in df:
        st.warning("⚠️ Pastikan data sudah diproses melalui Clustering Numerik dan Kategorik terlebih dahulu.")
    else:
        try:
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.manifold import TSNE

            st.subheader("⚙️ Proses Ensemble Clustering")

            # ===============================
            # FUNGSI ROCK Clustering
            # ===============================

            def jaccard_similarity_matrix(encoded):
                return 1 - pairwise_distances(encoded, metric="hamming")

            def get_neighbors(sim_matrix, theta):
                n = sim_matrix.shape[0]
                neighbors = [set(np.where(sim_matrix[i] >= theta)[0]) - {i} for i in range(n)]
                return neighbors

            def calculate_links(neighbors):
                n = len(neighbors)
                links = np.zeros((n, n), dtype=int)
                for i, j in combinations(range(n), 2):
                    common = neighbors[i].intersection(neighbors[j])
                    links[i, j] = links[j, i] = len(common)
                return links

            def rock_clustering(df_cat, theta, k_opt):
                encoded = pd.DataFrame()
                for col in df_cat.columns:
                    le = LabelEncoder()
                    encoded[col] = le.fit_transform(df_cat[col])

                sim_matrix = jaccard_similarity_matrix(encoded)
                neighbors = get_neighbors(sim_matrix, theta)
                links = calculate_links(neighbors)

                dist = 1 / (links + 1e-5)
                np.fill_diagonal(dist, 0)
                condensed_dist = squareform(dist, checks=False)

                linkage_matrix = linkage(condensed_dist, method='average')
                labels = fcluster(linkage_matrix, t=k_opt, criterion='maxclust')
                return labels

            def compute_cp_star(encoded, labels):
                sim_matrix = 1 - pairwise_distances(encoded, metric="hamming")
                N = len(labels)
                cp_total = 0
                unique_labels = np.unique(labels)

                for lbl in unique_labels:
                    indices = np.where(labels == lbl)[0]
                    n_k = len(indices)
                    if n_k <= 1:
                        continue
                    sim_sum = sum(sim_matrix[i, j] for i, j in combinations(indices, 2))
                    sim_avg = sim_sum / (n_k * (n_k - 1) / 2)
                    cp_total += n_k * sim_avg

                return cp_total / N

            # ===============================
            # PROSES CLUSTERING ENSEMBLE
            # ===============================
            df_ensemble_input = df[['cluster_numerik', 'cluster_kategorik']].astype(str)
            encoder = OneHotEncoder(sparse_output=False)
            encoded_ensemble = encoder.fit_transform(df_ensemble_input)

            theta_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            best_cp = -np.inf
            best_labels = None
            best_theta = None
            best_k = None

            progress = st.progress(0)
            total_loop = len(theta_list) * 3
            loop_count = 0

            for theta in theta_list:
                for k_opt in range(2, 5):
                    labels_ensemble = rock_clustering(df_ensemble_input, theta, k_opt=k_opt)
                    cp_star = compute_cp_star(pd.DataFrame(encoded_ensemble), labels_ensemble)

                    if cp_star > best_cp:
                        best_cp = cp_star
                        best_labels = labels_ensemble
                        best_theta = theta
                        best_k = k_opt

                    loop_count += 1
                    progress.progress(loop_count / total_loop)

            df['cluster_ensemble_rock'] = best_labels
            st.session_state.df = df

            st.success(f"✅ Clustering ensemble selesai! Theta terbaik = {best_theta}, k = {best_k}, CP* = {best_cp:.4f}")
            st.subheader("📋 Hasil Clustering Ensemble")
            st.dataframe(df[['cluster_numerik', 'cluster_kategorik', 'cluster_ensemble_rock']])

            st.subheader("📈 Distribusi Cluster Ensemble")
            fig, ax = plt.subplots()
            sns.countplot(x=df['cluster_ensemble_rock'], palette='magma', ax=ax)
            ax.set_title("Distribusi Cluster Ensemble (ROCK)")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Jumlah Data")
            st.pyplot(fig)

            # ===============================
            # t-SNE Visualisasi Hasil Ensemble
            # ===============================
            st.subheader("🔍 Visualisasi t-SNE Hasil Ensemble")

            sim_matrix = jaccard_similarity_matrix(pd.DataFrame(encoded_ensemble))
            dist_matrix = 1 - sim_matrix

            tsne = TSNE(n_components=2, metric='precomputed', init='random', random_state=42)
            X_tsne = tsne.fit_transform(dist_matrix)

            fig_tsne = plt.figure(figsize=(8, 6))
            for cl in np.unique(best_labels):
                plt.scatter(
                    X_tsne[np.array(best_labels) == cl, 0],
                    X_tsne[np.array(best_labels) == cl, 1],
                    label=f'Cluster {cl}'
                )
            plt.title(f'Visualisasi ROCK Clustering Ensemble\nTheta={best_theta}, k={best_k}, CP*={best_cp:.4f}')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.legend()
            plt.grid(True)
            st.pyplot(fig_tsne)

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat ensemble clustering: {e}")
