import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io

from scipy.stats import zscore
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder  # ⬅ Tambahkan LabelEncoder di sini
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from itertools import combinations
from sklearn.manifold import TSNE  # ⬅ Diperlukan untuk t-SNE di ROCK

# Konfigurasi halaman
st.set_page_config(page_title="Clustering UMKM", layout="wide")

# Sidebar Navigasi
st.sidebar.title("Menu Navigasi")
menu = st.sidebar.radio("Pilih halaman:", [
    "🏠 Home",
    "📂 Upload Data",
    "📈 Analisis Data",
    "⚙ Data Preprocessing",
    "📊 Clustering Numerik",
    "🧮 Clustering Kategorik",
    "🤝 Clustering Ensemble",
    "📏 Evaluasi Clustering Ensemble",
    "🧾 Interpretasi Hasil",
    "💾 Unduh Hasil Clustering Ensemble"
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
        *Format CSV wajib memuat kolom:*
        - modal, omset, tenaga_kerja: angka bulat
        - ojol: "ya" / "tidak"
        - jenis: "mamin" / "oleh"
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

# =============== ANALISIS DATA ===============
elif menu == "📈 Analisis Data":
    st.title("📈 Analisis Data UMKM")
    df = st.session_state.df
    if df is None:
        st.warning("⚠ Silakan unggah data terlebih dahulu.")
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
        except Exception as e:
            st.error(f"Terjadi kesalahan saat analisis data: {e}")

# =============== PREPROCESSING ===============
elif menu == "⚙ Data Preprocessing":
    st.title("⚙ Tahap Preprocessing Data")
    df = st.session_state.df
    if df is None:
        st.warning("⚠ Silakan unggah data terlebih dahulu.")
    else:
        try:
            cols_num = ['omset', 'tenaga kerja', 'modal']

            st.subheader("1. Missing Values")
            st.dataframe(df.isnull().sum())

            st.subheader("2. Boxplot Sebelum Outlier Handling")
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

            st.subheader("3. Boxplot Setelah Outlier Handling")
            fig4, ax4 = plt.subplots()
            sns.boxplot(data=df[cols_num], ax=ax4)
            st.pyplot(fig4)

            st.subheader("4. Normalisasi Data Z-Score")
            df_zscore = df.copy()
            df_zscore[cols_num] = df_zscore[cols_num].apply(zscore)
            st.session_state.df_zscore = df_zscore
            st.dataframe(df_zscore[cols_num].head())
        except Exception as e:
            st.error(f"Terjadi kesalahan saat preprocessing: {e}")

# =============== CLUSTERING NUMERIK ===============
elif menu == "📊 Clustering Numerik":
    st.title("📊 Clustering Data Numerik (AHC)")

    df_zscore = st.session_state.df_zscore
    df = st.session_state.df

    if df_zscore is None:
        st.warning("⚠ Data belum tersedia. Lakukan preprocessing terlebih dahulu.")
    else:
        try:
            X_scaled = df_zscore[['omset', 'tenaga kerja', 'modal']].values
            n = len(X_scaled)
            global_mean = np.mean(X_scaled, axis=0)

            linkage_types = ['single', 'complete', 'average']
            best_result = {'k': None, 'link': None, 'PseudoF': -np.inf, 'ICD': np.inf}
            results = []

            st.subheader("📋 Validasi Clustering Berdasarkan Pseudo-F dan ICD Rate")
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
            - Jumlah klaster optimum: *{best_result['k']}*
            - Metode linkage terbaik: *{best_result['link'].capitalize()}*
            - Nilai Pseudo-F tertinggi: *{best_result['PseudoF']:.4f}*
            - Nilai ICD terkecil: *{best_result['ICD']:.4f}*
            """)

            # 1. Clustering dengan model terbaik
            best_model = AgglomerativeClustering(n_clusters=best_result['k'], linkage=best_result['link'])
            best_labels = best_model.fit_predict(X_scaled)

            # Simpan hasil ke dataframe
            df['cluster_numerik'] = best_labels
            st.session_state.df = df

            # 2. Reduksi dimensi dengan t-SNE
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42)
            X_reduced = tsne.fit_transform(X_scaled)

            # 3. Visualisasi t-SNE
            st.subheader("🔸 Visualisasi t-SNE")
            fig_tsne, ax_tsne = plt.subplots(figsize=(8, 6))
            for cl in np.unique(best_labels):
                ax_tsne.scatter(
                    X_reduced[best_labels == cl, 0],
                    X_reduced[best_labels == cl, 1],
                    label=f'Cluster {cl+1}'
                )
            ax_tsne.set_title(f'Visualisasi Clustering dengan t-SNE\nLinkage={best_result["link"].upper()}, k={best_result["k"]}')
            ax_tsne.set_xlabel("t-SNE 1")
            ax_tsne.set_ylabel("t-SNE 2")
            ax_tsne.legend()
            ax_tsne.grid(True)
            st.pyplot(fig_tsne)

            # 4. Visualisasi dendrogram
            st.subheader("🧬 Dendrogram Hierarki")
            from scipy.cluster.hierarchy import linkage, dendrogram
            linked = linkage(X_scaled, method=best_result['link'])
            fig_dendro, ax_dendro = plt.subplots(figsize=(10, 6))
            dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False, ax=ax_dendro)
            ax_dendro.set_title(f'Dendrogram Linkage={best_result["link"].upper()}')
            ax_dendro.set_xlabel("Data")
            ax_dendro.set_ylabel("Jarak (distance)")
            st.pyplot(fig_dendro)

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {e}")

# =============== CLUSTERING KATEGORIK ===============
elif menu == "🧮 Clustering Kategorik":
    st.title("🧮 Clustering Data Kategorik")
    df = st.session_state.df

    if df is None:
        st.warning("⚠ Silakan unggah dan preprocessing data terlebih dahulu.")
    else:
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        from itertools import combinations
        import matplotlib.pyplot as plt
        import seaborn as sns

        # ===========================
        # 1. Jaccard Similarity Manual
        # ===========================
        def jaccard_similarity_matrix(encoded):
            n = encoded.shape[0]
            sim_matrix = np.zeros((n, n))
            data_as_sets = [
                set((col, val) for col, val in enumerate(row))
                for row in encoded.values
            ]
            for i in range(n):
                for j in range(i, n):
                    inter = data_as_sets[i].intersection(data_as_sets[j])
                    union = data_as_sets[i].union(data_as_sets[j])
                    sim = len(inter) / len(union) if union else 1
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim
            return sim_matrix

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

        def calculate_goodness(links, cluster_members, theta):
            f_theta = (1 - theta) / (1 + theta)
            goodness = np.zeros_like(links, dtype=float)
            n = len(cluster_members)

            for i, j in combinations(range(n), 2):
                link_ij = links[i, j]
                if link_ij == 0:
                    continue

                ni = len(cluster_members[i])
                nj = len(cluster_members[j])
                denominator = (ni + nj) ** (1 + 2 * f_theta) - ni ** (1 + 2 * f_theta) - nj ** (1 + 2 * f_theta)

                g = link_ij / denominator if denominator != 0 else 0
                goodness[i, j] = goodness[j, i] = g

            return goodness

        def rock_clustering(data, theta, target_cluster_count):
            encoded = data.apply(LabelEncoder().fit_transform)
            sim_matrix = jaccard_similarity_matrix(encoded)
            neighbors = get_neighbors(sim_matrix, theta)
            n = len(data)
            clusters = [{i} for i in range(n)]

            while len(clusters) > target_cluster_count:
                cluster_indices = list(range(len(clusters)))
                cluster_members = clusters
                cluster_neighbors = [set().union(*(neighbors[i] for i in cluster)) for cluster in clusters]

                links = np.zeros((len(clusters), len(clusters)), dtype=int)
                for i, j in combinations(cluster_indices, 2):
                    common_neighbors = cluster_neighbors[i].intersection(cluster_neighbors[j])
                    links[i, j] = links[j, i] = len(common_neighbors)

                goodness = calculate_goodness(links, [clusters[i] for i in cluster_indices], theta)
                i_max, j_max = np.unravel_index(np.argmax(goodness), goodness.shape)
                max_goodness = goodness[i_max, j_max]
                if max_goodness == 0:
                    break
                clusters[i_max] = clusters[i_max].union(clusters[j_max])
                del clusters[j_max]

            labels = np.zeros(n, dtype=int)
            for cluster_id, members in enumerate(clusters):
                for member in members:
                    labels[member] = cluster_id

            return labels, encoded

        def calculate_total_cp(labels, sim_matrix):
            df_labels = pd.DataFrame({'label': labels})
            total_sim = 0
            total_pairs = 0
            for cluster_id in np.unique(labels):
                members = df_labels[df_labels['label'] == cluster_id].index.tolist()
                n_k = len(members)
                if n_k <= 1:
                    continue
                for i, j in combinations(members, 2):
                    total_sim += sim_matrix[i, j]
                total_pairs += n_k * (n_k - 1) / 2
            return total_sim / total_pairs if total_pairs != 0 else 0

        # Evaluasi Semua Kombinasi Theta dan k
        data = df[['ojol', 'jenis']].copy()
        theta_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        cluster_range = [2, 3, 4, 5, 6, 7]
        total_cp_results = []

        for theta in theta_values:
            for k in cluster_range:
                try:
                    labels, encoded = rock_clustering(data, theta, k)
                    sim_matrix = jaccard_similarity_matrix(encoded)
                    total_cp = calculate_total_cp(labels, sim_matrix)
                    total_cp_results.append({
                        'theta': theta,
                        'k': k,
                        'cp_star_total': total_cp
                    })
                except Exception as e:
                    st.write(f"❌ Theta={theta:.2f}, k={k}: {e}")

        cp_summary = pd.DataFrame(total_cp_results)
        st.subheader("📊 Rangkuman Evaluasi CP* Total")
        st.dataframe(cp_summary.sort_values(by=['theta', 'k']))

        # Clustering dengan konfigurasi terbaik
        theta_final = 0.40
        k_final = 4
        labels, encoded = rock_clustering(data, theta_final, k_final)
        df['cluster_kategorik'] = labels
        st.session_state.df = df

        st.success(f"✅ Clustering ROCK selesai! Theta = {theta_final}, k = {k_final}")

                # Visualisasi t-SNE
        st.subheader("🌀 Visualisasi t-SNE Clustering Kategorik")

        labels_best = rock_clustering(data, theta=0.40, target_cluster_count=4)[0]
        df['cluster_kategorik'] = labels_best

        # Encode ulang untuk t-SNE
        encoded = data.apply(LabelEncoder().fit_transform)
        sim_matrix = jaccard_similarity_matrix(encoded)
        dist_matrix = 1 - sim_matrix

        # t-SNE reduction
        tsne = TSNE(n_components=2, metric='precomputed', init='random', random_state=42)
        X_tsne = tsne.fit_transform(dist_matrix)

        # Plot t-SNE
        fig_tsne, ax_tsne = plt.subplots(figsize=(8, 6))
        for cl in np.unique(labels_best):
            idx = np.array(labels_best) == cl
            ax_tsne.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=f'Cluster {cl}')
        ax_tsne.set_title("Visualisasi ROCK Clustering\nTheta = 0.40, k = 4")
        ax_tsne.set_xlabel("t-SNE 1")
        ax_tsne.set_ylabel("t-SNE 2")
        ax_tsne.legend()
        ax_tsne.grid(True)
        st.pyplot(fig_tsne)

        st.subheader("📋 Hasil Clustering")
        st.dataframe(df[['ojol', 'jenis', 'cluster_kategorik']])

        cluster_counts = df['cluster_kategorik'].value_counts().sort_index()
        st.subheader("📈 Distribusi Cluster")
        fig, ax = plt.subplots()
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax, palette='viridis')
        ax.set_title("Distribusi Jumlah Data per Cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Jumlah Data")
        st.pyplot(fig)

# =============== CLUSTERING ENSEMBLE ===============
elif menu == "🤝 Clustering Ensemble":
    st.title("🤝 Clustering Ensemble ROCK")
    df = st.session_state.df

    if df is None or 'cluster_numerik' not in df.columns or 'cluster_kategorik' not in df.columns:
        st.warning("⚠ Pastikan data sudah diproses dan memiliki kolom 'cluster_numerik' dan 'cluster_kategorik'.")
    else:
        st.write("✅ Mulai proses Clustering Ensemble...")

        # Gunakan data cluster sebagai kategorikal (harus string untuk LabelEncoder)
        df_ensemble = df[['cluster_numerik', 'cluster_kategorik']].astype(str).copy()

        # --- FUNGSI UTILITAS ---
        def jaccard_similarity_matrix(encoded):
            n = encoded.shape[0]
            sim_matrix = np.zeros((n, n))
            data_as_sets = [
                set((col, val) for col, val in enumerate(row))
                for row in encoded.values
            ]
            for i in range(n):
                for j in range(i, n):
                    inter = data_as_sets[i].intersection(data_as_sets[j])
                    union = data_as_sets[i].union(data_as_sets[j])
                    sim = len(inter) / len(union) if union else 1
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim
            return sim_matrix

        def get_neighbors(sim_matrix, theta):
            n = sim_matrix.shape[0]
            return [set(np.where(sim_matrix[i] >= theta)[0]) - {i} for i in range(n)]

        def calculate_goodness(links, cluster_members, theta):
            f_theta = (1 - theta) / (1 + theta)
            goodness = np.zeros_like(links, dtype=float)
            n = len(cluster_members)
            for i, j in combinations(range(n), 2):
                link_ij = links[i, j]
                if link_ij == 0:
                    continue
                ni = len(cluster_members[i])
                nj = len(cluster_members[j])
                denominator = (ni + nj) ** (1 + 2 * f_theta) - ni ** (1 + 2 * f_theta) - nj ** (1 + 2 * f_theta)
                g = link_ij / denominator if denominator != 0 else 0
                goodness[i, j] = goodness[j, i] = g
            return goodness

        def rock_clustering(data, theta, target_cluster_count):
            encoded = data.apply(LabelEncoder().fit_transform)
            sim_matrix = jaccard_similarity_matrix(encoded)
            neighbors = get_neighbors(sim_matrix, theta)
            n = len(data)
            clusters = [{i} for i in range(n)]
            while len(clusters) > target_cluster_count:
                cluster_indices = list(range(len(clusters)))
                cluster_members = clusters
                cluster_neighbors = [set().union(*(neighbors[i] for i in cluster)) for cluster in clusters]
                links = np.zeros((len(clusters), len(clusters)), dtype=int)
                for i, j in combinations(cluster_indices, 2):
                    common_neighbors = cluster_neighbors[i].intersection(cluster_neighbors[j])
                    links[i, j] = links[j, i] = len(common_neighbors)
                goodness = calculate_goodness(links, cluster_members, theta)
                i_max, j_max = np.unravel_index(np.argmax(goodness), goodness.shape)
                max_goodness = goodness[i_max, j_max]
                if max_goodness == 0:
                    break
                clusters[i_max] = clusters[i_max].union(clusters[j_max])
                del clusters[j_max]
            labels = np.zeros(n, dtype=int)
            for cluster_id, members in enumerate(clusters):
                for member in members:
                    labels[member] = cluster_id
            return labels, encoded

        def calculate_total_cp(labels, sim_matrix):
            df_labels = pd.DataFrame({'label': labels})
            total_sim = 0
            total_pairs = 0
            for cluster_id in np.unique(labels):
                members = df_labels[df_labels['label'] == cluster_id].index.tolist()
                n_k = len(members)
                if n_k <= 1:
                    continue
                for i, j in combinations(members, 2):
                    total_sim += sim_matrix[i, j]
                total_pairs += n_k * (n_k - 1) / 2
            return total_sim / total_pairs if total_pairs != 0 else 0

        # --- EVALUASI BEBERAPA KOMBINASI THETA DAN K ---
        theta_values = [0.3, 0.4, 0.5]  # Bisa diperluas lagi jika sudah stabil
        cluster_range = [3, 4, 5, 6, 7]
        ensemble_cp_results = []

        with st.spinner("🔍 Mengevaluasi kombinasi theta dan jumlah cluster..."):
            for theta in theta_values:
                for k in cluster_range:
                    try:
                        labels_try, encoded_try = rock_clustering(df_ensemble, theta, k)
                        sim_matrix_try = jaccard_similarity_matrix(encoded_try)
                        total_cp = calculate_total_cp(labels_try, sim_matrix_try)
                        ensemble_cp_results.append({
                            'theta': theta,
                            'k': k,
                            'cp_star_total': total_cp
                        })
                    except Exception as e:
                        st.write(f"❌ Theta={theta:.2f}, k={k}: {e}")

        cp_ensemble_summary = pd.DataFrame(ensemble_cp_results)
        st.subheader("📊 Rangkuman Evaluasi CP* Total (Clustering Ensemble)")
        st.dataframe(cp_ensemble_summary.sort_values(by=['theta', 'k']))

        # --- CLUSTERING TERBAIK SECARA MANUAL / OTOMATIS ---
        theta_final = 0.40
        k_final = 7

        st.markdown(f"### 🚀 Final Clustering: Theta = *{theta_final}, K = **{k_final}*")
        with st.spinner("🔄 Menjalankan final clustering ensemble..."):
            labels_final, encoded_final = rock_clustering(df_ensemble, theta_final, k_final)
            df['cluster_ensemble_rock'] = labels_final
            st.session_state.df = df

        st.success("✅ Clustering Ensemble selesai!")

        st.subheader("📋 Hasil Clustering Ensemble")
        st.dataframe(df[['cluster_numerik', 'cluster_kategorik', 'cluster_ensemble_rock']])

        # --- PLOT DISTRIBUSI CLUSTER ---
        cluster_counts = df['cluster_ensemble_rock'].value_counts().sort_index()
        st.subheader("📈 Distribusi Jumlah Data per Cluster")
        fig_bar, ax_bar = plt.subplots()
        ax_bar.bar(cluster_counts.index.astype(str), cluster_counts.values, color='mediumseagreen')
        ax_bar.set_xlabel("Cluster")
        ax_bar.set_ylabel("Jumlah Data")
        ax_bar.set_title("Distribusi Cluster (Ensemble)")
        st.pyplot(fig_bar)

        # --- VISUALISASI T-SNE ---
        st.subheader("🌀 Visualisasi t-SNE Clustering Ensemble")
        with st.spinner("⏳ Memproyeksikan data ke 2D dengan t-SNE..."):
            sim_matrix = jaccard_similarity_matrix(encoded_final)
            dist_matrix = 1 - sim_matrix
            tsne = TSNE(n_components=2, metric='precomputed', init='random', random_state=42)
            X_tsne = tsne.fit_transform(dist_matrix)

        fig_tsne, ax_tsne = plt.subplots(figsize=(8, 6))
        for cl in np.unique(labels_final):
            idx = np.array(labels_final) == cl
            ax_tsne.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=f'Cluster {cl}', s=60)
        ax_tsne.set_title(f"t-SNE Clustering Ensemble\nTheta = {theta_final}, k = {k_final}")
        ax_tsne.set_xlabel("t-SNE Komponen 1")
        ax_tsne.set_ylabel("t-SNE Komponen 2")
        ax_tsne.legend()
        ax_tsne.grid(True)
        st.pyplot(fig_tsne)

# =============== EVALUASI CLUSTERING ENSEMBLE ===============
elif menu == "📏 Evaluasi Clustering Ensemble":
    st.title("📏 Evaluasi Clustering Ensemble ROCK")
    df = st.session_state.df

    if df is None or 'cluster_ensemble_rock' not in df.columns:
        st.warning("⚠ Data hasil clustering ensemble belum tersedia.")
    else:
        st.markdown("### 📌 Evaluasi Menggunakan Silhouette Score dengan Jaccard Manual")

        # Gunakan data clustering ensemble
        df_eval = df[['cluster_numerik', 'cluster_kategorik']].astype(str)
        labels_final = df['cluster_ensemble_rock'].values
        theta_final = 0.40
        k_final = len(np.unique(labels_final))

        # Fungsi-fungsi pendukung
        def jaccard_similarity_matrix(encoded):
            n = encoded.shape[0]
            sim_matrix = np.zeros((n, n))
            data_as_sets = [
                set((col, val) for col, val in enumerate(row))
                for row in encoded.values
            ]
            for i in range(n):
                for j in range(i, n):
                    inter = data_as_sets[i].intersection(data_as_sets[j])
                    union = data_as_sets[i].union(data_as_sets[j])
                    sim = len(inter) / len(union) if union else 1
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim
            return sim_matrix

        def silhouette_score_jaccard(encoded, labels):
            from sklearn.metrics import silhouette_score
            sim_matrix = jaccard_similarity_matrix(encoded)
            dist_matrix = 1 - sim_matrix
            score = silhouette_score(dist_matrix, labels, metric='precomputed')
            return score

        # Jalankan evaluasi
        with st.spinner("🔍 Menghitung Silhouette Score..."):
            encoded_final = df_eval.apply(LabelEncoder().fit_transform)
            score = silhouette_score_jaccard(encoded_final, labels_final)

        st.success("✅ Evaluasi selesai!")
        st.markdown(f"""
        - *Theta (θ)* = {theta_final}
        - *Jumlah Cluster (k)* = {k_final}
        - *Silhouette Score* = {score:.4f}
        """)

        # Visualisasi tambahan (opsional)
        st.subheader("📈 Interpretasi Skor Silhouette")
        if score > 0.7:
            st.info("🔹 Skor Silhouette sangat baik — cluster sangat terpisah.")
        elif score > 0.5:
            st.info("🔸 Skor Silhouette cukup baik — cluster cukup terpisah.")
        elif score > 0.25:
            st.warning("⚠ Skor Silhouette sedang — mungkin ada tumpang tindih antar cluster.")
        else:
            st.error("🔻 Skor Silhouette rendah — hasil cluster kurang optimal.")

# =============== INTERPRETASI CLUSTERING ENSEMBLE ===============
elif menu == "🧾 Interpretasi Hasil":
    st.title("🧾 Interpretasi Hasil Clustering Ensemble")

    df = st.session_state.df
    if df is None or 'cluster_ensemble_rock' not in df:
        st.warning("⚠ Pastikan hasil clustering ensemble sudah tersedia.")
    else:
        try:
            st.subheader("📊 Tabel Ringkasan per Cluster")

            # 1. Rata-rata fitur numerik per cluster
            mean_stats = df.groupby('cluster_ensemble_rock')[['omset', 'tenaga kerja', 'modal']].mean().round(2)

            # 2. Dominasi 'jenis' per cluster
            jenis_dom = df.groupby('cluster_ensemble_rock')['jenis'].agg(lambda x: x.value_counts().idxmax()).rename('jenis_terbanyak')

            # 3. Dominasi 'ojol' per cluster
            ojol_dom = df.groupby('cluster_ensemble_rock')['ojol'].agg(lambda x: x.value_counts().idxmax()).rename('ojol_terbanyak')

            # 4. Gabungkan semua jadi satu dataframe
            summary = pd.concat([mean_stats, jenis_dom, ojol_dom], axis=1).reset_index()
            summary.columns = ['Cluster', 'Omset (rata2)', 'Tenaga Kerja (rata2)', 'Modal (rata2)', 'Jenis Dominan', 'Ojol Dominan']

            # 5. Tampilkan tabel
            st.dataframe(summary)

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat menggabungkan interpretasi: {e}")

# =============== UNDUH ===============
elif menu == "💾 Unduh Hasil Clustering Ensemble":
    st.title("💾 Unduh Hasil Clustering Ensemble")

    df = st.session_state.df

    if df is None or 'cluster_ensemble_rock' not in df:
        st.warning("⚠ Data belum tersedia atau clustering ensemble belum dilakukan.")
    else:
        st.markdown("### 🔽 Tabel Hasil Clustering Ensemble")
        st.dataframe(df)

        # Simpan ke CSV dalam memori
        csv = df.to_csv(index=False).encode('utf-8')

        # Tombol Unduh
        st.download_button(
            label="💾 Unduh sebagai CSV",
            data=csv,
            file_name="hasil_clustering_ensemble.csv",
            mime='text/csv'
        )
