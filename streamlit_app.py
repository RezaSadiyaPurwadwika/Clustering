import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
import io
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage

st.set_page_config(page_title="Clustering App", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "home"

query_params = st.query_params
if "page" in query_params:
    st.session_state.page = query_params["page"]

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

page = st.session_state.page
st.markdown(f"""
<div class="navbar">
    <a href="/?page=home" class="nav-item {'active' if page == 'home' else ''}">🏠 Home</a>
    <a href="/?page=about" class="nav-item {'active' if page == 'about' else ''}">📋 About</a>
    <a href="/?page=rules" class="nav-item {'active' if page == 'rules' else ''}">📜 Rules</a>
</div>
""", unsafe_allow_html=True)

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
                if 'jenis' in df.columns:
                    df['jenis'] = df['jenis'].astype(str).str.strip().str.lower()
                if 'ojol' in df.columns:
                    df['ojol'] = df['ojol'].astype(str).str.strip().str.lower()

                if 'jenis' in df.columns:
                    st.subheader("✅ Distribusi Kategori 'jenis'")
                    fig1, ax1 = plt.subplots(figsize=(4, 3))
                    sns.countplot(data=df, x='jenis', ax=ax1)
                    ax1.set_title("Distribusi Kategori: jenis")
                    st.pyplot(fig1)

                if 'ojol' in df.columns:
                    st.subheader("✅ Distribusi Kategori 'ojol'")
                    fig2, ax2 = plt.subplots(figsize=(4, 3))
                    sns.countplot(data=df, x='ojol', ax=ax2)
                    ax2.set_title("Distribusi Kategori: ojol")
                    st.pyplot(fig2)

                st.subheader("ℹ️ Info Dataset")
                buffer = io.StringIO()
                df.info(buf=buffer)
                st.text(buffer.getvalue())

                st.subheader("📊 Statistik Deskriptif (Numerik)")
                num_cols = [col for col in ['omset', 'tenaga_kerja', 'modal'] if col in df.columns]
                if not num_cols:
                    st.error("❌ Kolom numerik tidak ditemukan: 'omset', 'tenaga_kerja', atau 'modal'")
                else:
                    st.dataframe(df[num_cols].describe())

                    st.subheader("🔍 Missing Values")
                    st.dataframe(df[num_cols].isnull().sum())

                    st.subheader("📦 Boxplot Sebelum Penanganan Outlier")
                    fig3, ax3 = plt.subplots(figsize=(5, 3))
                    sns.boxplot(data=df[num_cols], ax=ax3)
                    ax3.set_title('Boxplot Sebelum Outlier Handling')
                    st.pyplot(fig3)

                    for col in ['omset', 'modal']:
                        if col in df.columns:
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower = Q1 - 1.5 * IQR
                            upper = Q3 + 1.5 * IQR
                            df[col] = df[col].clip(lower=lower, upper=upper)

                    st.subheader("📦 Boxplot Setelah Penanganan Outlier")
                    fig4, ax4 = plt.subplots(figsize=(5, 3))
                    sns.boxplot(data=df[num_cols], ax=ax4)
                    ax4.set_title('Boxplot Setelah Outlier Handling')
                    st.pyplot(fig4)

                    st.subheader("📈 Data Setelah Normalisasi Z-Score")
                    df_zscore = df.copy()
                    df_zscore[num_cols] = df_zscore[num_cols].apply(zscore)
                    st.dataframe(df_zscore[num_cols].head())

                    clustering_num = st.button("🔗 Clustering Numerik (Agglomerative)")
                    if clustering_num:
                        try:
                            X = df_zscore[['omset', 'tenaga_kerja', 'modal']]
                            X_scaled = StandardScaler().fit_transform(X)

                            n = len(X_scaled)
                            global_mean = np.mean(X_scaled, axis=0)

                            linkage_types = ['single', 'complete', 'average']
                            best_result = {'k': None, 'link': None, 'PseudoF': -np.inf, 'ICD': np.inf}

                            log_output = ""

                            for link in linkage_types:
                                for k in range(2, 7):
                                    model = AgglomerativeClustering(n_clusters=k, linkage=link)
                                    labels = model.fit_predict(X_scaled)

                                    SW, SB = 0, 0
                                    for cl in np.unique(labels):
                                        cluster_data = X_scaled[labels == cl]
                                        mean_cl = np.mean(cluster_data, axis=0)
                                        SW += np.sum((cluster_data - mean_cl) ** 2)
                                        SB += len(cluster_data) * np.sum((mean_cl - global_mean) ** 2)

                                    pseudoF = (SB / (k - 1)) / (SW / (n - k)) if SW != 0 else np.inf
                                    ICD = SW / n

                                    log_output += f"Linkage: {link}, k={k} → Pseudo-F: {pseudoF:.4f}, ICD: {ICD:.4f}\n"

                                    if pseudoF > best_result['PseudoF']:
                                        best_result = {'k': k, 'link': link, 'PseudoF': pseudoF, 'ICD': ICD}

                            st.code(log_output, language='text')

                            st.success(f"✅ Linkage terbaik: {best_result['link'].upper()}, Cluster: {best_result['k']}")
                            best_model = AgglomerativeClustering(n_clusters=best_result['k'], linkage=best_result['link'])
                            best_labels = best_model.fit_predict(X_scaled)
                            df['cluster_numerik'] = best_labels

                            st.subheader("📊 Visualisasi Clustering dengan t-SNE")
                            tsne = TSNE(n_components=2, random_state=42)
                            X_reduced = tsne.fit_transform(X_scaled)

                            fig_tsne, ax_tsne = plt.subplots(figsize=(6, 4))
                            for cl in np.unique(best_labels):
                                ax_tsne.scatter(
                                    X_reduced[best_labels == cl, 0],
                                    X_reduced[best_labels == cl, 1],
                                    label=f'Cluster {cl + 1}'
                                )
                            ax_tsne.set_title(f'Clustering t-SNE (Linkage={best_result["link"].upper()}, k={best_result["k"]})')
                            ax_tsne.set_xlabel('t-SNE 1')
                            ax_tsne.set_ylabel('t-SNE 2')
                            ax_tsne.legend()
                            ax_tsne.grid(True)
                            st.pyplot(fig_tsne)

                            st.subheader("🌳 Dendrogram Hirarki")
                            fig_dendro, ax_dendro = plt.subplots(figsize=(8, 4))
                            linked = linkage(X_scaled, method=best_result['link'])
                            dendrogram(linked, ax=ax_dendro, orientation='top', distance_sort='descending', show_leaf_counts=False)
                            ax_dendro.set_title(f'Dendrogram Linkage={best_result["link"].upper()}')
                            ax_dendro.set_xlabel('Data')
                            ax_dendro.set_ylabel('Jarak')
                            st.pyplot(fig_dendro)

                            st.subheader("🧾 Data dengan Label Cluster")
                            st.dataframe(df[['omset', 'tenaga_kerja', 'modal', 'cluster_numerik']])

                        except Exception as e:
                            st.error(f"🚨 Gagal menjalankan clustering: {e}")

            except Exception as e:
                st.error(f"🚨 Terjadi error saat preprocessing: {e}")

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
