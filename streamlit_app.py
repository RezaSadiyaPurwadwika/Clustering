import streamlit as st

# Simulasikan navigasi dengan query_params
query_params = st.query_params
page = query_params.get("page", ["home"])[0]

# Header / Navbar
st.markdown("""
    <style>
        .nav-container {
            background-color: #34496d;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #ccc;
        }
        .nav-title {
            font-weight: bold;
            font-size: 22px;
            color: white;
        }
        .nav-links a {
            margin-left: 25px;
            text-decoration: none;
            font-weight: bold;
            color: white;
        }
        .nav-links a:hover {
            text-decoration: underline;
            color: #aad3f7;
        }
    </style>
    <div class="nav-container">
        <div class="nav-title">🌐 Ensemble Rock Clustering</div>
        <div class="nav-links">
            <a href="/?page=home">Home</a>
            <a href="/?page=about">About</a>
            <a href="/?page=rules">Rules</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# === Tampilan Berdasarkan Menu Dipilih ===
if page == "home":
    st.title("Better Solutions for Your Clustering")
    st.write("Upload your data and get started with powerful clustering tools.")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

elif page == "about":
    st.header("About")
    st.write("Aplikasi ini dibuat untuk mempermudah proses clustering data UMKM dengan pendekatan Ensemble ROCK...")

elif page == "rules":
    st.header("Rules")
    st.write("""
    **Beberapa aturan penggunaan aplikasi ini:**
    1. File yang diunggah harus format `.csv`.
    2. Maksimal ukuran file 200MB.
    3. Pastikan kolom-kolom telah sesuai format yang diharapkan.
    """)
