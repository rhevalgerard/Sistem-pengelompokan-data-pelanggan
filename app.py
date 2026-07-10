import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="Segmentasi Pelanggan", layout="wide")

CLUSTER_COLORS = ["#2D3A8C", "#1E9E8B", "#E8A33D", "#D6486B", "#8B5CF6", "#4C9A6A"]

# Styling - palet warna & tipografi konsisten, tanpa emoticon
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    .stApp { background-color: #F4F5F7; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    h1, h2, h3 { font-family: 'Times New Roman', sans-serif !important; letter-spacing: -0.2px; }

    .app-header {
        background: #14171F;
        padding: 32px 36px;
        border-radius: 14px;
        margin-bottom: 28px;
    }
    .app-header h1 {
        color: #ffffff;
        font-size: 28px;
        margin: 0 0 8px 0;
    }
    .app-header p {
        color: #B7BCC9;
        font-size: 14.5px;
        line-height: 1.6;
        margin: 0;
        max-width: 640px;
    }

    .step-label {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #EEF0FA;
        color: #2D3A8C;
        font-weight: 600;
        font-size: 13px;
        padding: 5px 14px;
        border-radius: 20px;
        margin-bottom: 10px;
    }

    .section-divider {
        border: none;
        border-top: 1px solid #E1E4EA;
        margin: 32px 0 24px 0;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #E7E9EE;
        border-radius: 12px;
        padding: 14px 18px;
    }

    .persona-item {
        background: #ffffff;
        border: 1px solid #E7E9EE;
        border-left: 4px solid #2D3A8C;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 14px;
    }

    .stButton > button[kind="primary"] {
        background-color: #2D3A8C;
        border: none;
        font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #24307A;
    }
    </style>
""", unsafe_allow_html=True)

# Load model yang sudah dilatih sekali di notebook (bukan dilatih ulang di sini)
@st.cache_resource
def load_artifacts():
    kmeans = joblib.load("kmeans_model.joblib")
    scaler = joblib.load("scaler.joblib")
    pca = joblib.load("pca.joblib")
    metadata = joblib.load("metadata.joblib")
    return kmeans, scaler, pca, metadata

try:
    kmeans, scaler, pca, metadata = load_artifacts()
    model_loaded = True
except FileNotFoundError as e:
    model_loaded = False
    load_error = str(e)

# Header
st.markdown("""
    <div class="app-header">
        <h1>Segmentasi Pelanggan</h1>
        <p>
            Aplikasi ini memuat model K-Means yang sudah dilatih sebelumnya dan
            menerapkannya untuk mengelompokkan pelanggan baru. Model tidak dilatih
            ulang setiap kali dipakai - hasil akan konsisten dengan analisis awal.
        </p>
    </div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(
        f"Gagal memuat file model: {load_error}\n\n"
        "Pastikan kmeans_model.joblib, scaler.joblib, pca.joblib, dan "
        "metadata.joblib berada di folder yang sama dengan app.py.",
        icon=None,
    )
    st.stop()

with st.expander("Detail model yang dimuat"):
    m1, m2, m3 = st.columns(3)
    m1.metric("Jumlah Segmen (k)", metadata["k"])
    m2.metric("Silhouette Score (training)", f"{metadata['training_silhouette']:.3f}")
    m3.metric("Jumlah Fitur", len(metadata["feature_cols_order"]))
    st.caption("Fitur yang dipakai model: " + ", ".join(metadata["feature_cols_order"]))

# Langkah 1 - Upload data baru
st.markdown('<span class="step-label">LANGKAH 1</span>', unsafe_allow_html=True)
st.subheader("Unggah data pelanggan")
required_cols = metadata["raw_input_cols_needed"]
st.caption("File CSV harus memiliki kolom berikut: " + ", ".join(required_cols))

uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"], label_visibility="collapsed")

if uploaded_file is None:
    st.info("Menunggu file CSV diunggah.", icon=None)
    st.stop()

try:
    df_raw = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Gagal membaca CSV: {e}", icon=None)
    st.stop()

missing_cols = [c for c in required_cols if c not in df_raw.columns]
if missing_cols:
    st.error(f"Kolom berikut tidak ditemukan di file yang diunggah: {missing_cols}", icon=None)
    st.stop()

st.success(f"File dimuat: {uploaded_file.name} ({len(df_raw):,} baris)", icon=None)
with st.container(border=True):
    st.caption("Pratinjau 5 baris pertama")
    st.dataframe(df_raw.head(5), use_container_width=True)

run_clicked = st.button("Jalankan Analisis", type="primary", use_container_width=True)

if not run_clicked:
    st.info("Periksa pratinjau data di atas, lalu klik tombol Jalankan Analisis.", icon=None)
    st.stop()

with st.spinner("Memproses data dan menjalankan model..."):
    df = df_raw.copy()

df = df_raw.copy()
if df["Is_Returning_Customer"].dtype == object:
    df["Is_Returning_Customer"] = df["Is_Returning_Customer"].map(
        {"True": 1, "False": 0, "true": 1, "false": 0}
    ).fillna(df["Is_Returning_Customer"])
df["Is_Returning_Customer"] = df["Is_Returning_Customer"].astype(int)

for col in metadata["log_transform_cols"]:
    df[col + "_log"] = np.log1p(df[col].clip(lower=0))

X_new = df[metadata["feature_cols_order"]].copy()

before = len(X_new)
valid_mask = ~X_new.isnull().any(axis=1)
dropped = before - valid_mask.sum()
if dropped > 0:
    st.warning(f"{dropped} baris dibuang karena ada nilai kosong di fitur yang dibutuhkan model.", icon=None)

X_new_clean = X_new[valid_mask]
df_clean = df_raw[valid_mask].reset_index(drop=True)

if len(X_new_clean) == 0:
    st.error("Tidak ada baris valid tersisa setelah membuang data kosong.", icon=None)
    st.stop()
X_scaled_new = scaler.transform(X_new_clean)

# Predict pakai model yang Sudah Di Training
cluster_labels = kmeans.predict(X_scaled_new)
df_clean["Cluster"] = cluster_labels

# Proyeksi PCA pakai PCA yang sudah difit saat training
X_pca_new = pca.transform(X_scaled_new)

# Langkah 2 - Hasil segmentasi
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<span class="step-label">LANGKAH 2</span>', unsafe_allow_html=True)
st.subheader("Hasil segmentasi")

n_segments = df_clean["Cluster"].nunique()
mc1, mc2, mc3 = st.columns(3)
mc1.metric("Total Pelanggan Dianalisis", f"{len(df_clean):,}")
mc2.metric("Jumlah Segmen", n_segments)
mc3.metric("Variansi Terjelaskan (PCA 2D)", f"{pca.explained_variance_ratio_.sum()*100:.1f}%")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.markdown("**Distribusi pelanggan per segmen**")
        dist = df_clean["Cluster"].value_counts().sort_index()
        dist_df = pd.DataFrame({
            "Segmen": [f"Segmen {c}" for c in dist.index],
            "Jumlah Pelanggan": dist.values,
        })
        fig_bar = px.bar(
            dist_df, x="Segmen", y="Jumlah Pelanggan",
            color="Segmen",
            color_discrete_sequence=CLUSTER_COLORS,
        )
        fig_bar.update_layout(showlegend=False, plot_bgcolor="white", margin=dict(t=10))
        st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    with st.container(border=True):
        st.markdown("**Peta segmen (proyeksi 2 dimensi)**")
        pca_df = pd.DataFrame({
            "Komponen 1": X_pca_new[:, 0],
            "Komponen 2": X_pca_new[:, 1],
            "Segmen": [f"Segmen {c}" for c in cluster_labels],
        })
        fig_scatter = px.scatter(
            pca_df, x="Komponen 1", y="Komponen 2", color="Segmen",
            color_discrete_sequence=CLUSTER_COLORS,
            opacity=0.65,
        )
        fig_scatter.update_layout(plot_bgcolor="white", margin=dict(t=10))
        st.plotly_chart(fig_scatter, use_container_width=True)

#profiling pelanggan
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<span class="step-label">LANGKAH 3</span>', unsafe_allow_html=True)
st.subheader("Profiling segmen")
st.caption("Gunakan bagian ini untuk memahami karakteristik tiap segmen dan menentukan nama persona bisnisnya.")

profile_cols = metadata["raw_input_cols_needed"]
cluster_mean = df_clean.groupby("Cluster")[profile_cols].mean()
cluster_count = df_clean["Cluster"].value_counts().sort_index()

overall_mean = df_clean[profile_cols].mean()
overall_std = df_clean[profile_cols].std().replace(0, 1)
z_scores = (cluster_mean - overall_mean) / overall_std

def tag_label(z):
    if z > 0.3:
        return "Tinggi"
    if z < -0.3:
        return "Rendah"
    return "Sedang"

with st.container(border=True):
    st.markdown("**Rata-rata nilai asli per segmen**")
    profile_display = cluster_mean.round(2).copy()
    profile_display.insert(0, "Jumlah Pelanggan", cluster_count)
    st.dataframe(profile_display, use_container_width=True)

with st.container(border=True):
    st.markdown("**Perbandingan relatif antar segmen**")
    st.caption(
        "Nilai di atas 0 berarti segmen tersebut lebih tinggi dari rata-rata seluruh pelanggan; "
        "di bawah 0 berarti lebih rendah."
    )
    z_long = z_scores.reset_index().melt(id_vars="Cluster", var_name="Fitur", value_name="Z-Score")
    z_long["Segmen"] = "Segmen " + z_long["Cluster"].astype(str)
    fig_z = px.bar(
        z_long, x="Fitur", y="Z-Score", color="Segmen", barmode="group",
        color_discrete_sequence=CLUSTER_COLORS,
    )
    fig_z.update_layout(plot_bgcolor="white", xaxis_tickangle=-30, legend_title_text="", margin=dict(t=10))
    fig_z.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_z, use_container_width=True)

with st.container(border=True):
    st.markdown("**Fitur paling menonjol tiap segmen**")
    st.caption("Bahan acuan untuk memberi nama persona, misalnya 'Pelanggan Loyal Bernilai Tinggi'.")
    for c in sorted(z_scores.index):
        row = z_scores.loc[c].reindex(z_scores.loc[c].abs().sort_values(ascending=False).index)
        top_feats = row.index[:3]
        tags = ", ".join(f"{f}: {tag_label(row[f])}" for f in top_feats)
        st.markdown(
            f'<div class="persona-item"><b>Segmen {c}</b> ({cluster_count[c]:,} pelanggan)<br>{tags}</div>',
            unsafe_allow_html=True,
        )

OPTIONAL_CATEGORICAL_COLS = {
    "Product_Category": "Produk",
    "Payment_Method": "Metode Pembayaran",
    "Device_Type": "Perangkat",
}
available_cat_cols = [c for c in OPTIONAL_CATEGORICAL_COLS if c in df_clean.columns]

if available_cat_cols:
    with st.container(border=True):
        st.markdown("**Preferensi produk dan perilaku per segmen**")
        st.caption("Kolom ini tidak dipakai untuk membentuk segmen, tapi melengkapi gambaran persona tiap segmen.")

        tabs = st.tabs([OPTIONAL_CATEGORICAL_COLS[c] for c in available_cat_cols])
        for tab, col in zip(tabs, available_cat_cols):
            with tab:
                for c in sorted(df_clean["Cluster"].unique()):
                    sub = df_clean.loc[df_clean["Cluster"] == c, col].dropna()
                    if len(sub) == 0:
                        continue
                    top_val = sub.value_counts(normalize=True)
                    st.markdown(f"Segmen {c}: **{top_val.index[0]}** ({top_val.iloc[0]*100:.1f}% dari segmen ini)")

                ct = pd.crosstab(df_clean["Cluster"], df_clean[col], normalize="index") * 100
                ct_long = ct.reset_index().melt(id_vars="Cluster", var_name=col, value_name="Persentase")
                ct_long["Segmen"] = "Segmen " + ct_long["Cluster"].astype(str)
                fig_cat = px.bar(
                    ct_long, x=col, y="Persentase", color="Segmen", barmode="group",
                    color_discrete_sequence=CLUSTER_COLORS,
                )
                fig_cat.update_layout(plot_bgcolor="white", xaxis_tickangle=-30, legend_title_text="",
                                       yaxis_title="Persentase dalam segmen (%)", margin=dict(t=10))
                st.plotly_chart(fig_cat, use_container_width=True)
else:
    st.caption(
        "Tidak ada kolom kategori produk, metode pembayaran, atau perangkat di data yang diupload - "
        "bagian preferensi produk dilewati."
    )

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown('<span class="step-label">LANGKAH 4</span>', unsafe_allow_html=True)
st.subheader("Unduh hasil")
csv_output = df_clean.to_csv(index=False).encode("utf-8")
st.download_button(
    "Unduh CSV dengan label segmen",
    data=csv_output,
    file_name=uploaded_file.name.replace(".csv", "") + "_dengan_segmen.csv",
    mime="text/csv",
    type="primary",
    use_container_width=True,
)
