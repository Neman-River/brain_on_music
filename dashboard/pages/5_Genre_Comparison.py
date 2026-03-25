"""Page 5 — Genre Comparison: aggregated EDA from features_30_sec.csv."""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Genre Comparison", page_icon="📊", layout="wide")

st.title("Genre Comparison")
st.markdown(
    "Aggregated EDA using `features_30_sec.csv` — 1,000 tracks (100 per genre) "
    "with 57 pre-extracted audio features."
)

# ── Load data ─────────────────────────────────────────────────────────────────
CSV_PATH = "Data/Data/features_30_sec.csv"

@st.cache_data
def load_features() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    return df.rename(columns={
        "harmony_mean":  "harmonic_mean",
        "harmony_var":   "harmonic_var",
        "perceptr_mean": "percussive_mean",
        "perceptr_var":  "percussive_var",
    })

try:
    df = load_features()
except FileNotFoundError:
    st.error(f"Could not find `{CSV_PATH}`. Make sure the GTZAN dataset is downloaded.")
    st.stop()

st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

# ── Column groups ─────────────────────────────────────────────────────────────
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
feature_groups = {
    "ZCR": [c for c in numeric_cols if "zcr" in c],
    "Chroma": [c for c in numeric_cols if "chroma" in c and "stft" not in c],
    "Spectral": [c for c in numeric_cols if "spectral" in c],
    "MFCCs": [c for c in numeric_cols if "mfcc" in c],
    "Tempo / BPM": [c for c in numeric_cols if "tempo" in c or "beats" in c],
    "RMS / Energy": [c for c in numeric_cols if "rms" in c],
    "Harmonic / Percussive": [c for c in numeric_cols if "harmonic" in c or "percussive" in c],
}
summary_features = [
    "zcr_mean", "chroma_stft_mean", "spectral_centroid_mean",
    "spectral_bandwidth_mean", "rolloff_mean", "rms_mean",
    "harmonic_mean", "percussive_mean", "tempo",
]
summary_features = [f for f in summary_features if f in df.columns]

st.divider()

# ── Section 1: Distribution by genre ─────────────────────────────────────────
st.subheader("Feature distributions by genre")

feat = st.selectbox(
    "Feature to plot",
    summary_features + [c for c in numeric_cols if c not in summary_features and c != "length"],
    index=0,
)

fig = px.box(
    df,
    x="label",
    y=feat,
    color="label",
    points="outliers",
    title=f"Distribution of `{feat}` by genre",
    template="plotly_dark",
    color_discrete_sequence=px.colors.qualitative.Vivid,
)
fig.update_layout(showlegend=False, xaxis_title="Genre", yaxis_title=feat)
st.plotly_chart(fig)

st.divider()

# ── Section 2: Pairplot / scatter matrix ─────────────────────────────────────
st.subheader("Feature scatter matrix")

selected_feats = st.multiselect(
    "Select 3–5 features to compare",
    summary_features,
    default=summary_features[:4],
)

if len(selected_feats) >= 2:
    fig2 = px.scatter_matrix(
        df,
        dimensions=selected_feats,
        color="label",
        title="Scatter Matrix",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        opacity=0.5,
    )
    fig2.update_traces(marker=dict(size=3))
    st.plotly_chart(fig2)
else:
    st.info("Select at least 2 features.")

st.divider()

# ── Section 3: Parallel coordinates ──────────────────────────────────────────
st.subheader("Parallel coordinates plot")
st.markdown("Each line is one track. Colour = genre.")

pc_feats = st.multiselect(
    "Features for parallel coordinates",
    summary_features,
    default=summary_features,
    key="pc_feats",
)

if len(pc_feats) >= 2:
    df_norm = df[pc_feats + ["label"]].copy()
    for f in pc_feats:
        df_norm[f] = (df_norm[f] - df_norm[f].min()) / (df_norm[f].max() - df_norm[f].min() + 1e-9)
    genre_codes = {g: i for i, g in enumerate(sorted(df["label"].unique()))}
    df_norm["genre_code"] = df_norm["label"].map(genre_codes)

    fig3 = px.parallel_coordinates(
        df_norm,
        dimensions=pc_feats,
        color="genre_code",
        color_continuous_scale=px.colors.sequential.Turbo,
        title="Parallel Coordinates (normalised features)",
        template="plotly_dark",
    )
    st.plotly_chart(fig3)

st.divider()

# ── Section 4: Correlation heatmap ────────────────────────────────────────────
st.subheader("Correlation heatmap")

heatmap_feats = st.multiselect(
    "Select features for heatmap",
    summary_features,
    default=summary_features,
    key="heatmap_feats",
)

if len(heatmap_feats) >= 2:
    corr = df[heatmap_feats].corr()
    fig4, ax = plt.subplots(figsize=(max(6, len(heatmap_feats)), max(5, len(heatmap_feats) - 1)))
    fig4.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax,
        linewidths=0.5,
        annot_kws={"size": 8},
    )
    ax.set_title("Feature Correlation Matrix", color="white", pad=12)
    ax.tick_params(colors="white")
    plt.xticks(rotation=45, ha="right", color="white")
    plt.yticks(rotation=0, color="white")
    fig4.tight_layout()
    st.pyplot(fig4)

st.divider()

# ── Section 5: Genre mean radar ───────────────────────────────────────────────
st.subheader("Genre mean feature profiles")

genre_sel = st.multiselect(
    "Select genres to compare",
    sorted(df["label"].unique()),
    default=["blues", "classical", "metal", "jazz", "pop"],
)

if genre_sel and len(summary_features) >= 3:
    means = df[df["label"].isin(genre_sel)].groupby("label")[summary_features].mean()
    from sklearn.preprocessing import minmax_scale
    means_norm = pd.DataFrame(
        minmax_scale(means, axis=0),
        index=means.index,
        columns=means.columns,
    )
    fig5 = px.line_polar(
        means_norm.reset_index().melt(id_vars="label", var_name="feature", value_name="value"),
        r="value",
        theta="feature",
        color="label",
        line_close=True,
        template="plotly_dark",
        title="Normalised mean features (radar)",
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig5.update_traces(fill="toself", opacity=0.4)
    st.plotly_chart(fig5)
