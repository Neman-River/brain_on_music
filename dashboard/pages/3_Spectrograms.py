"""Page 3 — Spectrograms: linear, log, Mel with colormap selector."""

import streamlit as st
from utils.plots import plot_stft, plot_mel_spectrogram

st.set_page_config(page_title="Spectrograms", page_icon="🌈", layout="wide")
# 🌈 
st.title("Spectrograms")

# ── Theory ────────────────────────────────────────────────────────────────────
with st.expander("Reading spectrograms", expanded=True):
    st.markdown(
        """
        A **spectrogram** is a 2D image of the STFT: time on the x-axis, frequency on the y-axis,
        and colour representing energy (in dB).

        | Variant | Y-axis | Use case |
        |---------|--------|----------|
        | **Linear** | Hz, evenly spaced | See raw frequency bins |
        | **Log** | Hz, logarithmic scale | Matches human pitch perception better — octaves are equal height |
        | **Mel** | Mel scale, nonlinear | Matches how humans perceive pitch changes; widely used in ML |

        The **Mel scale** was designed so that perceptually equal pitch intervals are equally spaced.
        A filterbank of triangular filters maps linear Hz to Mel bins.

        > Most audio ML models (e.g. CNN classifiers) use **log-Mel spectrograms** as input.
        """
    )

st.divider()

# ── Audio check ───────────────────────────────────────────────────────────────
if "y" not in st.session_state:
    st.warning("No audio loaded — go to **Home** and select a track.")
    st.stop()

y = st.session_state["y"]
sr = int(st.session_state["sr"])
n_fft = st.session_state.get("n_fft", 2048)
hop_length = st.session_state.get("hop_length", 512)

# ── Controls ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
spectype = col1.radio(
    "Spectrogram type",
    ["Linear frequency", "Log frequency", "Mel scale"],
    horizontal=False,
)
cmap = col2.selectbox("Colormap", ["magma", "viridis", "coolwarm", "inferno", "plasma"], index=0)

st.caption(f"Using n_fft={n_fft}, hop_length={hop_length} (set on the Fourier Transform page)")

st.divider()

# ── Plot ──────────────────────────────────────────────────────────────────────
with st.spinner("Rendering spectrogram..."):
    if spectype == "Mel scale":
        fig = plot_mel_spectrogram(y, sr, n_fft=n_fft, hop_length=hop_length, cmap=cmap)
    elif spectype == "Log frequency":
        fig = plot_stft(y, n_fft=n_fft, hop_length=hop_length, sr=sr, scale="log", cmap=cmap)
    else:
        fig = plot_stft(y, n_fft=n_fft, hop_length=hop_length, sr=sr, scale="linear", cmap=cmap)

st.pyplot(fig)

# ── Compare side-by-side ──────────────────────────────────────────────────────
st.divider()
if st.checkbox("Compare all three side-by-side"):
    c1, c2, c3 = st.columns(3)
    with st.spinner("Rendering all three..."):
        fig_lin = plot_stft(y, n_fft=n_fft, hop_length=hop_length, sr=sr, scale="linear", cmap=cmap)
        fig_log = plot_stft(y, n_fft=n_fft, hop_length=hop_length, sr=sr, scale="log", cmap=cmap)
        fig_mel = plot_mel_spectrogram(y, sr, n_fft=n_fft, hop_length=hop_length, cmap=cmap)
    with c1:
        st.caption("Linear")
        st.pyplot(fig_lin)
    with c2:
        st.caption("Log")
        st.pyplot(fig_log)
    with c3:
        st.caption("Mel")
        st.pyplot(fig_mel)
