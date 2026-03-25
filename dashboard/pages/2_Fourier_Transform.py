"""Page 2 — Fourier Transform: FFT, STFT, resolution trade-off."""

import numpy as np
import streamlit as st
from utils.plots import plot_stft

st.set_page_config(page_title="Fourier Transform", page_icon="📡", layout="wide")
# 📡 
st.title("Fourier Transform & STFT")

# ── Theory ────────────────────────────────────────────────────────────────────
with st.expander("From time domain to frequency domain", expanded=True):
    st.markdown(
        """
        A raw audio waveform tells us *when* amplitude changes — but not *which frequencies* are present.
        The **Fourier Transform** decomposes any signal into a sum of sine waves at different frequencies.

        **Problem:** music is not static — the frequencies change over time (notes, beats).
        A global FFT would average everything and lose temporal structure.

        **Solution — STFT (Short-Time Fourier Transform):**
        1. Slice the signal into overlapping **windows** of length `n_fft`.
        2. Apply a **window function** (Hann) to reduce edge artefacts.
        3. Compute the **FFT** for each window.
        4. Stack results → a 2D **spectrogram** (time × frequency).

        > **Heisenberg uncertainty analogy:**
        > You cannot have perfect resolution in *both* time and frequency simultaneously.
        > Large `n_fft` → fine frequency resolution but coarse time resolution.
        > Small `n_fft` → fine time resolution but coarse frequency resolution.

        | Parameter | Effect |
        |-----------|--------|
        | `n_fft` | Window size (samples). Larger = more frequency bins, less time precision. |
        | `hop_length` | Step between windows. Smaller = finer time axis, more computation. |
        """
    )

st.divider()

# ── Audio check ───────────────────────────────────────────────────────────────
if "y" not in st.session_state:
    st.warning("No audio loaded — go to **Home** and select a track.")
    st.stop()

y = st.session_state["y"]
sr = int(st.session_state["sr"])

# ── Sliders ───────────────────────────────────────────────────────────────────
st.subheader("STFT parameter explorer")

col_a, col_b = st.columns(2)
n_fft = col_a.select_slider(
    "n_fft (window size, samples)",
    options=[256, 512, 1024, 2048, 4096],
    value=st.session_state.get("n_fft", 2048),
)
hop_length = col_b.select_slider(
    "hop_length (step size, samples)",
    options=[64, 128, 256, 512, 1024],
    value=st.session_state.get("hop_length", 512),
)

# persist to session_state
st.session_state["n_fft"] = n_fft
st.session_state["hop_length"] = hop_length

# derived info
freq_res = sr / n_fft
time_res = hop_length / sr * 1000  # ms
n_frames = 1 + (len(y) - n_fft) // hop_length

c1, c2, c3, c4 = st.columns(4)
c1.metric("Frequency resolution", f"{freq_res:.1f} Hz/bin")
c2.metric("Time resolution", f"{time_res:.1f} ms/frame")
c3.metric("Frequency bins", f"{n_fft // 2 + 1}")
c4.metric("Time frames", f"{n_frames}")

st.divider()

# ── Spectrogram ───────────────────────────────────────────────────────────────
scale = st.radio("Frequency scale", ["log", "linear"], horizontal=True)
cmap = st.selectbox("Colormap", ["magma", "viridis", "coolwarm"], index=0)

with st.spinner("Computing STFT..."):
    fig = plot_stft(y, n_fft=n_fft, hop_length=hop_length, sr=sr, scale=scale, cmap=cmap)
st.pyplot(fig)

st.info(
    f"**Tip:** try n_fft=256 (blurry frequency, sharp time) vs n_fft=4096 "
    f"(sharp frequency, blurry time) to feel the trade-off."
)

st.divider()

# ── Interactive FFT on a single frame ─────────────────────────────────────────
st.subheader("Single-frame FFT")
st.markdown("Inspect the frequency content of one window of the audio signal.")

import librosa
import matplotlib.pyplot as plt

max_frame = max(0, (len(y) - n_fft) // hop_length)
frame_idx = st.slider("Frame index", 0, max_frame, max_frame // 4)
start = frame_idx * hop_length
window = y[start : start + n_fft] * np.hanning(n_fft)
spectrum = np.abs(np.fft.rfft(window))
freqs = np.fft.rfftfreq(n_fft, d=1 / sr)

fig2, ax = plt.subplots(figsize=(10, 3))
fig2.patch.set_facecolor("#0e1117")
ax.set_facecolor("#0e1117")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
ax.title.set_color("white")
ax.fill_between(freqs, librosa.amplitude_to_db(spectrum, ref=np.max), alpha=0.7, color="#6366f1")
ax.set_xlim(0, sr / 2)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude (dB)")
ax.set_title(f"FFT of frame {frame_idx} (t={start/sr:.2f}s)")
fig2.tight_layout()
st.pyplot(fig2)
