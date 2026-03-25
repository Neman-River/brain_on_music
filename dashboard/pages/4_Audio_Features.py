"""Page 4 — Audio Features: 7 tabs with theory + plots + metrics."""

import numpy as np
import streamlit as st
from utils.audio import (
    extract_zero_crossing_rate,
    extract_harmonic_percussive,
    extract_tempo,
    extract_spectral_centroid,
    extract_spectral_rolloff,
    extract_mfcc,
    extract_chroma,
    load_audio,
    get_gtzan_path,
    GENRES,
)
from utils.plots import (
    plot_zcr,
    plot_harmonic_percussive,
    plot_spectral_centroid,
    plot_spectral_rolloff,
    plot_mfcc,
    plot_chroma,
    plot_waveform,
)

st.set_page_config(page_title="Audio Features", page_icon="🎛️", layout="wide")
st.title("🎛️ Audio Features")

st.markdown(
    "Seven features are extracted from each audio clip. "
    "Use the tabs below to explore each one with theory and live plots."
)

if "y" not in st.session_state:
    st.warning("No audio loaded — go to **Home** and select a track.")
    st.stop()

y = st.session_state["y"]
sr = int(st.session_state["sr"])
hop_length = st.session_state.get("hop_length", 512)
genre = st.session_state.get("selected_genre", "—")

# ── Genre overlay helper ──────────────────────────────────────────────────────
@st.cache_data
def _load_all_genres():
    """Load one track per genre for comparison."""
    data = {}
    for g in GENRES:
        path = get_gtzan_path(g)
        try:
            yg, sr_g = load_audio(path)
            data[g] = (yg, int(sr_g))
        except Exception:
            pass
    return data


# ── Tabs ─────────────────────────────────────────────────────────────────────
tabs = st.tabs(["ZCR", "Harmonic/Percussive", "Tempo", "Spectral Centroid", "Spectral Rolloff", "MFCCs", "Chroma"])

# ─── 1. ZCR ──────────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Zero Crossing Rate (ZCR)")
    with st.expander("Theory"):
        st.markdown(
            """
            The **Zero Crossing Rate** counts how often the waveform crosses the zero amplitude axis per frame.

            - High ZCR → noisy, percussive, or speech-like signals (lots of rapid oscillations)
            - Low ZCR → tonal, sustained sounds

            > Metal and rock typically have higher mean ZCR than classical or jazz,
            > because distorted guitar and drums create dense, noisy waveforms.
            """
        )
    zcr_arr, zcr_mean = extract_zero_crossing_rate(y)
    st.metric("Mean ZCR", f"{zcr_mean:.4f}")
    fig = plot_zcr(y, sr)
    st.pyplot(fig)

    if st.checkbox("Compare ZCR across all genres", key="zcr_compare"):
        import matplotlib.pyplot as plt
        all_data = _load_all_genres()
        fig2, ax = plt.subplots(figsize=(10, 3))
        fig2.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        means = {g: float(np.mean(extract_zero_crossing_rate(yd)[0])) for g, (yd, _) in all_data.items()}
        ax.bar(means.keys(), means.values(), color="#34d399")
        ax.set_ylabel("Mean ZCR")
        ax.set_title("Mean Zero Crossing Rate by Genre")
        fig2.tight_layout()
        st.pyplot(fig2)

# ─── 2. Harmonic / Percussive ─────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Harmonic & Percussive Separation (HPSS)")
    with st.expander("Theory"):
        st.markdown(
            """
            `librosa.effects.hpss` separates a signal into:
            - **Harmonic**: tonal content — sustained notes, melodies, chords
            - **Percussive**: transient content — drums, attacks, clicks

            This works by exploiting the structure of the spectrogram:
            - Harmonic components form **horizontal stripes** (constant pitch over time).
            - Percussive components form **vertical stripes** (broadband energy bursts).

            A median filter is applied along each axis to isolate each component.
            """
        )
    with st.spinner("Separating harmonic and percussive..."):
        y_h, y_p = extract_harmonic_percussive(y)
    fig = plot_harmonic_percussive(y_h, y_p, sr)
    st.pyplot(fig)

    c1, c2 = st.columns(2)
    c1.metric("Harmonic RMS", f"{float(np.sqrt(np.mean(y_h**2))):.4f}")
    c2.metric("Percussive RMS", f"{float(np.sqrt(np.mean(y_p**2))):.4f}")

# ─── 3. Tempo ─────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Tempo / BPM")
    with st.expander("Theory"):
        st.markdown(
            """
            `librosa.beat.beat_track` estimates the **global tempo** (in BPM) by:
            1. Computing an **onset strength envelope** — peaks at note/beat onsets.
            2. Auto-correlating the envelope to find the dominant periodicity.
            3. Returning the tempo corresponding to the peak lag.

            It also returns **beat frame indices** — the estimated positions of each beat.

            > Disco and hiphop tend to cluster around 100–130 BPM.
            > Classical and jazz vary widely.
            """
        )
    with st.spinner("Estimating tempo..."):
        bpm = extract_tempo(y, sr)
    st.metric("Estimated BPM", f"{bpm:.1f}")

    # Beat overlaid on waveform
    import librosa
    import matplotlib.pyplot as plt
    _, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    times = np.linspace(0, len(y) / sr, num=len(y))
    fig2, ax = plt.subplots(figsize=(10, 3))
    fig2.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.plot(times, y, color="#1DB954", linewidth=0.4, alpha=0.7)
    for bt in beat_times:
        ax.axvline(bt, color="#f59e0b", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Waveform with beat markers ({bpm:.1f} BPM)")
    fig2.tight_layout()
    st.pyplot(fig2)

    if st.checkbox("Compare BPM across all genres", key="bpm_compare"):
        all_data = _load_all_genres()
        bpms = {g: extract_tempo(yd, sr_g) for g, (yd, sr_g) in all_data.items()}
        fig3, ax = plt.subplots(figsize=(10, 3))
        fig3.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.bar(bpms.keys(), bpms.values(), color="#f59e0b")
        ax.set_ylabel("BPM")
        ax.set_title("Estimated BPM by Genre (track .00036)")
        fig3.tight_layout()
        st.pyplot(fig3)

# ─── 4. Spectral Centroid ─────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Spectral Centroid")
    with st.expander("Theory"):
        st.markdown(
            """
            The **spectral centroid** is the weighted mean of frequencies present in a frame,
            where the weight is the magnitude of each frequency bin.

            Intuitively, it's the "centre of mass" of the spectrum — a measure of **brightness**.

            - High centroid → bright, treble-heavy sound (cymbals, distorted guitar)
            - Low centroid → dark, bass-heavy sound (bass guitar, low vocals)

            Formula: `centroid = Σ(f * |X(f)|) / Σ|X(f)|`
            """
        )
    centroid = extract_spectral_centroid(y, sr, hop_length)
    st.metric("Mean Spectral Centroid", f"{float(np.mean(centroid)):.0f} Hz")
    fig = plot_spectral_centroid(y, sr, hop_length)
    st.pyplot(fig)

# ─── 5. Spectral Rolloff ──────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Spectral Rolloff")
    with st.expander("Theory"):
        st.markdown(
            """
            The **spectral rolloff** is the frequency below which a given percentage
            (default 85%) of the total spectral energy is contained.

            It is another measure of spectral shape / brightness:
            - High rolloff → energy spread across a wide frequency range
            - Low rolloff → energy concentrated in low frequencies

            Useful for distinguishing speech (high rolloff) from music with heavy bass (low rolloff).
            """
        )
    rolloff = extract_spectral_rolloff(y, sr, hop_length)
    st.metric("Mean Spectral Rolloff", f"{float(np.mean(rolloff)):.0f} Hz")
    fig = plot_spectral_rolloff(y, sr, hop_length)
    st.pyplot(fig)

# ─── 6. MFCCs ─────────────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("MFCCs — Mel-Frequency Cepstral Coefficients")
    with st.expander("Theory"):
        st.markdown(
            """
            MFCCs are one of the most widely used features in audio classification.

            **How they're computed:**
            1. Apply STFT → power spectrum per frame
            2. Map to **Mel filterbank** (logarithmic frequency scale)
            3. Take the **log** of Mel energies
            4. Apply **DCT** (Discrete Cosine Transform) → decorrelate coefficients
            5. Keep the first N coefficients (typically 13–20)

            **Intuition:** MFCCs capture the *shape* (timbre) of the spectrum, not the pitch.
            Two instruments playing the same note will have different MFCCs.

            > The first MFCC (C0 or C1) captures overall loudness.
            > Higher coefficients capture finer spectral detail.
            """
        )
    n_mfcc = st.slider("Number of MFCC coefficients", 5, 40, 20, key="n_mfcc_slider")
    mfcc = extract_mfcc(y, sr, n_mfcc=n_mfcc)
    st.metric("MFCC shape", f"{mfcc.shape[0]} coefficients × {mfcc.shape[1]} frames")

    fig = plot_mfcc(y, sr, n_mfcc=n_mfcc, hop_length=hop_length)
    st.pyplot(fig)

    # Mean MFCC bar chart
    import matplotlib.pyplot as plt
    mean_mfcc = np.mean(mfcc, axis=1)
    fig2, ax = plt.subplots(figsize=(10, 2.5))
    fig2.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.bar(range(len(mean_mfcc)), mean_mfcc, color="#6366f1")
    ax.set_xlabel("MFCC Coefficient Index")
    ax.set_ylabel("Mean Value")
    ax.set_title("Mean MFCC per Coefficient")
    fig2.tight_layout()
    st.pyplot(fig2)

# ─── 7. Chroma ────────────────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("Chroma Features")
    with st.expander("Theory"):
        st.markdown(
            """
            **Chroma features** (also called *Pitch Class Profiles*) represent the energy
            distribution across the 12 pitch classes (C, C#, D, D#, E, F, F#, G, G#, A, A#, B).

            They are **octave-invariant**: a C played at 261 Hz and 523 Hz map to the same bin.

            - Useful for chord and key detection
            - Classical music often shows strong chroma patterns (clear harmonic structure)
            - Percussion-heavy genres show more uniform chroma

            Computed via `librosa.feature.chroma_stft` from the STFT.
            """
        )
    chroma = extract_chroma(y, sr, hop_length)
    fig = plot_chroma(y, sr, hop_length)
    st.pyplot(fig)

    import matplotlib.pyplot as plt
    pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    mean_chroma = np.mean(chroma, axis=1)
    fig2, ax = plt.subplots(figsize=(10, 2.5))
    fig2.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.bar(pitch_classes, mean_chroma, color="#a78bfa")
    ax.set_xlabel("Pitch Class")
    ax.set_ylabel("Mean Energy")
    ax.set_title("Mean Chroma Energy per Pitch Class")
    fig2.tight_layout()
    st.pyplot(fig2)
