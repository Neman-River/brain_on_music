"""Home page — audio source selector, waveform, track description."""

import os
import numpy as np
import librosa
import streamlit as st
from utils.audio import GENRES, get_gtzan_path, load_audio, load_audio_bytes
from utils.plots import plot_waveform

st.set_page_config(
    page_title="Music Classification Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Explore Audio")

# ── Audio Source Selector ─────────────────────────────────────────────────────
st.subheader("Choose an audio source")

source_tab, upload_tab = st.tabs(["GTZAN pre-loaded track", "Upload your own .wav"])

with source_tab:
    selected_genre = st.selectbox(
        "Genre",
        GENRES,
        index=GENRES.index(st.session_state.get("selected_genre", "jazz")),
        key="genre_select",
    )
    track_num = st.selectbox(
        "Track number",
        [str(i).zfill(5) for i in range(100)],
        index=36,
        key="track_num_select",
    )
    if st.button("Load this track", key="load_gtzan"):
        path = get_gtzan_path(selected_genre, track_num)
        if os.path.exists(path):
            y, sr = load_audio(path)
            st.session_state["audio_source"] = "gtzan"
            st.session_state["selected_genre"] = selected_genre
            st.session_state["audio_path"] = path
            st.session_state["y"] = y
            st.session_state["sr"] = sr
            st.session_state.setdefault("n_fft", 2048)
            st.session_state.setdefault("hop_length", 512)
            with open(path, "rb") as f:
                st.session_state["audio_bytes"] = f.read()
            st.success(f"Loaded **{selected_genre} #{track_num}** — {len(y)/sr:.1f}s @ {sr} Hz")
        else:
            st.error(f"File not found: `{path}`")

with upload_tab:
    uploaded = st.file_uploader("Upload a .wav file", type=["wav"], key="wav_uploader")
    if uploaded is not None:
        audio_bytes = uploaded.read()
        y, sr = load_audio_bytes(audio_bytes)
        st.session_state["audio_source"] = "upload"
        st.session_state["selected_genre"] = "custom"
        st.session_state["audio_path"] = uploaded.name
        st.session_state["y"] = y
        st.session_state["sr"] = sr
        st.session_state.setdefault("n_fft", 2048)
        st.session_state.setdefault("hop_length", 512)
        st.session_state["audio_bytes"] = audio_bytes
        st.success(f"Loaded **{uploaded.name}** — {len(y)/sr:.1f}s @ {sr} Hz")

# ── Audio check ───────────────────────────────────────────────────────────────
if "y" not in st.session_state:
    st.warning("No audio loaded yet — select a track above and click **Load this track**.")
    st.stop()

y_orig = st.session_state["y"]
sr_orig = st.session_state["sr"]
genre  = st.session_state.get("selected_genre", "—")

st.info(f"**Active track:** {genre} | {len(y_orig)/sr_orig:.1f}s | {sr_orig} Hz | {len(y_orig):,} samples")

st.subheader(f"Active track: {genre}")

# ── Metrics ───────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Sample rate", f"{sr_orig:,} Hz")
c2.metric("Duration", f"{len(y_orig)/sr_orig:.2f} s")
c3.metric("# Samples", f"{len(y_orig):,}")

if "audio_bytes" in st.session_state:
    st.audio(st.session_state["audio_bytes"], format="audio/wav")

st.divider()

# ── Waveform ──────────────────────────────────────────────────────────────────
st.subheader("Waveform")

resample_sr = st.slider(
    "Simulate lower sample rate (resamples for display only)",
    min_value=4000,
    max_value=int(sr_orig),
    value=int(sr_orig),
    step=1000,
    help="Lower the sample rate to see how aliasing degrades the waveform.",
)

if resample_sr < sr_orig:
    y_display = librosa.resample(y_orig, orig_sr=int(sr_orig), target_sr=resample_sr)
    sr_display = resample_sr
    st.caption(f"Displaying at {resample_sr:,} Hz  (original: {int(sr_orig):,} Hz)")
else:
    y_display = y_orig
    sr_display = int(sr_orig)

fig = plot_waveform(y_display, sr_display)
st.pyplot(fig)

st.markdown(
    """
    > **Try it:** drag the slider down to 4,000 Hz. Notice how the waveform becomes blocky and loses
    > high-frequency detail — this is **aliasing** in action.
    """
)

# ── Track Description ─────────────────────────────────────────────────────────

def describe_track(y, sr) -> str:
    """Extract key audio features and return a formatted summary string."""
    # Rhythm
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    zcr_mean = float(librosa.feature.zero_crossing_rate(y).mean())

    # Timbre
    y_harm, y_perc = librosa.effects.hpss(y)
    harmonic_mean  = float(np.abs(y_harm).mean())
    percussive_mean = float(np.abs(y_perc).mean())
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = mfccs.mean(axis=1)

    # Brightness
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()

    # Harmony
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    dominant_pitch = int(chroma.mean(axis=1).argmax())
    PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    dominant_note = PITCH_NAMES[dominant_pitch]

    # Timbre character
    h_ratio = harmonic_mean / (harmonic_mean + percussive_mean + 1e-8)
    if h_ratio > 0.6:
        timbre_char = "predominantly harmonic (tonal, melodic)"
    elif h_ratio < 0.4:
        timbre_char = "predominantly percussive (rhythmic, drum-heavy)"
    else:
        timbre_char = "balanced harmonic/percussive mix"

    # Brightness character
    brightness_hz = float(centroid)
    if brightness_hz > 3000:
        brightness_char = "bright (high-frequency energy dominant)"
    elif brightness_hz > 1500:
        brightness_char = "mid-range brightness"
    else:
        brightness_char = "dark / bass-heavy"

    lines = [
        "── Track Description ──────────────────────────────────────",
        f"  Rhythm    tempo       : {float(tempo):.1f} BPM",
        f"            zcr (mean)  : {zcr_mean:.4f}  {'(noisy/percussive)' if zcr_mean > 0.1 else '(tonal/smooth)'}",
        f"  Timbre    character   : {timbre_char}",
        f"            harmonic    : {harmonic_mean:.5f}",
        f"            percussive  : {percussive_mean:.5f}",
        f"            MFCC[0-2]   : {mfcc_means[0]:.1f}, {mfcc_means[1]:.1f}, {mfcc_means[2]:.1f}",
        f"  Brightness centroid   : {brightness_hz:.0f} Hz  ({brightness_char})",
        f"            rolloff     : {float(rolloff):.0f} Hz",
        f"  Harmony   dominant    : {dominant_note}  (strongest chroma class)",
        "────────────────────────────────────────────────────────────",
    ]
    return "\n".join(lines)


st.divider()
st.subheader("Track Description")
with st.spinner("Analysing track…"):
    desc = describe_track(y_orig, sr_orig)
st.code(desc, language=None)
