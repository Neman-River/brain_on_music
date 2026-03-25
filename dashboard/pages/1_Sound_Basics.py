"""Page 1 — Sound Basics: sine wave explorer, sound theory, feature reference."""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Sound Basics", page_icon="🔊", layout="wide")

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🎵 Understand Sound")
st.markdown(
    """
    Sound is a vibration that travels through the air as a wave of pressure changes.
    When something vibrates — a guitar string, a vocal cord, a speaker — it pushes
    and pulls the surrounding air molecules, creating alternating regions of
    compression and rarefaction that propagate outward. Your ear detects these
    pressure fluctuations and your brain interprets them as sound.
    """
)

st.divider()

# ── Sound Chain Diagram ───────────────────────────────────────────────────────
_CAPTIONS = [
    "Long/thick strings → low pitch<br>Short/thin strings → high pitch",
    "Hammer shank strikes the string,<br>setting it vibrating",
    "Vibrating string displaces air —<br>pressure wave radiates outward at ~343 m/s",
    "Eardrum wiggles in/out<br>at the same frequency",
    "Inner ear & brain decode<br>motion → perceived sound",
]
_STAGE_X = [0.5, 2.0, 3.5, 5.0, 6.5]

_fig_chain = go.Figure()

# ── Captions above pictures ───────────────────────────────────────────────────
for _i, _cap in enumerate(_CAPTIONS):
    _fig_chain.add_annotation(x=_STAGE_X[_i], y=0.90, text=_cap,
        showarrow=False, font=dict(size=11, color="#aaaaaa"), align="center")

# ── Emoji icons: piano, ear, brain ───────────────────────────────────────────
for _idx, _icon in [(0, "🎹"), (3, "👂"), (4, "🧠")]:
    _fig_chain.add_annotation(x=_STAGE_X[_idx], y=0.38, text=_icon,
        showarrow=False, font=dict(size=72), align="center")

# ── Hammer hitting string (x=2.0) ────────────────────────────────────────────
# Two piano strings (horizontal)
for _sy in [0.50, 0.58]:
    _fig_chain.add_shape(type="line", x0=1.30, x1=2.70, y0=_sy, y1=_sy,
        line=dict(color="#6366f1", width=2))
# Hammer shank
_fig_chain.add_shape(type="line", x0=2.0, x1=2.0, y0=0.18, y1=0.38,
    line=dict(color="#6366f1", width=3))
# Hammer head (felt pad — wide rectangle just touching bottom string)
_fig_chain.add_shape(type="rect", x0=1.80, x1=2.20, y0=0.38, y1=0.50,
    fillcolor="rgba(99,102,241,0.45)", line=dict(color="#6366f1", width=2))
# Impact sparks radiating from contact point
for _dx, _dy in [(-0.15, 0.10), (0.15, 0.10), (-0.19, 0.01), (0.19, 0.01)]:
    _fig_chain.add_shape(type="line", x0=2.0, y0=0.50, x1=2.0 + _dx, y1=0.50 + _dy,
        line=dict(color="#6366f1", width=1.5, dash="dot"))

# ── Sound wave (x=3.5): 3 full cycles with amplitude reference ───────────────
_wx = np.linspace(2.85, 4.15, 400)
_wy = 0.38 + 0.22 * np.sin(3 * 2 * np.pi * (_wx - 2.85) / 1.3)
_fig_chain.add_trace(go.Scatter(x=_wx, y=_wy, mode="lines",
    line=dict(color="#f59e0b", width=3), hoverinfo="skip", showlegend=False))
_fig_chain.add_shape(type="line", x0=2.85, x1=4.15, y0=0.38, y1=0.38,
    line=dict(color="#f59e0b", width=1, dash="dot"))

# ── Connecting arrows (at y=0.38) ─────────────────────────────────────────────
for _ax, _x in [(1.05, 1.30), (2.70, 2.85), (4.15, 4.45), (5.55, 5.95)]:
    _fig_chain.add_annotation(
        x=_x, y=0.38, ax=_ax, ay=0.38,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2, arrowcolor="#555",
        text="",
    )

_fig_chain.update_layout(
    height=270, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(visible=False, range=[-0.1, 7.2]),
    yaxis=dict(visible=False, range=[0.0, 1.1]),
)
st.plotly_chart(_fig_chain, use_container_width=True)

st.divider()

# ── Sine wave explorer ────────────────────────────────────────────────────────
st.subheader("Explore how frequency and amplitude relate to what we hear")
st.caption(
    "Amplitude controls the height of the wave (loudness), but browsers apply automatic "
    "gain control to audio playback — normalizing volume regardless of the signal level. "
    "Frequency changes are always audible because they change pitch, which cannot be normalized away."
)

col_f, col_a = st.columns(2)
freq_hz   = col_f.slider("Frequency (Hz)", 100, 4000, 440, 10)
amplitude = col_a.slider("Amplitude", 0.1, 1.0, 0.5, 0.1)

demo_sr = 22050
t    = np.linspace(0, 1.0, demo_sr, endpoint=False)
sine = amplitude * np.sin(2 * np.pi * freq_hz * t)

fig, ax = plt.subplots(figsize=(10, 2))
fig.patch.set_facecolor("#0e1117")
ax.set_facecolor("#0e1117")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
ax.title.set_color("white")
ax.plot(t[:2000], sine[:2000], color="#1DB954", linewidth=1.2)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title(f"Sine wave: {freq_hz} Hz, amplitude={amplitude}")
fig.tight_layout()
st.pyplot(fig)
st.audio(sine.astype(np.float32), sample_rate=demo_sr)

NOTE_NAMES = {
    "C4 (middle C)": (262, "the middle C on a piano — the anchor note for reading music"),
    "A4 (concert pitch)": (440, "the universal tuning reference — orchestras tune to this note"),
    "A5": (880, "one octave above A4 — doubling frequency raises pitch by exactly one octave"),
}
note = next((f"**{name}** ({hz} Hz) — {desc}" for name, (hz, desc) in NOTE_NAMES.items() if abs(freq_hz - hz) <= 20), None)
if note:
    st.caption(f"You're near {note}")
else:
    st.caption(
        "**440 Hz** is A4 — the concert pitch reference used by orchestras worldwide. "
        "Human hearing spans roughly 20 Hz (deep bass) to 20,000 Hz (high treble)."
    )


st.markdown(
    """
    - **Sample** — a single amplitude measurement at one point in time.
    - **Sample rate (Hz)** — how many samples are taken per second.

    <span style="color:#1DB954;">**Nyquist theorem**</span> — to accurately reconstruct a sound wave, your sample rate must be at least twice the highest frequency present in the audio.
    """,
    unsafe_allow_html=True,
)

st.divider()

# ── Pipeline Diagram ──────────────────────────────────────────────────────────
st.subheader("From raw audio to model input")

# Step 1 ── Load raw audio
st.markdown("**Step 1 — Load raw audio**")
st.caption("Reads a WAV file and returns an array of amplitude samples at the target sample rate.")
st.code("y, sr = librosa.load(path, sr=22050)", language="python")

st.markdown("<p style='text-align:center;font-size:1.4rem;margin:0'>↓</p>", unsafe_allow_html=True)

# Step 2 ── STFT
st.markdown("**Step 2 — Short-Time Fourier Transform (STFT)**")
st.caption(
    "Slices `y` into overlapping windows of `n_fft=2048` samples shifted by `hop_length=512` (~23 ms steps), "
    "multiplies each window by a Hann taper to avoid edge artifacts, then computes the DFT per window. "
    "Result: complex matrix of shape (1025 × ~2584) for a 30-sec clip."
)
st.code("D = librosa.stft(y, n_fft=2048, hop_length=512)", language="python")
st.latex(r"X[f,\,t] = \sum_{n=0}^{N-1} x[n]\cdot w[n - t\cdot h]\cdot e^{-j2\pi fn/N}")
st.caption("WHY: time-domain samples carry no frequency information — STFT reveals which frequencies are active at every moment.")

st.markdown("<p style='text-align:center;font-size:1.4rem;margin:0'>↓</p>", unsafe_allow_html=True)

# Step 3 ── Magnitude spectrogram
st.markdown("**Step 3 — Magnitude spectrogram**")
st.caption("Discard phase, keep only amplitude. Still (1025 × 2584), now real-valued.")
st.code("S = np.abs(D)", language="python")
st.latex(r"S[f,\,t] = \bigl|X[f,\,t]\bigr|")

st.markdown("<p style='text-align:center;font-size:1.4rem;margin:0'>↓</p>", unsafe_allow_html=True)

# Step 4 ── Log-magnitude (dB)
st.markdown("**Step 4 — Log-magnitude (dB) spectrogram**")
st.caption(
    "WHY: human hearing is logarithmic — the dB scale matches perception "
    "and compresses the dynamic range from ~10 000× down to ~80 dB."
)
st.code("S_db = librosa.amplitude_to_db(S, ref=np.max)", language="python")
st.latex(r"\text{dB}[f,\,t] = 20\cdot\log_{10}\!\bigl(S[f,\,t]\bigr)")

st.markdown("<p style='text-align:center;font-size:1.4rem;margin:0'>↓</p>", unsafe_allow_html=True)

# Step 5 ── Mel spectrogram (optional)
st.markdown("**Step 5 — Mel spectrogram** *(optional)*")
st.caption(
    "Warps the linear frequency axis to the Mel scale, spacing frequencies the way humans perceive pitch. "
    "Produces a compact 2-D image ready for CNNs or direct visualization."
)
st.code(
    "M    = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)\n"
    "M_db = librosa.power_to_db(M, ref=np.max)",
    language="python",
)
st.latex(r"\text{Mel}(f) = 2595\cdot\log_{10}\!\!\left(1 + \frac{f}{700}\right)")

st.divider()

# ── Feature Category Bubble Map ───────────────────────────────────────────────
st.subheader("Audio features at a glance")

_CATEGORIES = [
    {"name": "RHYTHM",     "desc": "how beats and onsets<br>unfold over time",                              "features": ["ZCR", "Tempo"],                          "x": 1.0, "color": "#34d399"},
    {"name": "TIMBRE",     "desc": "the texture and color<br>of a sound",                                   "features": ["Harmonic/Percussive", "MFCCs"],          "x": 2.5, "color": "#6366f1"},
    {"name": "BRIGHTNESS", "desc": "how energy is distributed<br>across frequencies<br>in a given frame",  "features": ["Spectral Centroid", "Spectral Rolloff"], "x": 4.0, "color": "#f59e0b"},
    {"name": "HARMONY",    "desc": "which pitch classes<br>and chords are present",                         "features": ["Chroma"],                                "x": 5.5, "color": "#a78bfa"},
]

_fig_map = go.Figure()

# bubbles — markers only (scatter text doesn't support multiline)
_fig_map.add_trace(go.Scatter(
    x=[c["x"] for c in _CATEGORIES],
    y=[0.0] * 4,
    mode="markers",
    marker=dict(size=190, color=[c["color"] for c in _CATEGORIES], opacity=0.88),
    hoverinfo="skip",
))

# text via annotations so <br> line-breaks work
for c in _CATEGORIES:
    lines = [f"<b>{c['name']}</b>", "─────"] + c["features"]
    _fig_map.add_annotation(
        x=c["x"], y=0.0,
        text="<br>".join(lines),
        showarrow=False,
        font=dict(size=12, color="white", family="Arial Black"),
        align="center",
        xref="x", yref="y",
    )
_fig_map.update_layout(
    height=320,
    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",
    showlegend=False,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(visible=False, range=[0.0, 6.5]),
    yaxis=dict(visible=False, range=[-1.2, 1.2]),
)
st.plotly_chart(_fig_map, width="stretch")

st.markdown("### Feature extraction schema")
st.caption("Each feature is extracted from `y` and summarised as mean + variance for the 30-second CSV.")

# ── RHYTHM ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='background:#34d39922;border-left:4px solid #34d399;"
    "padding:8px 14px;border-radius:4px;margin:18px 0 10px 0'>"
    "<span style='color:#34d399;font-weight:700;font-size:1rem;letter-spacing:.08em'>RHYTHM</span>"
    "<span style='color:#aaa;font-size:0.85rem;margin-left:10px'>how beats and onsets unfold over time</span>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("**Zero Crossing Rate (ZCR)**")
st.caption("Counts how often the waveform crosses zero per frame. High in noisy/percussive sounds, low in tonal ones. → `zcr_mean`, `zcr_var`")
st.code("zcr = librosa.feature.zero_crossing_rate(y, hop_length=512)", language="python")
with st.expander("formula"):
    st.latex(r"\mathrm{ZCR}[t] = \frac{1}{N} \sum_{n=1}^{N} \mathbf{1}\bigl[x[n]\cdot x[n-1] < 0\bigr]")

st.markdown("<p style='text-align:center;font-size:1.4rem;margin:4px 0'>↓</p>", unsafe_allow_html=True)

st.markdown("**Tempo / BPM**")
st.caption("It estimates the most consistent repeating time interval between beats. The algorithm tries to find the main steady pulse → `tempo`")
st.code("tempo, beats = librosa.beat.beat_track(y=y, sr=sr)", language="python")
with st.expander("formula"):
    st.latex(r"\hat{T} = \arg\max_{T}\;\sum_{t}\,o[t]\cdot o[t + T]")
    st.caption("where *o[t]* is the onset strength envelope (peaks at note/beat onsets)")

# ── TIMBRE ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='background:#6366f122;border-left:4px solid #6366f1;"
    "padding:8px 14px;border-radius:4px;margin:18px 0 10px 0'>"
    "<span style='color:#6366f1;font-weight:700;font-size:1rem;letter-spacing:.08em'>TIMBRE</span>"
    "<span style='color:#aaa;font-size:0.85rem;margin-left:10px'>the texture and color of a sound</span>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("**Harmonic / Percussive Separation (HPSS)**")
st.caption("Splits the signal into tonal (harmonic, smooth, continuous sounds like notes or chords) and transient (percussive, sharp, hit-like sounds like drums) components via median filtering on the regular spectrogram - STFT magnitude(by separating steady patterns from sudden changes). → `harmonic_mean/var`, `percussive_mean/var`")
st.code("y_harm, y_perc = librosa.effects.hpss(y)", language="python")
with st.expander("formula"):
    st.latex(r"\hat{H}[f,t] = \mathrm{MedFilt}_t\!\bigl(S[f,t]\bigr), \qquad \hat{P}[f,t] = \mathrm{MedFilt}_f\!\bigl(S[f,t]\bigr)")
    st.caption(
        "Median filtering is a way of smoothing data by replacing each value with the \"middle\" value of its nearby "
        "neighbors, which removes sudden spikes while keeping the overall pattern. For example, in a sequence like "
        "(2, 50, 3), the median is 3, so the large spike (50) disappears, leaving a smoother signal. In audio "
        "spectrograms, this helps highlight structure: smoothing across time keeps steady tones like a sustained "
        "piano note or a violin, while smoothing across frequency keeps short, sharp sounds like drum hits or claps."
    )

st.markdown("<p style='text-align:center;font-size:1.4rem;margin:4px 0'>↓</p>", unsafe_allow_html=True)

st.markdown("**MFCCs** *(Mel-Frequency Cepstral Coefficients)*")
st.caption("Compact timbre fingerprint: spectrum → Mel filterbank → log → DCT. → `mfcc1–20_mean/var` (40 columns)")
st.code("mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)", language="python")
with st.expander("formula"):
    st.latex(r"c_n[t] = \sum_{m=1}^{M} \log S_{\mathrm{mel}}[m,t]\cdot \cos\!\left(\frac{\pi n\!\left(m - \tfrac{1}{2}\right)}{M}\right)")

# ── BRIGHTNESS ────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='background:#f59e0b22;border-left:4px solid #f59e0b;"
    "padding:8px 14px;border-radius:4px;margin:18px 0 10px 0'>"
    "<span style='color:#f59e0b;font-weight:700;font-size:1rem;letter-spacing:.08em'>BRIGHTNESS</span>"
    "<span style='color:#aaa;font-size:0.85rem;margin-left:10px'>how energy is distributed across frequencies in a given frame</span>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("**Spectral Centroid**")
st.caption("Weighted mean of active frequencies — perceived 'brightness'. → `spectral_centroid_mean`, `spectral_centroid_var`")
st.code("centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)", language="python")
with st.expander("formula"):
    st.latex(r"C[t] = \frac{\displaystyle\sum_{f} f \cdot |X[f,t]|}{\displaystyle\sum_{f} |X[f,t]|}")

st.markdown("<p style='text-align:center;font-size:1.4rem;margin:4px 0'>↓</p>", unsafe_allow_html=True)

st.markdown("**Spectral Rolloff**")
st.caption("Frequency below which 85 % of total spectral energy is concentrated. → `rolloff_mean`, `rolloff_var`")
st.code("rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512, roll_percent=0.85)", language="python")
with st.expander("formula"):
    st.latex(r"\sum_{f \leq f_R} |X[f,t]|^2 \;=\; 0.85 \cdot \sum_{f} |X[f,t]|^2")

# ── HARMONY ───────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='background:#a78bfa22;border-left:4px solid #a78bfa;"
    "padding:8px 14px;border-radius:4px;margin:18px 0 10px 0'>"
    "<span style='color:#a78bfa;font-weight:700;font-size:1rem;letter-spacing:.08em'>HARMONY</span>"
    "<span style='color:#aaa;font-size:0.85rem;margin-left:10px'>which pitch classes and chords are present</span>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("**Chroma (Pitch Class Profile)**")
st.caption("Energy across 12 pitch classes (C–B), octave-invariant. Captures harmonic and chord content. → `chroma_stft_mean`, `chroma_stft_var`")
st.code("chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)", language="python")
with st.expander("formula"):
    st.latex(r"\mathrm{Chroma}_p[t] = \sum_{\{f\;:\;\mathrm{pitch}(f)\bmod 12\,=\,p\}} |X[f,t]|^2")

st.caption("**Total: 57 feature columns** extracted per track (plus `filename` and `label`)")
