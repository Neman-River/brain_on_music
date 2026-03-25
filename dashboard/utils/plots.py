"""Reusable matplotlib/plotly figure builders."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
import librosa
import librosa.display
from sklearn.preprocessing import minmax_scale


def _new_fig(figsize=(10, 3)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    return fig, ax


def plot_waveform(y: np.ndarray, sr: int) -> matplotlib.figure.Figure:
    fig, ax = _new_fig(figsize=(10, 3))
    times = np.linspace(0, len(y) / sr, num=len(y))
    ax.plot(times, y, color="#1DB954", linewidth=0.5, alpha=0.9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    ax.set_xlim(0, times[-1])
    fig.tight_layout()
    return fig


def plot_stft(
    y: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    sr: int = 22050,
    scale: str = "log",
    cmap: str = "magma",
) -> matplotlib.figure.Figure:
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    fig, ax = _new_fig(figsize=(10, 4))
    if scale == "log":
        img = librosa.display.specshow(
            S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="log", ax=ax, cmap=cmap
        )
        ax.set_ylabel("Frequency (Hz, log)")
    else:
        img = librosa.display.specshow(
            S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz", ax=ax, cmap=cmap
        )
        ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(f"STFT Spectrogram (n_fft={n_fft}, hop_length={hop_length})")
    ax.set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def plot_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    cmap: str = "magma",
) -> matplotlib.figure.Figure:
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = _new_fig(figsize=(10, 4))
    img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel", ax=ax, cmap=cmap)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Mel)")
    fig.tight_layout()
    return fig


def plot_spectral_centroid(y: np.ndarray, sr: int, hop_length: int = 512) -> matplotlib.figure.Figure:
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    frames = np.arange(len(centroid))
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

    fig, ax = _new_fig(figsize=(10, 3))
    ax.plot(times, minmax_scale(centroid), color="#FF6B6B", linewidth=1.2, label="Spectral Centroid")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Centroid")
    ax.set_title("Spectral Centroid (normalized)")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    fig.tight_layout()
    return fig


def plot_spectral_rolloff(y: np.ndarray, sr: int, hop_length: int = 512) -> matplotlib.figure.Figure:
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    frames = np.arange(len(rolloff))
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

    fig, ax = _new_fig(figsize=(10, 3))
    ax.plot(times, minmax_scale(rolloff), color="#4ECDC4", linewidth=1.2, label="Spectral Rolloff")
    ax.plot(times, minmax_scale(centroid), color="#FF6B6B", linewidth=1.2, alpha=0.7, label="Spectral Centroid")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Frequency")
    ax.set_title("Spectral Rolloff vs Centroid (normalized)")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    fig.tight_layout()
    return fig


def plot_mfcc(y: np.ndarray, sr: int, n_mfcc: int = 20, hop_length: int = 512) -> matplotlib.figure.Figure:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    fig, ax = _new_fig(figsize=(10, 4))
    img = librosa.display.specshow(mfcc, sr=sr, hop_length=hop_length, x_axis="time", ax=ax, cmap="coolwarm")
    fig.colorbar(img, ax=ax)
    ax.set_title(f"MFCCs (n_mfcc={n_mfcc})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MFCC Coefficient")
    fig.tight_layout()
    return fig


def plot_chroma(y: np.ndarray, sr: int, hop_length: int = 512) -> matplotlib.figure.Figure:
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

    fig, ax = _new_fig(figsize=(10, 4))
    img = librosa.display.specshow(chroma, sr=sr, hop_length=hop_length, x_axis="time", y_axis="chroma", ax=ax, cmap="viridis")
    fig.colorbar(img, ax=ax)
    ax.set_title("Chroma Features")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch Class")
    fig.tight_layout()
    return fig


def plot_harmonic_percussive(y_harmonic: np.ndarray, y_percussive: np.ndarray, sr: int) -> matplotlib.figure.Figure:
    times = np.linspace(0, len(y_harmonic) / sr, num=len(y_harmonic))

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    axes[0].plot(times, y_harmonic, color="#a78bfa", linewidth=0.5)
    axes[0].set_title("Harmonic Component")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(times, y_percussive, color="#f59e0b", linewidth=0.5)
    axes[1].set_title("Percussive Component")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Time (s)")

    fig.tight_layout()
    return fig


def plot_zcr(y: np.ndarray, sr: int) -> matplotlib.figure.Figure:
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    frames = np.arange(len(zcr))
    times = librosa.frames_to_time(frames, sr=sr)

    fig, ax = _new_fig(figsize=(10, 3))
    ax.plot(times, zcr, color="#34d399", linewidth=1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Zero Crossing Rate")
    ax.set_title("Zero Crossing Rate over Time")
    fig.tight_layout()
    return fig
