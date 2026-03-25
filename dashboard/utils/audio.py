"""Core audio processing functions extracted from genres_features.ipynb."""

import os
import numpy as np
import librosa
import streamlit as st


GENRES = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
DEFAULT_TRACK = "00036"


def setup_kaggle_credentials() -> None:
    """Write Kaggle creds from st.secrets to ~/.kaggle/kaggle.json (Streamlit Cloud)."""
    import json
    from pathlib import Path
    if "kaggle" not in st.secrets:
        return
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    creds = {"username": st.secrets["kaggle"]["username"], "key": st.secrets["kaggle"]["key"]}
    cred_file = kaggle_dir / "kaggle.json"
    cred_file.write_text(json.dumps(creds))
    os.chmod(cred_file, 0o600)


@st.cache_resource
def get_data_root() -> str:
    """Return local Data/Data if present, else download via kagglehub."""
    local = "Data/Data"
    if os.path.isdir(local):
        return local
    import kagglehub
    path = kagglehub.dataset_download(
        "andradaolteanu/gtzan-dataset-music-genre-classification"
    )
    return path


def get_gtzan_path(genre: str, track: str = DEFAULT_TRACK) -> str:
    return f"{get_data_root()}/genres_original/{genre}/{genre}.{track}.wav"


@st.cache_data
def load_audio(path: str, sr: int = 22050) -> tuple[np.ndarray, int | float]:
    """Load a WAV file and return (signal, sample_rate)."""
    y, sr_loaded = librosa.load(path, sr=sr)
    return y, sr_loaded


@st.cache_data
def load_audio_bytes(audio_bytes: bytes, sr: int = 22050) -> tuple[np.ndarray, int | float]:
    """Load audio from uploaded bytes."""
    import io
    y, sr_loaded = librosa.load(io.BytesIO(audio_bytes), sr=sr)
    return y, sr_loaded


def compute_stft(y: np.ndarray, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Compute STFT magnitude spectrogram."""
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    return np.abs(D)


def get_waveform_data(y: np.ndarray, sr: int) -> dict:
    """Return waveform metadata."""
    return {
        "signal": y,
        "sr": sr,
        "duration": len(y) / sr,
        "n_samples": len(y),
    }


def extract_zero_crossing_rate(y: np.ndarray) -> tuple[np.ndarray, float]:
    """Return ZCR array and mean ZCR."""
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    return zcr, float(np.mean(zcr))


def extract_harmonic_percussive(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separate harmonic and percussive components."""
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    return y_harmonic, y_percussive


def extract_tempo(y: np.ndarray, sr: int) -> float:
    """Extract tempo in BPM."""
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    arr = np.asarray(tempo).ravel()
    return float(arr[0])


def extract_spectral_centroid(y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """Return spectral centroid array."""
    return librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]


def extract_spectral_rolloff(y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """Return spectral rolloff array."""
    return librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]


def extract_mfcc(y: np.ndarray, sr: int, n_mfcc: int = 20) -> np.ndarray:
    """Return MFCC matrix (n_mfcc x frames)."""
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)


def extract_chroma(y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """Return chroma feature matrix (12 x frames)."""
    return librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
