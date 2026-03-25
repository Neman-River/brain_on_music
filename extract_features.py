"""
Extract audio features from GTZAN WAV files and save to CSV.
Produces Data/my_features.csv with the same schema as features_30_sec.csv.
"""

import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 20
DATA_ROOT = "Data/Data/genres_original"
OUTPUT_PATH = "Data/my_features.csv"


def extract_features(y: np.ndarray, sr: int) -> dict:
    """Extract all features from a single audio track."""
    features = {}

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features["chroma_stft_mean"] = float(np.mean(chroma))
    features["chroma_stft_var"] = float(np.var(chroma))

    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    features["rms_mean"] = float(np.mean(rms))
    features["rms_var"] = float(np.var(rms))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_centroid_var"] = float(np.var(centroid))

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
    features["spectral_bandwidth_var"] = float(np.var(bandwidth))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features["rolloff_mean"] = float(np.mean(rolloff))
    features["rolloff_var"] = float(np.var(rolloff))

    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    features["zero_crossing_rate_mean"] = float(np.mean(zcr))
    features["zero_crossing_rate_var"] = float(np.var(zcr))

    harmony, perceptr = librosa.effects.hpss(y)
    features["harmony_mean"] = float(np.mean(harmony))
    features["harmony_var"] = float(np.var(harmony))
    features["perceptr_mean"] = float(np.mean(perceptr))
    features["perceptr_var"] = float(np.var(perceptr))

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    features["tempo"] = float(np.atleast_1d(tempo)[0])

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    for i in range(N_MFCC):
        features[f"mfcc{i+1}_mean"] = float(np.mean(mfccs[i]))
        features[f"mfcc{i+1}_var"] = float(np.var(mfccs[i]))

    return features


def process_dataset(data_root: str) -> pd.DataFrame:
    """Walk all genre folders and extract features from each WAV file."""
    rows = []
    genres = sorted(
        d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))
    )

    files = []
    for genre in genres:
        genre_dir = os.path.join(data_root, genre)
        for fname in sorted(os.listdir(genre_dir)):
            if fname.endswith(".wav"):
                files.append((fname, genre, os.path.join(genre_dir, fname)))

    for fname, genre, fpath in tqdm(files, desc="Extracting features"):
        try:
            y, sr = librosa.load(fpath, mono=True)
            row = {"filename": fname, "label": genre, "length": len(y) / sr}
            row.update(extract_features(y, sr))
            rows.append(row)
        except Exception as e:
            print(f"\nSkipped {fpath}: {e}")

    return pd.DataFrame(rows)


def main():
    print(f"Source: {DATA_ROOT}")
    print(f"Output: {OUTPUT_PATH}\n")

    df = process_dataset(DATA_ROOT)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print(f"\nLabel counts:\n{df['label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
