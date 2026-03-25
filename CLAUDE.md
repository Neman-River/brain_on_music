# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Jupyter notebook-based EDA project for music genre classification using the GTZAN dataset. The project explores audio signal processing concepts and feature extraction to distinguish music genres.

## Environment Setup

This project uses `uv` for package management:

```bash
uv add librosa pandas numpy matplotlib seaborn scikit-learn kagglehub ipython jupyter
uv run jupyter notebook
```

**Kaggle credentials** must be configured at `~/.kaggle/kaggle.json` before downloading the dataset.

## Running the Notebook

```bash
uv run jupyter notebook genres_features.ipynb
# or
uv run jupyter lab
```

## Architecture

Single-notebook project — all logic lives in `genres_features.ipynb` (57 cells).

**Data flow:**
1. Download GTZAN dataset via `kagglehub` (andradaolteanu/gtzan-dataset-music-genre-classification)
2. Load WAV files with `librosa` (10 genres, 30-sec clips at 22,050 Hz)
3. Extract audio features using STFT parameters: `n_fft=2048`, `hop_length=512`
4. Visualize features across genres

**Dataset structure after download:**
```
Data/
├── features_30_sec.csv     # Pre-extracted features
├── features_3_sec.csv
├── genres_original/        # Raw WAV files by genre
└── images_original/
```

**Seven core audio features extracted:**
- Zero Crossing Rate
- Harmonic & Percussive separation (`librosa.effects.hpss`)
- Tempo/BPM (`librosa.beat.beat_track`)
- Spectral Centroid
- Spectral Rolloff
- MFCCs (Mel-Frequency Cepstral Coefficients)
- Chroma Features

**Key normalization utilities used:**
- `librosa.amplitude_to_db()` — convert amplitude to decibels
- `sklearn.preprocessing.minmax_scale()` — normalize to [0,1]
- `sklearn.preprocessing.scale()` — standardize features

## Current State

The notebook is in the EDA phase. Cells 1–55 cover data loading, STFT theory, spectrogram visualization, and individual feature exploration. Cell 56 begins aggregated EDA (in progress). The natural next step is loading `features_30_sec.csv` and building a genre classification model.
