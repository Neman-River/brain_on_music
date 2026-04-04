# Brain on Music

**What does sound actually do to us — and can we see it in the signal?**

---

## Live App

**[brainonmusic.streamlit.app](https://brainonmusic-jfx9ycnyn98447vn87nsyf.streamlit.app)**

This app is continuously evolving. New pages, features, and analysis are added regularly as the exploration deepens.

---

## About

I've been playing piano most of my life. At some point the question stopped being *what* to play and started being *why* certain music feels the way it does — why a minor second creates tension, why a slow vibrato in a cello line can make a room go quiet, why rhythm changes energy in a room before anyone consciously registers it.

That curiosity eventually led me here: audio signal processing as a way to bridge music theory and neuroscience. Not as an engineer building a product, but as a musician trying to understand what's actually happening in the sound — and maybe, eventually, in the brain listening to it.

This project is not finished. It's an ongoing exploration, built in public.

---

## What's Inside

The dashboard has five sections, each tackling a different layer of the question:

| Page | What it does |
|------|-------------|
| **Explore Audio** | Load any track (GTZAN sample or your own file) and get an instant signal portrait — waveform, tempo, timbre, MFCCs |
| **Sound Basics** | Interactive sine wave demo, the signal processing pipeline explained visually, feature reference |
| **Fourier Transform** | Hands-on STFT explorer — adjust `n_fft` and `hop_length` and watch the spectrogram change in real time |
| **Spectrograms** | Linear, log, and mel-scale views side by side; tweak colormap and see what the same sound looks like through different lenses |
| **Audio Features** | Deep dives into 7 core features (ZCR, HPSS, tempo, spectral centroid, rolloff, MFCCs, chroma) with theory + genre comparison |
| **Genre Comparison** | Full EDA on the GTZAN dataset — boxplots, pairplots, radar charts, parallel coordinates, correlation heatmap |

---

## Dataset

[GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) — 1,000 audio tracks across 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock), each 30 seconds at 22,050 Hz. The standard benchmark for music genre classification research.

---

## Tech Stack

- **[librosa](https://librosa.org)** — audio analysis and feature extraction
- **[Streamlit](https://streamlit.io)** — interactive app framework
- **[Plotly](https://plotly.com/python/)** — interactive visualizations
- **[scikit-learn](https://scikit-learn.org)** — preprocessing and (eventually) classification
- **NumPy / Pandas** — data handling
- **Python 3.12**, managed with **[uv](https://docs.astral.sh/uv/)**

---

## Run Locally

```bash
git clone https://github.com/natallialantukh/music_classification
cd music_classification

# Install dependencies
uv sync

# Set up Kaggle credentials (needed to download dataset)
# Place your kaggle.json at ~/.kaggle/kaggle.json

# Launch the dashboard
uv run streamlit run dashboard/Explore_Audio.py

# Or open the EDA notebook
uv run jupyter notebook genres_features.ipynb
```

The dataset downloads automatically on first run if Kaggle credentials are configured.

**Dataset structure after download:**
```
Data/
├── features_30_sec.csv     # Pre-extracted features
├── features_3_sec.csv
├── genres_original/        # Raw WAV files by genre
└── images_original/
```

---

## What's Next

This project is continuously growing. Things currently in progress or planned:

- **Genre classifier** — train a model on the extracted features and let you predict genre from any uploaded file
- **Mood mapping** — explore correlations between audio features and psychological valence/arousal models
- **Cross-genre feature fingerprints** — what makes jazz sound like jazz at the signal level?
- **Your own audio** — deeper support for uploading and analyzing personal recordings
- **Neuroscience layer** — connect signal features to what's known about auditory cortex response and emotional processing

If you're a musician, a DSP learner, or someone curious about how sound works on the brain — follow along.

---

*Built by a pianist asking too many questions.*
