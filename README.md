# Music Genre Classification and Instrumentation Detection

**Bachelor Thesis Project** - Gauri Nitin Wagholikar
Department of Computer Science
Academic Year 2025/2026

---

## Project Overview

This project implements a machine learning pipeline for automatic music genre
classification and musical instrument detection from audio files.

A user uploads an MP3 file through a web dashboard. The system extracts acoustic
features from the audio using the `librosa` library and passes them through
trained scikit-learn classifiers to predict the music genre and identify which
instruments are present. Results are displayed on an interactive dashboard and
can be exported as a TXT or CSV report.

---

## Project Structure

```
music_genre_classifier/
    app.py                  # Streamlit web dashboard (main entry point)
    feature_extraction.py   # Audio loading and feature extraction (librosa)
    model_train.py          # Model training script (Random Forest + SVM)
    requirements.txt        # Python dependencies
    README.md               # This file
    models/                 # Saved model files (created after training)
        genre_classifier.pkl
        instrument_classifiers.pkl
        scaler.pkl
```

---

## Setup Instructions

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `librosa` requires `ffmpeg` to load MP3 files. Install it via:
> - **Ubuntu/Debian:** `sudo apt install ffmpeg`
> - **macOS (Homebrew):** `brew install ffmpeg`
> - **Windows:** Download from https://ffmpeg.org/download.html

### 3. Train the models

Run the training script once to generate the model files:

```bash
python model_train.py
```

This creates the `models/` directory with three pickle files:
- `genre_classifier.pkl` -- Random Forest trained for genre classification
- `instrument_classifiers.pkl` -- SVM ensemble for instrument detection
- `scaler.pkl` -- StandardScaler fitted on training data

### 4. Launch the dashboard

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`. Upload an MP3 file and click
**Analyse Audio** to see the results.

---

## Features Extracted

| Feature               | Description                                          | Count |
|-----------------------|------------------------------------------------------|-------|
| MFCCs                 | Mel-Frequency Cepstral Coefficients (timbral texture)| 13    |
| Spectral Centroid     | Brightness / center of mass of the frequency spectrum| 1     |
| Zero Crossing Rate    | Signal noisiness / percussiveness                    | 1     |
| Spectral Rolloff      | Frequency containing 85% of spectral energy          | 1     |
| Chroma Features       | Energy across 12 musical pitch classes               | 12    |
| RMS Energy            | Average loudness of the audio signal                 | 1     |
| **Total**             |                                                      | **29**|

---

## Machine Learning Models

| Task                     | Model               | Rationale                                      |
|--------------------------|---------------------|------------------------------------------------|
| Genre Classification     | Random Forest       | Robust ensemble; handles non-linear features   |
| Instrument Detection     | SVM (RBF kernel)    | Effective binary classifier; one per instrument|

---

## Supported Genres

blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

## Detectable Instruments

guitar, piano, drums, bass, violin, trumpet, saxophone, synthesizer

---

## Note on the Training Data

This prototype uses **synthetically generated** feature vectors for training,
as obtaining a large labelled audio dataset was outside the scope of this thesis.

For a production-ready system, replace the synthetic data in `model_train.py`
with real feature vectors extracted from a labelled dataset such as:
- **GTZAN Genre Dataset** (genres): http://marsyas.info/downloads/datasets.html
- **MedleyDB** (instruments): https://medleydb.weebly.com/

---

## How to Run on Real Audio Data (Extension)

To train on real data, modify `model_train.py` to:

1. Point to a directory of labelled audio files.
2. Call `extract_features(file_path)` for each file.
3. Build `X` (feature matrix) and `y` (label array) from those results.
4. Feed them into the existing `train_genre_classifier()` function.

---

## Dependencies

- Python 3.9+
- numpy, pandas, scikit-learn, librosa, streamlit, soundfile

---

*This project was developed as part of a Bachelor's thesis in Computer Science from Gauri Nitin Wagholikar*
