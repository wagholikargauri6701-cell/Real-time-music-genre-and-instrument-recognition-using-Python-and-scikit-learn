"""
feature_extraction.py
---------------------
This module handles all audio loading and feature extraction logic.

We use the librosa library to load audio files and extract meaningful
acoustic features from them. These features are what the machine learning
model actually learns from -- not the raw audio waveform itself.

Features extracted:
  - MFCCs (Mel-Frequency Cepstral Coefficients): Capture the timbral texture
    of the audio. Widely used in speech and music recognition tasks.
  - Spectral Centroid: Indicates where the center of mass of the spectrum is.
    Brighter sounds (e.g., cymbals) have a higher spectral centroid.
  - Zero Crossing Rate (ZCR): How often the signal crosses zero. Useful for
    distinguishing percussive vs. tonal sounds.
  - Spectral Rolloff: The frequency below which 85% of the energy is contained.
    Helps distinguish harmonic from noisy content.
  - Chroma Features: Represent the 12 pitch classes (C, C#, D, ...). Useful
    for detecting harmony and chord progressions.
  - RMS Energy: Root Mean Square energy -- a measure of loudness over time.

All features are summarised using their mean across time frames, so that
each audio file is represented as a single fixed-length feature vector.
This is necessary because ML classifiers expect a consistent input shape.
"""

import numpy as np
import librosa


# Duration (in seconds) to load from each audio file.
# Using a fixed duration ensures all feature vectors have the same shape.
AUDIO_DURATION = 30  # seconds


def load_audio(file_path):
    """
    Load an audio file and return the waveform and sample rate.

    Parameters:
        file_path (str): Path to the audio file (MP3, WAV, etc.)

    Returns:
        y (np.ndarray): Audio time series (mono).
        sr (int): Sample rate of the audio.

    We use mono=True to convert stereo to single-channel audio,
    which simplifies the feature extraction step.
    """
    y, sr = librosa.load(file_path, sr=None, mono=True, duration=AUDIO_DURATION)
    return y, sr


def extract_features(file_path):
    """
    Extract a feature vector from a single audio file.

    Parameters:
        file_path (str): Path to the audio file.

    Returns:
        feature_vector (np.ndarray): A 1D array of acoustic features.
        feature_names (list of str): Labels for each feature (for reporting).

    The returned vector is what gets passed to the trained ML model for
    prediction. The order of features here must exactly match the order
    used during training in model_train.py.
    """
    y, sr = load_audio(file_path)

    feature_values = []
    feature_names = []

    # 1. MFCCs (Mel-Frequency Cepstral Coefficients)
    # We extract 13 MFCC coefficients, then take the mean of each
    # across all time frames. This gives us 13 scalar values.
    # MFCCs are the most commonly used features in music/speech ML tasks.
    n_mfcc = 13
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_means = np.mean(mfccs, axis=1)  # shape: (13,)
    feature_values.extend(mfcc_means)
    feature_names.extend([f"mfcc_{i+1}" for i in range(n_mfcc)])

    # 2. Spectral Centroid
    # Mean of the spectral centroid across all frames.
    # Higher values indicate brighter, treble-heavy audio.
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    feature_values.append(np.mean(spectral_centroid))
    feature_names.append("spectral_centroid_mean")

    # 3. Zero Crossing Rate
    # How often the audio signal changes sign (crosses zero).
    # High ZCR is typical of noisy or percussive sounds.
    zcr = librosa.feature.zero_crossing_rate(y)
    feature_values.append(np.mean(zcr))
    feature_names.append("zero_crossing_rate_mean")

    # 4. Spectral Rolloff
    # Frequency below which 85% of the spectral energy is concentrated.
    # Helps differentiate harmonic (tonal) from noisy audio content.
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    feature_values.append(np.mean(rolloff))
    feature_names.append("spectral_rolloff_mean")

    # 5. Chroma Features
    # 12 values representing the energy in each pitch class (C, C#, D, ...).
    # Useful for detecting musical key and harmony.
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_means = np.mean(chroma, axis=1)  # shape: (12,)
    feature_values.extend(chroma_means)
    feature_names.extend([f"chroma_{i+1}" for i in range(12)])

    # 6. RMS Energy
    # Root Mean Square energy measures the average loudness of the signal.
    rms = librosa.feature.rms(y=y)
    feature_values.append(np.mean(rms))
    feature_names.append("rms_energy_mean")

    # Convert the list of scalar values into a numpy array.
    # Final vector length: 13 (MFCCs) + 1 + 1 + 1 + 12 (chroma) + 1 = 29 features
    feature_vector = np.array(feature_values, dtype=np.float32)

    return feature_vector, feature_names
