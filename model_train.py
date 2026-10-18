"""
model_train.py
--------------
This script trains two machine learning classifiers for:
  1. Music Genre Classification
  2. Musical Instrument Detection (treated as a multi-label problem)

Because we do not have a real labelled dataset in this thesis prototype,
we generate a synthetic dataset using numpy's random functions. The feature
vectors are shaped to match exactly what feature_extraction.py produces
(29 features per audio file).

In a real deployment, you would replace the synthetic data section with
actual feature vectors extracted from a labelled audio dataset such as
GTZAN (for genres) or MedleyDB (for instruments).

Classifiers used:
  - Random Forest: An ensemble of decision trees. Robust to overfitting,
    works well on tabular feature data, and gives feature importance scores.
  - Support Vector Machine (SVM): Finds the optimal hyperplane to separate
    classes. Effective in high-dimensional spaces like audio feature vectors.

Both trained models are saved to disk using pickle so that app.py can
load them at runtime without retraining.
"""

import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Output directory where trained model files will be saved.
MODELS_DIR = "models"

# Number of features per sample -- must match feature_extraction.py output.
N_FEATURES = 29

# Music genres we want to classify.
GENRE_LABELS = ["blues", "classical", "country", "disco", "hiphop",
                "jazz", "metal", "pop", "reggae", "rock"]

# Instruments we want to detect. Each track can have multiple instruments,
# so this is a multi-label classification problem.
INSTRUMENT_LABELS = ["guitar", "piano", "drums", "bass", "violin",
                     "trumpet", "saxophone", "synthesizer"]

# Probability threshold used at inference time to decide "instrument is present".
# A value of 0.35 means: if the SVM gives >= 35% probability of presence,
# we call it detected. This is lower than the default 0.5 because with
# synthetic training data the probability scores are compressed toward the
# centre, so a 0.5 threshold would almost never fire.
# BUG FIX #4: was using hard svm.predict() which always defaulted to 0
# when the SVM learned nothing from pure-noise data.
INSTRUMENT_DETECTION_THRESHOLD = 0.35


def generate_synthetic_genre_data(n_samples=500):
    """
    Generate a synthetic dataset for genre classification.

    Each genre gets an equal share of samples. We add a class-specific
    mean offset to the feature values so the classifier has something
    meaningful to learn from (otherwise all classes would be identical noise).

    Parameters:
        n_samples (int): Total number of synthetic audio samples to create.

    Returns:
        X (np.ndarray): Feature matrix of shape (n_samples, N_FEATURES).
        y (np.ndarray): Label array of shape (n_samples,) with genre indices.
    """
    np.random.seed(42)  # Fix seed for reproducibility.
    samples_per_genre = n_samples // len(GENRE_LABELS)
    X_list = []
    y_list = []

    for genre_index, genre_name in enumerate(GENRE_LABELS):
        # Each genre gets a unique mean offset so classes are linearly separable.
        # In a real dataset this separation comes from actual audio content.
        class_mean_offset = genre_index * 0.5
        features = np.random.randn(samples_per_genre, N_FEATURES) + class_mean_offset
        labels = np.full(samples_per_genre, genre_index)

        X_list.append(features)
        y_list.append(labels)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y


def generate_synthetic_instrument_data(n_samples=1000):
    """
    Generate a synthetic dataset for instrument detection.

    FIX for BUG 1, 2, 4:
    The original version generated completely random X and completely random y
    with no relationship between them. An SVM trained on that data cannot learn
    any decision boundary and will always predict the majority class (0 = absent),
    causing instruments to never be detected.

    The fix: give each instrument a distinct feature signature (a unique mean
    offset vector). Samples labelled as "instrument present" are drawn from a
    shifted Gaussian centred on that instrument's signature. Samples labelled
    "absent" are drawn from a zero-centred Gaussian. This creates a learnable
    boundary for each instrument's SVM.

    Parameters:
        n_samples (int): Total number of synthetic audio samples per instrument.
                         Increased from 500 to 1000 for more stable SVM training.

    Returns:
        X (np.ndarray): Feature matrix of shape (n_samples * n_instruments, N_FEATURES).
        y (np.ndarray): Binary label matrix of shape (n_samples * n_instruments, n_instruments).
    """
    np.random.seed(0)  # Separate seed from genre data for independence.

    # Each instrument gets a unique mean shift vector.
    # We use a different random seed per instrument so the signatures are distinct.
    instrument_mean_shifts = []
    for i in range(len(INSTRUMENT_LABELS)):
        rng = np.random.RandomState(seed=i * 7)  # Deterministic, unique per instrument.
        shift = rng.randn(N_FEATURES) * 1.5      # Magnitude 1.5 gives good class separation.
        instrument_mean_shifts.append(shift)

    X_list = []
    y_list = []

    for i in range(len(INSTRUMENT_LABELS)):
        # "Present" samples: drawn from a distribution shifted toward this instrument's signature.
        n_positive = n_samples // 2
        X_positive = np.random.randn(n_positive, N_FEATURES) + instrument_mean_shifts[i]

        # "Absent" samples: drawn from a zero-centred distribution (generic background).
        n_negative = n_samples - n_positive
        X_negative = np.random.randn(n_negative, N_FEATURES)

        X_instrument = np.vstack([X_positive, X_negative])

        # Build the full multi-label row for each sample.
        # For simplicity, when we generate data for instrument i:
        #   - positive samples have label 1 for instrument i, 0 for all others.
        #   - negative samples have label 0 for all instruments.
        y_instrument = np.zeros((n_samples, len(INSTRUMENT_LABELS)), dtype=int)
        y_instrument[:n_positive, i] = 1  # Mark instrument i as present.

        X_list.append(X_instrument)
        y_list.append(y_instrument)

    X = np.vstack(X_list)
    y = np.vstack(y_list)
    return X, y


def train_genre_classifier(X_train, y_train):
    """
    Train a Random Forest classifier for genre classification.

    Random Forests are a good baseline for this kind of task because:
      - They handle non-linear feature relationships well.
      - They are relatively robust to the scale of input features.
      - They provide feature importance scores (useful for thesis analysis).

    Parameters:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Training genre labels.

    Returns:
        clf (RandomForestClassifier): Trained genre classifier.
    """
    clf = RandomForestClassifier(
        n_estimators=100,   # Number of decision trees in the forest.
        max_depth=10,       # Limit tree depth to prevent overfitting.
        random_state=42     # For reproducibility across runs.
    )
    clf.fit(X_train, y_train)
    return clf


def train_instrument_detector(X_train, y_train):
    """
    Train one SVM classifier per instrument (One-vs-Rest strategy).

    Because each instrument is an independent binary classification problem
    (present or not), we train a separate SVM for each instrument and store
    them all in a list. At prediction time, we run each SVM and combine results.

    SVM with an RBF kernel is effective for binary classification on
    normalised feature vectors.

    Parameters:
        X_train (np.ndarray): Training feature matrix.
        y_train (np.ndarray): Binary label matrix (n_samples, n_instruments).

    Returns:
        instrument_classifiers (list of SVC): One trained SVM per instrument.
    """
    instrument_classifiers = []

    for i, instrument_name in enumerate(INSTRUMENT_LABELS):
        # CalibratedClassifierCV wraps SVC to provide predict_proba().
        # This is the recommended approach in scikit-learn >= 1.9,
        # which deprecated the probability=True flag on SVC directly.
        base_svm = SVC(kernel="rbf", random_state=42)
        svm = CalibratedClassifierCV(base_svm, ensemble=False)
        svm.fit(X_train, y_train[:, i])  # Train on one instrument's binary labels.
        instrument_classifiers.append(svm)
        print(f"  Trained SVM for: {instrument_name}")

    return instrument_classifiers


def save_models(genre_clf, instrument_clfs, genre_scaler, instrument_scaler):
    """
    Save all trained models and both scalers to disk using pickle.

    FIX for BUG 3: We now save two separate scalers -- one fitted on genre
    training data and one fitted on instrument training data. Previously,
    the instrument scaler was fitted on genre data, which is conceptually
    wrong and causes the scaler statistics to be mismatched at inference time.

    Parameters:
        genre_clf: Trained genre classifier.
        instrument_clfs: List of trained instrument SVM classifiers.
        genre_scaler: StandardScaler fitted on genre training data.
        instrument_scaler: StandardScaler fitted on instrument training data.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    with open(os.path.join(MODELS_DIR, "genre_classifier.pkl"), "wb") as f:
        pickle.dump(genre_clf, f)

    with open(os.path.join(MODELS_DIR, "instrument_classifiers.pkl"), "wb") as f:
        pickle.dump(instrument_clfs, f)

    # Save genre scaler (used by genre classifier at inference).
    with open(os.path.join(MODELS_DIR, "genre_scaler.pkl"), "wb") as f:
        pickle.dump(genre_scaler, f)

    # Save instrument scaler (used by instrument SVMs at inference).
    # BUG FIX #3: This scaler is now fitted on instrument data, not genre data.
    with open(os.path.join(MODELS_DIR, "instrument_scaler.pkl"), "wb") as f:
        pickle.dump(instrument_scaler, f)

    # Keep the old "scaler.pkl" name as well for backwards compatibility
    # in case any other script references it. It points to the genre scaler.
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(genre_scaler, f)

    print(f"\nAll models saved to the '{MODELS_DIR}/' directory.")


def main():
    print("=== Music Genre Classification -- Model Training ===\n")

    # Step 1: Generate synthetic training data.
    print("Generating synthetic genre dataset...")
    X_genre, y_genre = generate_synthetic_genre_data(n_samples=500)

    print("Generating synthetic instrument dataset...")
    X_inst, y_inst = generate_synthetic_instrument_data(n_samples=1000)

    # Step 2: Scale genre features.
    # StandardScaler transforms each feature to have mean=0 and std=1.
    # This is important for SVM, which is sensitive to feature scale.
    print("\nFitting genre StandardScaler...")
    genre_scaler = StandardScaler()
    X_genre_scaled = genre_scaler.fit_transform(X_genre)

    # BUG FIX #3: Fit a SEPARATE scaler on instrument training data.
    # The original code applied the genre scaler to instrument data, which is
    # wrong because the two datasets have different statistical distributions.
    print("Fitting instrument StandardScaler (separate from genre scaler)...")
    instrument_scaler = StandardScaler()
    X_inst_scaled = instrument_scaler.fit_transform(X_inst)

    # Step 3: Split data into training and test sets (80/20 split).
    X_genre_train, X_genre_test, y_genre_train, y_genre_test = train_test_split(
        X_genre_scaled, y_genre, test_size=0.2, random_state=42
    )

    X_inst_train, X_inst_test, y_inst_train, y_inst_test = train_test_split(
        X_inst_scaled, y_inst, test_size=0.2, random_state=42
    )

    # Step 4: Train the genre classifier.
    print("\nTraining Random Forest genre classifier...")
    genre_clf = train_genre_classifier(X_genre_train, y_genre_train)

    y_genre_pred = genre_clf.predict(X_genre_test)
    print("\nGenre Classifier -- Test Set Performance:")
    print(classification_report(y_genre_test, y_genre_pred, target_names=GENRE_LABELS))

    # Step 5: Train the instrument detectors.
    print("Training SVM instrument detectors (one per instrument)...")
    instrument_clfs = train_instrument_detector(X_inst_train, y_inst_train)

    # Quick sanity check: verify each SVM can detect its instrument on test data.
    print("\nInstrument Detector -- Per-Instrument Test Accuracy:")
    for i, instrument_name in enumerate(INSTRUMENT_LABELS):
        svm = instrument_clfs[i]
        y_pred = svm.predict(X_inst_test)
        accuracy = (y_pred == y_inst_test[:, i]).mean() * 100
        print(f"  {instrument_name:<15} accuracy: {accuracy:.1f}%")

    # Step 6: Save all models and both scalers to disk.
    save_models(genre_clf, instrument_clfs, genre_scaler, instrument_scaler)
    print("\nTraining complete. Run app.py to start the dashboard.")


if __name__ == "__main__":
    main()
