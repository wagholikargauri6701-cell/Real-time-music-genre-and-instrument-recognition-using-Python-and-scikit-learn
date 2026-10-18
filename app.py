"""
app.py
------
Main Streamlit web application for Music Genre Classification
and Instrumentation Detection.

This file ties together all components of the project:
  - The user uploads an MP3 file via the browser.
  - We extract audio features using feature_extraction.py.
  - We run the trained ML models (loaded from the models/ folder).
  - We display the predicted genre and detected instruments on a dashboard.
  - The user can download a CSV or TXT report of the results.

To run this application:
    streamlit run app.py

Make sure you have run model_train.py at least once before launching the app,
so that the trained model files exist in the models/ directory.
"""

import os
import pickle
import tempfile
import datetime
import numpy as np
import pandas as pd
import streamlit as st

from feature_extraction import extract_features
from model_train import GENRE_LABELS, INSTRUMENT_LABELS


# ─── Page Configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="centered"
)

# Keep custom styling compatible with Streamlit's active light or dark theme.
st.markdown("""
    <style>
        .stButton > button {
            background-color: #1DB954;
            color: white;
            border-radius: 8px;
            padding: 0.5em 2em;
            font-size: 1rem;
        }
        .result-box {
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            border-radius: 10px;
            padding: 20px;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)


# ─── Helper: Load Trained Models ──────────────────────────────────────────────

@st.cache_resource
def load_models():
    """
    Load the trained genre classifier, instrument detectors, and scaler
    from disk. We use @st.cache_resource so models are only loaded once
    per session, not on every user interaction.

    Returns:
        genre_clf: Trained Random Forest genre classifier.
        instrument_clfs: List of trained SVM instrument detectors.
        scaler: Fitted StandardScaler used during training.

    Raises:
        FileNotFoundError if the model files do not exist yet.
    """
    models_dir = "models"

    genre_clf_path = os.path.join(models_dir, "genre_classifier.pkl")
    instrument_clf_path = os.path.join(models_dir, "instrument_classifiers.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    # Check that all model files exist before trying to load them.
    if not all(os.path.exists(p) for p in [genre_clf_path, instrument_clf_path, scaler_path]):
        return None, None, None

    with open(genre_clf_path, "rb") as f:
        genre_clf = pickle.load(f)

    with open(instrument_clf_path, "rb") as f:
        instrument_clfs = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return genre_clf, instrument_clfs, scaler


# ─── Helper: Run Predictions ──────────────────────────────────────────────────

def predict_genre(feature_vector, scaler, genre_clf):
    """
    Predict the music genre from a feature vector.

    We scale the features first using the same scaler that was used
    during training. This is critical -- without scaling, the model
    would receive inputs in a different distribution than it was trained on.

    Parameters:
        feature_vector (np.ndarray): Raw feature vector (29 values).
        scaler: Fitted StandardScaler.
        genre_clf: Trained genre classifier.

    Returns:
        genre_name (str): The predicted genre label.
        confidence (float): Confidence score as a percentage.
    """
    # Reshape to (1, n_features) because sklearn expects a 2D array.
    X = feature_vector.reshape(1, -1)
    X_scaled = scaler.transform(X)

    genre_index = genre_clf.predict(X_scaled)[0]
    genre_name = GENRE_LABELS[genre_index]

    # Get class probabilities to show a confidence score.
    probabilities = genre_clf.predict_proba(X_scaled)[0]
    confidence = round(probabilities[genre_index] * 100, 2)

    return genre_name, confidence


def predict_instruments(feature_vector, scaler, instrument_clfs):
    """
    Detect which instruments are present in the audio.

    Each SVM in instrument_clfs was trained to detect one instrument.
    We run all of them and collect those that predict "present" (label=1).

    Parameters:
        feature_vector (np.ndarray): Raw feature vector (29 values).
        scaler: Fitted StandardScaler.
        instrument_clfs: List of trained SVM classifiers (one per instrument).

    Returns:
        detected_instruments (list of str): Names of detected instruments.
        instrument_confidences (dict): {instrument_name: confidence_percent}.
    """
    X = feature_vector.reshape(1, -1)
    X_scaled = scaler.transform(X)

    detected_instruments = []
    instrument_confidences = {}

    for i, svm in enumerate(instrument_clfs):
        instrument_name = INSTRUMENT_LABELS[i]
        prediction = svm.predict(X_scaled)[0]
        confidence = svm.predict_proba(X_scaled)[0][1] * 100  # Probability of "present"

        instrument_confidences[instrument_name] = round(confidence, 2)

        if prediction == 1:
            detected_instruments.append(instrument_name)

    # If no instruments are detected (unlikely but possible), show a fallback.
    if not detected_instruments:
        detected_instruments = ["No instruments detected with high confidence"]

    return detected_instruments, instrument_confidences


# ─── Helper: Generate Downloadable Report ─────────────────────────────────────

def generate_txt_report(filename, genre, confidence, instruments, instrument_confidences, feature_names, feature_vector):
    """
    Generate a plain-text summary report of the classification results.

    Parameters:
        filename (str): Name of the uploaded audio file.
        genre (str): Predicted genre.
        confidence (float): Genre prediction confidence.
        instruments (list): Detected instrument names.
        instrument_confidences (dict): Per-instrument confidence scores.
        feature_names (list): Feature labels.
        feature_vector (np.ndarray): Extracted feature values.

    Returns:
        report_text (str): Formatted text report.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 55,
        "  Music Genre Classification Report",
        "  Bachelor Thesis Project -- Audio ML System",
        "=" * 55,
        f"Generated   : {timestamp}",
        f"Audio File  : {filename}",
        "",
        "--- Genre Classification ---",
        f"Predicted Genre : {genre.upper()}",
        f"Confidence      : {confidence}%",
        "",
        "--- Instrument Detection ---",
        f"Detected Instruments: {', '.join(instruments)}",
        "",
        "Instrument Confidence Scores:",
    ]
    for inst, conf in instrument_confidences.items():
        lines.append(f"  {inst:<15} {conf:.2f}%")

    lines += [
        "",
        "--- Extracted Audio Features ---",
    ]
    for name, value in zip(feature_names, feature_vector):
        lines.append(f"  {name:<35} {value:.6f}")

    lines.append("=" * 55)
    return "\n".join(lines)


def generate_csv_report(filename, genre, confidence, instruments, instrument_confidences, feature_names, feature_vector):
    """
    Generate a CSV report of the classification results using pandas.

    Parameters:
        (same as generate_txt_report)

    Returns:
        csv_string (str): CSV content as a string.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build a summary dataframe for the main results.
    summary_data = {
        "Field": ["Timestamp", "File Name", "Predicted Genre", "Genre Confidence (%)"],
        "Value": [timestamp, filename, genre.upper(), confidence]
    }
    summary_df = pd.DataFrame(summary_data)

    # Build a dataframe for instrument detection results.
    inst_data = {
        "Instrument": list(instrument_confidences.keys()),
        "Confidence (%)": list(instrument_confidences.values()),
        "Detected": ["Yes" if inst in instruments else "No"
                     for inst in instrument_confidences.keys()]
    }
    inst_df = pd.DataFrame(inst_data)

    # Build a dataframe for extracted features.
    feature_df = pd.DataFrame({
        "Feature Name": feature_names,
        "Feature Value": feature_vector
    })

    # Combine everything into a single CSV string with section headers.
    csv_parts = [
        "SUMMARY",
        summary_df.to_csv(index=False),
        "\nINSTRUMENT DETECTION",
        inst_df.to_csv(index=False),
        "\nEXTRACTED FEATURES",
        feature_df.to_csv(index=False)
    ]
    return "\n".join(csv_parts)


# ─── Main Dashboard UI ────────────────────────────────────────────────────────

def main():
    # Page title and description.
    st.title("Music Genre Classifier")
    st.markdown(
        "Upload an MP3 audio file to classify its **genre** and detect **instruments** "
        "using machine learning. This is a prototype built as part of a Bachelor's thesis project."
    )
    st.markdown("---")

    # Load models at startup. Show a warning if they are not found.
    genre_clf, instrument_clfs, scaler = load_models()

    if genre_clf is None:
        st.error(
            "Trained models not found. Please run `python model_train.py` first "
            "to generate the model files in the `models/` directory."
        )
        st.stop()

    # File uploader widget. Accepts MP3 and WAV files.
    st.subheader("Step 1: Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an MP3 or WAV file",
        type=["mp3", "wav"],
        help="Upload a music file to analyse. Only the first 30 seconds will be processed."
    )

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/mp3")
        st.success(f"File uploaded: **{uploaded_file.name}**")

        st.subheader("Step 2: Run Classification")
        run_button = st.button("Analyse Audio")

        if run_button:
            # Save the uploaded file to a temporary location on disk.
            # librosa requires a file path, not an in-memory buffer.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            try:
                # Extract audio features from the uploaded file.
                with st.spinner("Extracting audio features..."):
                    feature_vector, feature_names = extract_features(tmp_file_path)

                # Run genre classification.
                with st.spinner("Running genre classification..."):
                    predicted_genre, genre_confidence = predict_genre(
                        feature_vector, scaler, genre_clf
                    )

                # Run instrument detection.
                with st.spinner("Detecting instruments..."):
                    detected_instruments, instrument_confidences = predict_instruments(
                        feature_vector, scaler, instrument_clfs
                    )

                # Clean up the temporary file.
                os.unlink(tmp_file_path)

                # ─── Display Results ───────────────────────────────────────
                st.markdown("---")
                st.subheader("Step 3: Classification Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Predicted Genre**")
                    st.markdown(
                        f"<div class='result-box'><h2>"
                        f"{predicted_genre.upper()}</h2>"
                        f"<p>Confidence: {genre_confidence}%</p></div>",
                        unsafe_allow_html=True
                    )

                with col2:
                    st.markdown("**Detected Instruments**")
                    instrument_list_html = "".join(
                        [f"<li>{inst}</li>" for inst in detected_instruments]
                    )
                    st.markdown(
                        f"<div class='result-box'><ul>{instrument_list_html}</ul></div>",
                        unsafe_allow_html=True
                    )

                # Show per-instrument confidence scores as a bar chart.
                st.markdown("**Instrument Detection Confidence Scores**")
                conf_df = pd.DataFrame({
                    "Instrument": list(instrument_confidences.keys()),
                    "Confidence (%)": list(instrument_confidences.values())
                }).set_index("Instrument")
                st.bar_chart(conf_df)

                # Show extracted features in a collapsible section.
                with st.expander("View Extracted Audio Features"):
                    feature_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Value": [round(float(v), 6) for v in feature_vector]
                    })
                    st.dataframe(feature_df, use_container_width=True)

                # ─── Download Report Section ───────────────────────────────
                st.markdown("---")
                st.subheader("Step 4: Download Report")

                report_col1, report_col2 = st.columns(2)

                txt_report = generate_txt_report(
                    uploaded_file.name, predicted_genre, genre_confidence,
                    detected_instruments, instrument_confidences,
                    feature_names, feature_vector
                )

                csv_report = generate_csv_report(
                    uploaded_file.name, predicted_genre, genre_confidence,
                    detected_instruments, instrument_confidences,
                    feature_names, feature_vector
                )

                with report_col1:
                    st.download_button(
                        label="Download TXT Report",
                        data=txt_report,
                        file_name="classification_report.txt",
                        mime="text/plain"
                    )

                with report_col2:
                    st.download_button(
                        label="Download CSV Report",
                        data=csv_report,
                        file_name="classification_report.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                # Basic error handling -- print the error and show it to the user.
                # In a thesis prototype, simple error messages are appropriate.
                print(f"Error during processing: {e}")
                st.error(f"An error occurred during audio processing: {e}")

    else:
        st.info("Please upload an audio file to get started.")


if __name__ == "__main__":
    main()
