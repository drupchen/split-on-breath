import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import re
import os


def parse_srt_time(time_str):
    """Converts '00:00:20,500' to seconds (float)."""
    hours, minutes, seconds = time_str.split(':')
    seconds, millis = seconds.split(',')
    return (int(hours) * 3600) + (int(minutes) * 60) + int(seconds) + (int(millis) / 1000)


def load_srt_labels(srt_path):
    """Parses SRT file and returns a DataFrame with 'start' and 'end' columns."""
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to find timestamps: "00:00:00,000 --> 00:00:00,000"
    # This pattern captures the start group and end group
    pattern = re.compile(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})')

    matches = pattern.findall(content)

    data = []
    for start_str, end_str in matches:
        start = parse_srt_time(start_str)
        end = parse_srt_time(end_str)
        data.append({'start': start, 'end': end})

    print(f"   -> Parsed {len(data)} breath segments from SRT.")
    return pd.DataFrame(data)


def extract_features(y, sr):
    """
    Extracts lightweight features:
    1. MFCCs (Texture/Timbre)
    2. Spectral Centroid (Brightness)
    3. Zero Crossing Rate (Noisiness)
    4. RMS Energy (Loudness)
    """
    try:
        if len(y) < 512: return None  # Skip tiny fragments

        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
        return np.hstack([mfcc, cent, zcr, rms])
    except Exception as e:
        return None


def train_model():
    print(f"--- Step 1: Loading Audio ({AUDIO_FILE}) ---")
    # Load at 16kHz (sufficient for voice/breath)
    y, sr = librosa.load(AUDIO_FILE, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"Audio Duration: {duration / 60:.2f} minutes")

    print(f"--- Step 2: Loading SRT Labels ({LABELS_FILE}) ---")
    labels = load_srt_labels(LABELS_FILE)

    X = []
    y_labels = []

    # Parameters for scanning
    WINDOW_LENGTH = 0.5  # Analysis window size in seconds
    STEP_SIZE = 0.25  # Overlap

    print("--- Step 3: Extracting Features (This takes 1-3 mins) ---")

    # A. Extract POSITIVE samples (The Breaths from SRT)
    print("   -> Extracting Breath samples...")
    breath_mask = np.zeros(len(y), dtype=bool)

    for _, row in labels.iterrows():
        start_sample = int(row['start'] * sr)
        end_sample = int(row['end'] * sr)

        # Safety check for timestamps going beyond audio length
        if start_sample >= len(y): continue
        end_sample = min(end_sample, len(y))

        # Mark this region in the mask so we don't use it for negatives later
        breath_mask[start_sample:end_sample] = True

        chunk = y[start_sample:end_sample]

        # If breath is super short, just take the whole thing
        if len(chunk) < int(WINDOW_LENGTH * sr):
            feats = extract_features(chunk, sr)
            if feats is not None:
                X.append(feats)
                y_labels.append(1)
        else:
            # Slide a window over larger breaths
            for i in range(0, len(chunk) - int(WINDOW_LENGTH * sr) + 1, int(STEP_SIZE * sr)):
                window = chunk[i: i + int(WINDOW_LENGTH * sr)]
                feats = extract_features(window, sr)
                if feats is not None:
                    X.append(feats)
                    y_labels.append(1)

                    # B. Extract NEGATIVE samples (The Speech/Silence)
    print("   -> Extracting Non-Breath samples...")

    # Get indices where breath_mask is False (speech/silence)
    # We create chunks from the FALSE areas

    # Strategy: Just pick random valid start points where mask is False
    num_breath_samples = len(y_labels)
    # Get more negatives to ensure model sees enough variety of speech
    num_negatives_needed = int(num_breath_samples * 1.5)

    attempts = 0
    collected_negatives = 0
    max_attempts = num_negatives_needed * 5

    while collected_negatives < num_negatives_needed and attempts < max_attempts:
        attempts += 1
        # Random start point
        rand_idx = np.random.randint(0, len(y) - int(WINDOW_LENGTH * sr))

        # Check if this whole window is clean (no overlap with breath mask)
        window_mask = breath_mask[rand_idx: rand_idx + int(WINDOW_LENGTH * sr)]
        if not np.any(window_mask):
            window = y[rand_idx: rand_idx + int(WINDOW_LENGTH * sr)]
            feats = extract_features(window, sr)
            if feats is not None:
                X.append(feats)
                y_labels.append(0)
                collected_negatives += 1

    print(f"Total Training Samples: {len(X)} ({num_breath_samples} Breaths, {collected_negatives} Non-Breaths)")

    # --- Step 4: Training ---
    print("--- Step 4: Training Classifier ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # --- Step 5: Evaluation ---
    print("\n--- Model Results ---")
    predictions = clf.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions, target_names=['Speech/Silence', 'Breath']))

    # Save
    joblib.dump(clf, MODEL_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")


# --- CONFIGURATION ---
AUDIO_FILE = "input/112 A-Dzogchen Lamrin Yeshey Drupa_improved.wav"  # Your 30 min audio file
LABELS_FILE = "input/112 A-Dzogchen Lamrin Yeshey Drupa_improved_inbreaths.srt"  # Your SRT file where subs = breaths
MODEL_SAVE_PATH = "split_on_inbreaths_model.pkl"

if __name__ == "__main__":
    train_model()
