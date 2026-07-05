import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import re
import os
import json
import tempfile
import subprocess
import argparse
from pathlib import Path
import m2cgen as m2c
import soundfile as sf  # <-- Added this import

def parse_srt_time(time_str):
    """Converts '00:00:20,500' to seconds (float)."""
    hours, minutes, seconds = time_str.split(':')
    seconds, millis = seconds.split(',')
    return (int(hours) * 3600) + (int(minutes) * 60) + int(seconds) + (int(millis) / 1000)

def load_srt_labels(srt_path):
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    pattern = re.compile(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})')
    matches = pattern.findall(content)

    data = []
    for start_str, end_str in matches:
        data.append({'start': parse_srt_time(start_str), 'end': parse_srt_time(end_str)})
    return pd.DataFrame(data)

def extract_features_python(y, sr):
    """Original Python extraction."""
    try:
        if len(y) < 512: return None
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
        return np.hstack([mfcc, cent, zcr, rms])
    except Exception:
        return None

def train_model(flavor):
    X = []
    y_labels = []

    srt_files = list(SRT_DIR.glob("*.srt"))
    if not srt_files:
        print(f"No SRT files found in '{SRT_DIR}'. Please run the extraction script first.")
        return

    print(f"Found {len(srt_files)} SRT files. Starting batch processing...")

    for srt_path in srt_files:
        # Reconstruct the original WAV filename
        base_name = srt_path.stem.replace("_training_breaths", "").replace("_inbreaths", "")
        wav_path = AUDIO_DIR / f"{base_name}.wav"

        if not wav_path.exists():
            print(f"  [!] Missing Audio: '{wav_path.name}' - Skipping.")
            continue

        print(f"\n--- Processing: {base_name} ---")
        y, sr = librosa.load(wav_path, sr=16000)
        
        labels = load_srt_labels(srt_path)
        print(f"   -> Parsed {len(labels)} breath segments.")
        
        sample_ranges = []
        raw_labels = []

        WINDOW_LENGTH = 0.512
        STEP_SIZE = 0.25
        window_samples = int(WINDOW_LENGTH * sr) # Exactly 8192

        breath_mask = np.zeros(len(y), dtype=bool)

        # 1. Map Positives (Breaths)
        for _, row in labels.iterrows():
            start_sample = int(row['start'] * sr)
            end_sample = int(row['end'] * sr)
            if start_sample >= len(y): continue
            end_sample = min(end_sample, len(y))
            breath_mask[start_sample:end_sample] = True

            chunk_len = end_sample - start_sample
            if chunk_len < window_samples:
                center = start_sample + (chunk_len // 2)
                new_start = max(0, center - (window_samples // 2))
                new_end = new_start + window_samples
                
                if new_end > len(y):
                    new_end = len(y)
                    new_start = max(0, new_end - window_samples)
                    
                sample_ranges.append([new_start, new_end])
                raw_labels.append(1)
            else:
                for i in range(0, chunk_len - window_samples + 1, int(STEP_SIZE * sr)):
                    sample_ranges.append([start_sample + i, start_sample + i + window_samples])
                    raw_labels.append(1)

        # 2. Map Negatives (Speech/Silence)
        num_breath_samples = len(raw_labels)
        num_negatives_needed = int(num_breath_samples * 1.5)
        attempts, collected_negatives = 0, 0
        max_attempts = num_negatives_needed * 5

        while collected_negatives < num_negatives_needed and attempts < max_attempts:
            attempts += 1
            rand_idx = np.random.randint(0, len(y) - window_samples)
            window_mask = breath_mask[rand_idx: rand_idx + window_samples]
            if not np.any(window_mask):
                sample_ranges.append([rand_idx, rand_idx + window_samples])
                raw_labels.append(0)
                collected_negatives += 1

        print(f"   -> Extracting {len(sample_ranges)} windows via '{flavor.upper()}' math...")

        # 3. Extract Features
        if flavor == "python":
            for idx, (start, end) in enumerate(sample_ranges):
                feats = extract_features_python(y[start:end], sr)
                if feats is not None:
                    X.append(feats)
                    y_labels.append(raw_labels[idx])
                    
        elif flavor == "js":
            # --- THE FIX: Create a 16kHz Temp Audio File for Node.js ---
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as f_in, \
                 tempfile.NamedTemporaryFile(mode='r', delete=False, suffix='.json') as f_out, \
                 tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
                
                json.dump(sample_ranges, f_in)
                f_in.close()
                temp_wav.close() # Close it so soundfile can use the path

                # Write the 16kHz librosa array 'y' to the temp file
                sf.write(temp_wav.name, y, 16000)

                subprocess.run(["node", "extract_js.js", temp_wav.name, f_in.name, f_out.name], check=True)
                
                js_features = json.load(f_out)
                
                for idx, feats in enumerate(js_features):
                    if feats is not None:
                        X.append(feats)
                        y_labels.append(raw_labels[idx])
                        
                os.unlink(f_in.name)
                os.unlink(f_out.name)
                os.unlink(temp_wav.name)

    print("\n======================================")
    print(f"Total Combined Training Samples: {len(X)}")
    print("======================================")

    if len(X) == 0:
        print("No valid training data extracted. Exiting.")
        return

    print("\n--- Training Classifier ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(
    n_estimators=80, 
    max_depth=12, 
    min_samples_split=5,
    random_state=42
)
    clf.fit(X_train, y_train)

    print("\n--- Model Results ---")
    predictions = clf.predict(X_test)
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions, target_names=['Speech/Silence', 'Breath']))

    save_name = f"split_on_inbreaths_model_{flavor}.pkl"
    joblib.dump(clf, save_name)
    print(f"Model saved to: {save_name}")

    if flavor == "js":
        print("\n--- Exporting Model to JavaScript ---")
        js_code = m2c.export_to_javascript(clf)
        
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "breath_model.js")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(js_code)
            
        print(f"   -> Export complete! 'breath_model.js' is ready for your React app.")


# --- CONFIGURATION ---
AUDIO_DIR = Path("/run/media/drupchen/Khyentse Önang/K-Ö Archives/Transcriptions/སྙིང་ཐིག་མ་བུའི་ཁྲིད་ཡིག་དྲི་མེད་ཞལ་ལུང་།")
SRT_DIR = Path("training_breaths")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flavor", choices=["python", "js"], default="python", help="Choose math flavor")
    args = parser.parse_args()
    
    train_model(args.flavor)