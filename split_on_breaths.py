from pathlib import Path

import librosa
import numpy as np
import joblib
import datetime
import sys
import os


# --- FEATURE EXTRACTOR (Must match trainer) ---
def extract_features(y, sr):
    try:
        if len(y) < 512: return None
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
        return np.hstack([mfcc, cent, zcr, rms])
    except Exception:
        return None


def format_srt_time(seconds):
    seconds = max(0, seconds)
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def scan_audio(audio_path, model_path):
    print(f"Loading Audio: {audio_path} ...")
    y, sr = librosa.load(audio_path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)

    print(f"Loading Model: {model_path} ...")
    clf = joblib.load(model_path)

    # Scanning Parameters
    window_size = 0.5
    step_size = 0.1

    print(f"Scanning audio (Threshold: {BREATH_THRESHOLD})...")

    predictions = []
    timestamps = []

    total_steps = int((len(y) - int(window_size * sr)) / int(step_size * sr))

    for i, start_sample in enumerate(range(0, len(y) - int(window_size * sr), int(step_size * sr))):
        end_sample = start_sample + int(window_size * sr)
        window = y[start_sample:end_sample]

        feat = extract_features(window, sr)

        if feat is not None:
            # Use predict_proba to get confidence score
            probs = clf.predict_proba([feat])[0]
            # probs[0] is probability of '0' (Speech)
            # probs[1] is probability of '1' (Breath)

            breath_prob = probs[1]

            # Apply custom threshold
            is_breath = 1 if breath_prob >= BREATH_THRESHOLD else 0

            current_time = start_sample / sr
            predictions.append(is_breath)
            timestamps.append(current_time)

        if i % 2000 == 0:
            print(f"   Progress: {i / total_steps * 100:.1f}%", end='\r')

    print("\nmerging raw predictions...")

    candidate_breaths = []
    in_breath = False
    breath_start = 0

    for i in range(1, len(predictions)):
        pred = predictions[i]
        prev_pred = predictions[i - 1]
        time = timestamps[i]

        if pred == 1 and prev_pred == 1 and not in_breath:
            in_breath = True
            breath_start = timestamps[i - 1]

        elif pred == 0 and in_breath:
            in_breath = False
            breath_end = time - TRIM_END_SECONDS

            if (breath_end - breath_start) > 0.2:
                candidate_breaths.append((breath_start, breath_end))

    print(f"Found {len(candidate_breaths)} candidates. Applying 5s rule...")

    # --- 5 SECOND RULE ---
    final_breaths = []
    last_accepted_split_end = 0.0

    for b_start, b_end in candidate_breaths:
        speech_duration = b_start - last_accepted_split_end

        if speech_duration >= MIN_SPEECH_DURATION:
            final_breaths.append((b_start, b_end))
            last_accepted_split_end = b_end

    print(f"Final Count: {len(final_breaths)} breaths.")
    return final_breaths, duration


def generate_srt(breaths, duration, output_path):
    segments = []
    current_start = 0.0

    for i, (breath_start, breath_end) in enumerate(breaths):
        segments.append({
            "index": i + 1,
            "start": current_start,
            "end": breath_end,
            "text": ""
        })
        current_start = breath_end

    if current_start < duration:
        segments.append({
            "index": len(segments) + 1,
            "start": current_start,
            "end": duration,
            "text": ""
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"{seg['index']}\n")
            f.write(f"{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n")
            f.write(f"{seg['text']}\n\n")

    print(f"SRT saved to: {output_path}")


# --- CONFIGURATION ---
INPUT_AUDIO_PATH = Path('/run/media/drupchen/Khyentse Önang/K-Ö Archives/Transcriptions/སྙིང་ཐིག་མ་བུའི་ཁྲིད་ཡིག་དྲི་མེད་ཞལ་ལུང་།/')  # The file you want to split
MODEL_PATH = "split_on_inbreaths_model.pkl"  # The model you just trained


# --- TUNING KNOBS ---
# 1. BREATH SENSITIVITY (0.0 to 1.0)
# Lower this to catch missed breaths. Try 0.35 or 0.4.
# (0.5 is the default "strict" mode)
BREATH_THRESHOLD = 0.35

# 2. MIN SPEECH DURATION (Seconds)
# Enforces the 5s rule to prevent rapid-fire splitting
MIN_SPEECH_DURATION = 2.0

# 3. TRIM END (Seconds)
TRIM_END_SECONDS = 0.0


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("Model file not found! Please train it first.")
    else:
        for f in Path(INPUT_AUDIO_PATH).glob("*.wav"):
            if "09" not in f.stem:
                continue
            OUTPUT_SRT = INPUT_AUDIO_PATH / (f.stem + '.srt')
            breaths, dur = scan_audio(f, MODEL_PATH)
            generate_srt(breaths, dur, OUTPUT_SRT)