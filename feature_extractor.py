import librosa
import numpy as np
import subprocess
import json
import os

def extract_features_python(y, sr):
    """The original Librosa extraction method."""
    if len(y) < 512: return None
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
    return np.hstack([mfcc, cent, zcr, rms])

def get_file_features(audio_path, flavor="python"):
    """
    Returns a list of feature arrays for the entire audio file.
    Routes to either Python/Librosa or JS/Meyda.
    """
    if flavor == "python":
        # Standard Python approach
        y, sr = librosa.load(audio_path, sr=16000)
        window_size = 0.5
        step_size = 0.1
        
        features_list = []
        for start_sample in range(0, len(y) - int(window_size * sr), int(step_size * sr)):
            end_sample = start_sample + int(window_size * sr)
            window = y[start_sample:end_sample]
            feat = extract_features_python(window, sr)
            if feat is not None:
                features_list.append(feat)
        return features_list

    elif flavor == "js":
        # Bridge to Node.js approach
        if not os.path.exists("extract_js.js"):
            raise FileNotFoundError("extract_js.js not found in current directory.")
            
        # Call the Node script and capture the JSON output
        result = subprocess.run(
            ["node", "extract_js.js", str(audio_path)], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Node.js extraction failed: {result.stderr}")
            
        # Parse the JSON string back into a Python list of lists
        return json.loads(result.stdout)