import librosa
import numpy as np
import re
from pathlib import Path
import datetime

# --- CONFIGURATION ---
INPUT_DIR = Path("/run/media/drupchen/Khyentse Önang/K-Ö Archives/Transcriptions/སྙིང་ཐིག་མ་བུའི་ཁྲིད་ཡིག་དྲི་མེད་ཞལ་ལུང་།")                   # Set to "." for current directory, or "data_folder"
OUTPUT_DIR = Path("training_breaths")   # A new folder will be created for the clean SRTs

SEARCH_WINDOW_SEC = 1.5    # How much time to look at around your rough boundary (+/- 0.75s)
SPEECH_THRESHOLD = 0.2     # Relative volume level that defines the "mountains" (speech)
SILENCE_THRESHOLD = 0.02   # Relative volume level that defines the flatline (true silence)


def parse_srt_boundaries(srt_path):
    """Extracts all unique start/end timestamps from the rough SRT to use as search centers."""
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = re.compile(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})')
    matches = pattern.findall(content)

    boundaries = []
    for start_str, end_str in matches:
        def to_sec(t_str):
            h, m, s_ms = t_str.split(':')
            s, ms = s_ms.split(',')
            return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000.0
        
        boundaries.extend([to_sec(start_str), to_sec(end_str)])

    return sorted(list(set(boundaries)))


def format_srt_time(seconds):
    seconds = max(0, seconds)
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def find_rugby_ball(y, sr, center_time):
    """Hunts for the isolated breath bump within a window."""
    half_window = SEARCH_WINDOW_SEC / 2.0
    start_time = max(0, center_time - half_window)
    end_time = min(len(y)/sr, center_time + half_window)
    
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    window = y[start_sample:end_sample]
    if len(window) == 0: return None
    
    # 1. Calculate energy envelope (RMS)
    rms = librosa.feature.rms(y=window, frame_length=512, hop_length=128)[0]
    
    if np.max(rms) == 0: return None
    rms_norm = rms / np.max(rms)
    
    times = librosa.frames_to_time(np.arange(len(rms_norm)), sr=sr, hop_length=128) + start_time
    
    # 2. Find the center of our window
    center_idx = np.argmin(np.abs(times - center_time))
    
    # 3. Find the "Valley" (The gap between speech mountains)
    valley_start_idx = center_idx
    while valley_start_idx > 0 and rms_norm[valley_start_idx] < SPEECH_THRESHOLD:
        valley_start_idx -= 1
        
    valley_end_idx = center_idx
    while valley_end_idx < len(rms_norm) - 1 and rms_norm[valley_end_idx] < SPEECH_THRESHOLD:
        valley_end_idx += 1
        
    # 4. Find the "Rugby Ball" inside the valley
    valley_rms = rms_norm[valley_start_idx:valley_end_idx]
    if len(valley_rms) == 0: return None
    
    peak_in_valley_idx = np.argmax(valley_rms)
    absolute_peak_idx = valley_start_idx + peak_in_valley_idx
    
    if rms_norm[absolute_peak_idx] < SILENCE_THRESHOLD:
        return None
        
    # Walk left and right to find where the breath fades into true silence
    breath_start_idx = absolute_peak_idx
    while breath_start_idx > valley_start_idx and rms_norm[breath_start_idx] > SILENCE_THRESHOLD:
        if rms_norm[breath_start_idx - 1] > rms_norm[breath_start_idx] * 1.5:
            break
        breath_start_idx -= 1
        
    breath_end_idx = absolute_peak_idx
    while breath_end_idx < valley_end_idx and rms_norm[breath_end_idx] > SILENCE_THRESHOLD:
        if rms_norm[breath_end_idx + 1] > rms_norm[breath_end_idx] * 1.5:
            break
        breath_end_idx += 1
        
    final_start = times[breath_start_idx]
    final_end = times[breath_end_idx]
    
    if 0.1 < (final_end - final_start) < 1.5:
        return (final_start, final_end)
    return None


def process_file(wav_path, srt_path, output_path):
    print(f"\nProcessing: {wav_path.name}")
    y, sr = librosa.load(wav_path, sr=16000)
    
    boundaries = parse_srt_boundaries(srt_path)
    print(f"  -> Found {len(boundaries)} rough boundaries to scan.")
    
    refined_breaths = []
    
    for center_time in boundaries:
        breath_times = find_rugby_ball(y, sr, center_time)
        if breath_times:
            # Prevent overlapping duplicates
            if not refined_breaths or breath_times[0] > refined_breaths[-1][1]:
                refined_breaths.append(breath_times)
                
    print(f"  -> Successfully extracted {len(refined_breaths)} breath segments.")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, (start, end) in enumerate(refined_breaths):
            f.write(f"{idx + 1}\n")
            f.write(f"{format_srt_time(start)} --> {format_srt_time(end)}\n\n")


def main():
    # Ensure the output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all .wav files in the target directory
    wav_files = list(INPUT_DIR.glob("*.wav"))
    
    if not wav_files:
        print(f"No .wav files found in '{INPUT_DIR.absolute()}'")
        return

    for wav_path in wav_files:
        # Look for a matching .srt file
        srt_path = wav_path.with_suffix('.srt')
        
        if srt_path.exists():
            output_name = f"{wav_path.stem}_training_breaths.srt"
            output_path = OUTPUT_DIR / output_name
            process_file(wav_path, srt_path, output_path)
        else:
            print(f"\nSkipping: {wav_path.name} (No matching .srt file found)")

    print(f"\nBatch processing complete! Cleaned SRTs are saved in '{OUTPUT_DIR.absolute()}'")


if __name__ == "__main__":
    main()