import os
import glob
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment

input_folder = r"./extracted_speakers"
output_folder = os.path.join(input_folder, "processed_files")
target_length = 6  # seconds
min_length = 2  # minimum length
sample_rate = 22050

os.makedirs(output_folder, exist_ok=True)

def process_audio_files():
    audio_files = glob.glob(os.path.join(input_folder, "*.wav"))
    file_groups = {}
    
    for file_path in audio_files:
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        
        if len(parts) < 6 or parts[0] != "trial":
            print(f"Skipping file with unexpected name format: {filename}")
            continue
            
        verdict = parts[1]
        index = parts[2]
        speaker_id = parts[4]
        
        key = (speaker_id, verdict, index)
        
        if key not in file_groups:
            file_groups[key] = []
        file_groups[key].append(file_path)
    
    for key, files in file_groups.items():
        speaker_id, verdict, index = key
        combined_audio = np.array([], dtype=np.float32)
        
        files.sort()
        for file_path in files:
            try:
                audio, sr = librosa.load(file_path, sr=sample_rate)
                combined_audio = np.concatenate((combined_audio, audio))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        total_duration = len(combined_audio) / sample_rate
        
        if total_duration == 0:
            print(f"No audio data for group {key}, skipping")
            continue

        samples_per_segment = target_length * sample_rate
        num_segments = int(np.ceil(total_duration / target_length))
        
        for i in range(num_segments):
            start_sample = i * samples_per_segment
            end_sample = start_sample + samples_per_segment
            
            segment = combined_audio[start_sample:end_sample]
            
            segment_duration = len(segment) / sample_rate
            
            if segment_duration < min_length:
                print(f"Skipping segment {i+1} for group {key} - duration {segment_duration:.2f}s < {min_length}s")
                continue
            
            if len(segment) < samples_per_segment:
                padding = samples_per_segment - len(segment)
                segment = np.pad(segment, (0, padding), mode='constant')
            
            output_filename = f"trial_{verdict}_{index}_speaker{speaker_id}_segment{i+1:02d}.wav"
            output_path = os.path.join(output_folder, output_filename)
            
            sf.write(output_path, segment, sample_rate)
            print(f"Saved: {output_path}")

if __name__ == "__main__":
    process_audio_files()
    print("Processing complete!")