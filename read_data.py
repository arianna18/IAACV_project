import os
import csv
import sys
from scipy.io import wavfile
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python3 read_data.py <folder_path>")
    sys.exit(1)

base_dir = sys.argv[1]

audio_dir = os.path.join(base_dir, "extrAudio")
annotation_dir = os.path.join(base_dir, "datasetAnnotation")
output_dir = os.path.join(base_dir, "extracted_speakers")

os.makedirs(output_dir, exist_ok=True)

for csv_file in os.listdir(annotation_dir):
    if csv_file.endswith('.csv'):
        wav_file = csv_file.replace('.csv', '.wav')
        wav_path = os.path.join(audio_dir, wav_file)
        
        if not os.path.exists(wav_path):
            print(f"Audio file {wav_file} does not exist")
            continue
        
        sample_rate, audio_data = wavfile.read(wav_path)
        
        csv_path = os.path.join(annotation_dir, csv_file)
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            
            for row in reader:
                start_time = float(row['Start time'])
                end_time = float(row['Stop time'])
                speaker = row['Speaker']
                gender = row['Gender']
                
                if speaker == 'TM':
                    continue
                
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                segment = audio_data[start_sample:end_sample]
                
                output_filename = (
                    f"{wav_file.replace('.wav', '')}_"
                    f"speaker{speaker}_{gender}_"
                    f"{start_time:.2f}_{end_time:.2f}.wav"
                )
                output_path = os.path.join(output_dir, output_filename)
                
                wavfile.write(output_path, sample_rate, segment)

print("Finished extraction!")