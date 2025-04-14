import os
import re
from pydub import AudioSegment

input_folder = "S:\master poli\sem 2\iaaecv\extracted_speakers"
output_folder = "S:\master poli\sem 2\iaaecv\concatenated_files"
os.makedirs(output_folder, exist_ok=True)

# 6s - dimensiune aleasa empiric 
MAX_LENGTH_MS = 6000  

def split_audio_into_chunks(audio, max_length_ms=6000):
    chunks = []
    start = 0
    while start < len(audio):
        end = start + max_length_ms
        chunk = audio[start:end]
        chunks.append(chunk)
        start = end
    return chunks

def pad_audio(audio, target_length_ms=6000):
    if len(audio) < target_length_ms:
        silence_needed = target_length_ms - len(audio)
        silence_segment = AudioSegment.silent(duration=silence_needed)
        audio = audio + silence_segment
    return audio

all_files = [f for f in os.listdir(input_folder) if f.endswith(".wav")]

groups = {}
for filename in all_files:
    name_parts = filename.split("_")

    group_prefix = "_".join(name_parts[:5])
    if group_prefix not in groups:
        groups[group_prefix] = []
    groups[group_prefix].append(filename)

for group_prefix, filenames in groups.items():
    filenames.sort()

    segment_list = []  
    current_segment = AudioSegment.empty() 

    segment_counter = 1  

    for filename in filenames:
        file_path = os.path.join(input_folder, filename)
        audio = AudioSegment.from_wav(file_path)

        sub_chunks = split_audio_into_chunks(audio, MAX_LENGTH_MS)

        for sc in sub_chunks:
            if len(current_segment) + len(sc) <= MAX_LENGTH_MS:
                current_segment += sc
            else:
                remainder_allowed = MAX_LENGTH_MS - len(current_segment)
                if remainder_allowed > 0:
                    current_segment += sc[:remainder_allowed]
                    sc = sc[remainder_allowed:] 

                out_name = f"{group_prefix}_segment{segment_counter:02d}.wav"
                out_path = os.path.join(output_folder, out_name)
                current_segment.export(out_path, format="wav")
                segment_counter += 1

                current_segment = sc

                while len(current_segment) > MAX_LENGTH_MS:
                    chunk_to_export = current_segment[:MAX_LENGTH_MS]
                    current_segment = current_segment[MAX_LENGTH_MS:]

                    out_name = f"{group_prefix}_segment{segment_counter:02d}.wav"
                    out_path = os.path.join(output_folder, out_name)
                    chunk_to_export.export(out_path, format="wav")
                    segment_counter += 1

    if len(current_segment) > 0:
        current_segment = pad_audio(current_segment, MAX_LENGTH_MS)
        out_name = f"{group_prefix}_segment{segment_counter:02d}.wav"
        out_path = os.path.join(output_folder, out_name)
        current_segment.export(out_path, format="wav")
        segment_counter += 1

    print(f"finished file : {group_prefix}")

print("finished processing")
