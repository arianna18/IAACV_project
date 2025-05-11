import re
import glob
import os
import numpy as np
import pandas as pd
import librosa
from scipy.stats import skew, kurtosis


AUDIO_GLOB = r"S:/master_poli/sem_2/iaeecv2/IAACV_project/processed_files/*.wav"
OUTPUT_CSV = "audio_features.csv"

pat = re.compile(r"(?i)(?:^|_)(truth|lie)[ _]?(\d{1,3})")

records = []
for filepath in glob.glob(AUDIO_GLOB):
    fname = os.path.basename(filepath)
    m = pat.search(fname)
    label, sid = m.groups()
    label = label.lower()
    speaker_id = f"{label}_{int(sid):03d}"

    y, sr = librosa.load(filepath, sr=16000)
    frame_length = int(0.025*sr)
    hop_length = int(0.010*sr)

    amp_std = np.std(y)
    amp_skew = skew(y)
    amp_kurt = kurtosis(y)
    energy = float(np.sum(y**2))

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    mean_rms = float(np.mean(rms))
    var_rms = float(np.var(rms))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)))
    mean_zcr = float(np.mean(zcr))

    records.append({
        'speaker': speaker_id,
        'std': amp_std,
        'skewness': amp_skew,
        'kurtosis': amp_kurt,
        'energy': energy,
        'mean_rms': mean_rms,
        'var_rms': var_rms,
        'mean_zcr' : mean_zcr,
    })

# output csv file 
df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)
print(f"extracted features from {len(df)} samples to {OUTPUT_CSV}")
