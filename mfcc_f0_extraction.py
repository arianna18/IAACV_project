import re
import glob
import os
import numpy as np
import pandas as pd
import librosa

# ------------ paths / patterns ----------------------------------------------
AUDIO_GLOB = r"S:/master_poli/sem_2/iaeecv2/IAACV_project/processed_files/*.wav"
OUTPUT_CSV = "mfcc_f0_extraction.csv"
# ----------------------------------------------------------------------------

# Regex to extract label (truth or lie + 3-digit ID) from filename
pat_label = re.compile(r"(?i)(truth|lie)[ _]?(\d{1,3})")

# Frame / hop sizes in samples for 25 ms / 10 ms at 16 kHz
FRAME_LENGTH = int(0.025 * 16000)    # 400 samples
HOP_LENGTH   = int(0.010 * 16000)    # 160 samples

records = []
for filepath in glob.glob(AUDIO_GLOB):
    fname = os.path.basename(filepath)
    m = pat_label.search(fname)
    if not m:
        print(f"⚠️  Skipping (no label): {fname}")
        continue
    label, sid = m.groups()
    label = f"{label.lower()}_{int(sid):03d}"
    print('process:', label)

    # Load audio at 16 kHz
    y, sr = librosa.load(filepath, sr=16000)

    # 1) MFCC extraction using 25 ms frames, 10 ms hop
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13,
        n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH
    )
    mfcc_means = np.mean(mfcc, axis=1)
    mfcc_stds  = np.std(mfcc, axis=1)

    # 2) Fundamental frequency (pitch) estimation via librosa.pyin
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )
    # f0 has NaN for unvoiced frames
    voiced_f0 = f0[~np.isnan(f0)]
    if voiced_f0.size == 0:
        f0_mean = np.nan
        f0_std  = np.nan
    else:
        f0_mean = float(np.mean(voiced_f0))
        f0_std  = float(np.std(voiced_f0))

    # Build record
    rec = {'filename': fname, 'label': label}
    for i in range(13):
        rec[f"mfcc{i+1}_mean"] = float(mfcc_means[i])
        rec[f"mfcc{i+1}_std"]  = float(mfcc_stds[i])
    rec['f0_mean'] = f0_mean
    rec['f0_std']  = f0_std
    records.append(rec)

# Save to CSV
if records:
    df = pd.DataFrame(records)
    cols = [f"mfcc{i+1}_mean" for i in range(13)]
    cols += [f"mfcc{i+1}_std" for i in range(13)]
    cols += ['f0_mean', 'f0_std']
    cols += ['filename', 'label']

    df.to_csv(OUTPUT_CSV, index=False, columns=cols)
    print(f"✅  Extracted {len(df)} recordings to {OUTPUT_CSV}")
else:
    print("⚠️  No records extracted. Check AUDIO_GLOB and filenames.")
