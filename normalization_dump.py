#!/usr/bin/env python3
"""
Normalize features per speaker, ignoring speakers with only one recording.
Input : signal_features2_std_gender_no_mfcc_norm.csv
Output: signal_features_per_speaker_normalized.csv
"""
import re
import numpy as np
import pandas as pd

# ------------ paths ---------------------------------------------------------
INPUT_PATH  = "mfcc_f0_features.csv"
OUTPUT_PATH = "norm_dump.csv"
# ----------------------------------------------------------------------------

print(f"🔄  Loading {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH)

# ---------------------------------------------------------------------------
# 1. Detect the filename column
# ---------------------------------------------------------------------------
filename_col = next(
    (c for c in df.columns if "file" in c.lower() or "name" in c.lower()),
    None,
)
if filename_col is None:
    raise RuntimeError("No filename column found (look for 'file' or 'name').")
print(f"📂  Filename column identified: {filename_col}")

# ---------------------------------------------------------------------------
# 2. Extract full speaker code (lie|truth + zero-padded digits)
# ---------------------------------------------------------------------------
# Updated regex to match 'lie' or 'truth' even within longer words
pat = re.compile(r"(?i)(lie|truth)_?(\d{1,3})")

def get_speaker(text: str) -> str:
    m = pat.search(str(text))
    if not m:
        return None
    prefix, num = m.groups()
    return f"{prefix.lower()}_{int(num):03d}"  # e.g. lie_007

# Apply extraction, count skips
speakers = df[filename_col].apply(get_speaker)
skipped = speakers.isna().sum()
if skipped > 0:
    print(f"⚠️  {skipped} rows have no valid speaker code and will be dropped.")
df = df[speakers.notna()].copy()
df['speaker'] = speakers[speakers.notna()]

print(f"🗣️   Unique speakers before filtering: {df['speaker'].nunique()}")

# ---------------------------------------------------------------------------
# 3. Filter out speakers with only one recording
# ---------------------------------------------------------------------------
counts = df['speaker'].value_counts()
single = counts[counts == 1].index.tolist()
if single:
    print(f"⚠️  {len(single)} speakers with a single recording will be removed.")
# Keep only speakers with >1 recordings
valid_speakers = counts[counts > 1].index
filtered_df = df[df['speaker'].isin(valid_speakers)].copy()
print(f"🔄  Speakers retained for normalization: {len(valid_speakers)} (out of {df['speaker'].nunique()})")

# ---------------------------------------------------------------------------
# 4. Choose numeric feature columns (skip meta columns)
# ---------------------------------------------------------------------------
numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
meta_cols    = {"gender", "label", "speaker"}
feature_cols = [c for c in numeric_cols if c not in meta_cols]
print(f"🧮  Features to normalize: {len(feature_cols)} columns")

# ---------------------------------------------------------------------------
# 5. Z-score normalization within each speaker
# ---------------------------------------------------------------------------
means = filtered_df.groupby('speaker')[feature_cols].transform('mean')
stds  = filtered_df.groupby('speaker')[feature_cols].transform('std').replace(0, 1)
filtered_df[feature_cols] = (filtered_df[feature_cols] - means) / stds

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
filtered_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅  Normalized CSV written to {OUTPUT_PATH}")
print(f"🗣️   Final unique speakers in output: {filtered_df['speaker'].nunique()}")
