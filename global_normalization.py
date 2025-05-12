#!/usr/bin/env python3
"""
Globally normalize features across the entire dataset.
Input : signal_features2_std_gender_no_mfcc_norm.csv
Output: signal_features_global_normalized.csv
"""
import re
import numpy as np
import pandas as pd

# ------------ paths ---------------------------------------------------------
INPUT_PATH  = "signal_features2_std_gender_mfcc13_per_file.csv"
OUTPUT_PATH = "signal_features_global_normalized.csv"
# ----------------------------------------------------------------------------

print(f"Loading {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH)

# ---------------------------------------------------------------------------
# 1. Detect the filename column (optional extraction of speaker codes for reference)
# ---------------------------------------------------------------------------
filename_col = next(
    (c for c in df.columns if "file" in c.lower() or "name" in c.lower()),
    None,
)
if filename_col:
    print(f"Filename column identified: {filename_col}")
    # Extract speaker code for informational purposes
    pat = re.compile(r"(?i)(lie|truth)_?(\d{1,3})")
    df["speaker"] = df[filename_col].apply(lambda x: (
        (lambda m: f"{m.group(1).lower()}_{int(m.group(2)):03d}") if (m := pat.search(str(x))) else None
    ))
    print(f"Unique speakers detected (info only): {df['speaker'].nunique(dropna=True)}")
else:
    print("No filename column found; skipping speaker extraction.")

# ---------------------------------------------------------------------------
# 2. Choose numeric feature columns (skip non-features)
# ---------------------------------------------------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
meta_cols    = {"gender", "label", "speaker"}
feature_cols = [c for c in numeric_cols if c not in meta_cols]
print(f"Numeric features to normalize globally: {len(feature_cols)} columns")

# ---------------------------------------------------------------------------
# 3. Compute global statistics and apply Z-score normalization
# ---------------------------------------------------------------------------
# Compute global mean and std for each feature
global_means = df[feature_cols].mean()
global_stds = df[feature_cols].std().replace(0, 1)
print("Applying global Z-score normalization across all samples.")
# Normalize all feature columns
normalized_df = df.copy()
normalized_df[feature_cols] = (df[feature_cols] - global_means) / global_stds

# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------
normalized_df.to_csv(OUTPUT_PATH, index=False)
print(f"Global normalized CSV written to {OUTPUT_PATH}")
print(f"Final unique speakers in output (for reference): {normalized_df['speaker'].nunique(dropna=True) if 'speaker' in normalized_df else 'N/A'}")
