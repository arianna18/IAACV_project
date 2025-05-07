import numpy as np 
import scipy as sp
import scipy.io.wavfile as wav
from fractions import Fraction
from python_speech_features import mfcc
import os
import pandas as pd
import librosa

# Parameters
Tw = 0.025                      # Frame duration in seconds
Tstep = 0.010                   # Frame shift in seconds
Fs = int(16e3)                  # Target sampling rate [Hz]
Nw = int(np.ceil(Tw * Fs))      # Frame size (samples)
Nstep = int(np.ceil(Tstep * Fs))# Frame shift (samples)
Nfft = int(2 ** np.ceil(np.log2(Nw)))
Nfilt = 26
Nmfcc = 13
f_low = 0
f_high = Fs / 2

# F0 extraction parameters
f_min = 75          # Minimum F0 [Hz]
f_max = 300         # Maximum F0 [Hz]

wav_dir = './processed_files'  

# This will store our features per file
file_features = []

# Process each .wav file in the directory
for filename in os.listdir(wav_dir):
    if filename.endswith('.wav'):
        filepath = os.path.join(wav_dir, filename)
        print(f"Processing file: {filename}")
        
        try:
            # Read the audio file
            Fs0, x = wav.read(filepath)
            
            # Convert to mono if needed and normalize
            if len(x.shape) > 1:
                x = np.mean(x, axis=1)
            x = x / np.maximum(np.amax(np.absolute(x)), 0.01)
            
            # Resample if necessary to the target Fs
            res_rat = Fs / Fs0
            if res_rat != 1:
                F_res_rat = Fraction(str(res_rat)).limit_denominator(1000)
                P = F_res_rat.numerator
                Q = F_res_rat.denominator
                b_res = sp.signal.firwin(1001, 1 / max(P, Q))
                x = sp.signal.resample_poly(x, P, Q, window=b_res)
            x = x / np.maximum(np.amax(np.absolute(x)), 0.01)
            x = sp.signal.medfilt(x, 7)

            # Extract MFCC features for all frames
            mfcc_feat = mfcc(x, Fs, Tw, Tstep, Nmfcc, Nfilt, Nfft, f_low, f_high,
                             appendEnergy=False, winfunc=sp.signal.windows.hamming)
            
            # Calculate mean MFCC coefficients across all frames
            mean_mfcc = np.mean(mfcc_feat, axis=0)
            
            # Normalize the mean MFCC features
            mean_mfcc -= np.mean(mean_mfcc)
            std_val = np.std(mean_mfcc)
            if std_val > 0:
                mean_mfcc /= std_val

            # Extract F0 values for all frames
            f0, voiced_flag, voiced_prob = librosa.pyin(x, fmin=f_min, fmax=f_max, sr=Fs,
                                                      frame_length=Nw, hop_length=Nstep)
            # Replace unvoiced frame estimates (NaN) with zeros
            f0_est = np.where(np.isnan(f0), 0, f0)
            
            # Calculate mean F0 (only considering voiced frames)
            voiced_f0 = f0_est[f0_est > 0]
            # mean_f0 = np.mean(voiced_f0) if len(voiced_f0) > 0 else 0
            mean_f0 = np.mean([i for i in voiced_f0 if i > 0]) if len(voiced_f0) > 0 else 0



            # Extract signal label from filename
            if "lie" in filename.lower():
                signal_label = "lie"
            elif "truth" in filename.lower():
                signal_label = "truth"
            else:
                signal_label = "unknown"

            # Create a row with 13 mean MFCC coefficients, mean F0, and the signal label
            row = list(mean_mfcc) + [mean_f0, signal_label, filename]
            file_features.append(row)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

# Create column names for the DataFrame
column_names = [f"MFCC_{i+1}" for i in range(Nmfcc)] + ["F0", "label", "filename"]

# Create a Pandas DataFrame from the extracted features
DATASET = pd.DataFrame(file_features, columns=column_names)
print(DATASET.head())

# Save the dataset to a CSV file
DATASET.to_csv('./signal_features2.csv', index=False)