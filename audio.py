import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the first WAV file
audio_file1 = 'trial_lie_006_speaker1_F_10.84_17.05.wav'
y1, sr1 = librosa.load(audio_file1, duration=6)  # Load first 6 seconds

# Load the second WAV file
audio_file2 = 'trial_lie_001_speaker1_F_15.03_15.76.wav'
y2, sr2 = librosa.load(audio_file2, duration=6)  # Load first 6 seconds

# Determine the longer signal length
max_len = max(len(y1), len(y2))

# Extend shorter signals to match the length of the longer signal
y1 = np.pad(y1, (0, max_len - len(y1)), mode='constant')
y2 = np.pad(y2, (0, max_len - len(y2)), mode='constant')

# Calculate time array based on the longer signal
time = np.arange(max_len) / max(sr1, sr2)

# Plotting
plt.figure(figsize=(12, 6))

# Plot the first waveform
plt.subplot(2, 1, 1)
librosa.display.waveshow(y1, sr=sr1, x_axis='time')
plt.ylabel('Amplitudine')
plt.xlabel('Timp (s)')
plt.title('Exemplu de rostire în 6s')

# Plot the second waveform
plt.subplot(2, 1, 2)
librosa.display.waveshow(y2, sr=sr2, x_axis='time')
plt.xlabel('Timp (s)')
plt.ylabel('Amplitudine')
plt.title('Exemplu de rostire sub 1s')

plt.tight_layout()
plt.show()
