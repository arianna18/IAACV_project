import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, hamming
import matplotlib.pyplot as plt
import pickle

def extract_features(audio_path, output_dir):
    # Parametrii recomandați
    fs = 16000  # Frecvența de eșantionare
    frame_length = 400  # 25ms = 400 samples @16kHz
    frame_overlap = 240  # 15ms = 240 samples
    frame_step = 160  # 10ms = 160 samples
    nfft = 512  # Puncte DFT
    max_freq = 8000  # Frecvența maximă de interes [Hz]
    
    # Citire fișier audio
    sample_rate, audio = wavfile.read(audio_path)
    if sample_rate != fs:
        raise ValueError(f"Frecvența de eșantionare trebuie să fie {fs}Hz")
    
    # Normalizare și conversie la mono dacă e stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    # audio = audio / np.max(np.abs(audio))
    
    # Padding cu zerouri pentru ultimul cadru
    pad_length = frame_length - (len(audio) - frame_length) % frame_step
    audio = np.pad(audio, (0, pad_length), 'constant')
    
    # Extragere cadre
    frames = []
    for i in range(0, len(audio)-frame_length+1, frame_step):
        frame = audio[i:i+frame_length]
        frames.append(frame)
    
    # Calcul spectrogramă
    window = hamming(frame_length, sym=False)
    f, t, Zxx = stft(audio, fs=fs, window=window, nperseg=frame_length,
                     noverlap=frame_overlap, nfft=nfft)
    
    # Filtrăm doar frecvențele până la 8kHz (primele 257 puncte)
    freq_mask = f <= max_freq
    Zxx = Zxx[freq_mask, :]
    f = f[freq_mask]
    
    # Conversie la dB
    Sxx = 10 * np.log10(np.abs(Zxx)**2 + 1e-10)
    
    # Salvăm rezultatele
    base_name = os.path.basename(audio_path).replace('.wav', '')
    
    # Salvăm cadrele brute
    np.save(os.path.join(output_dir, f'{base_name}_frames.npy'), np.array(frames))
    
    # Salvăm spectrograma
    np.save(os.path.join(output_dir, f'{base_name}_spectrogram.npy'), Sxx)
    
    # Salvăm și o imagine a spectrogramei pentru vizualizare
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.colorbar(label='Intensitate [dB]')
    plt.ylabel('Frecvență [Hz]')
    plt.xlabel('Timp [s]')
    plt.title('Spectrogramă')
    plt.savefig(os.path.join(output_dir, f'{base_name}_spectrogram.png'))
    plt.close()
    
    # Returnăm și metadate
    metadata = {
        'sample_rate': fs,
        'frame_length': frame_length,
        'frame_overlap': frame_overlap,
        'frame_step': frame_step,
        'nfft': nfft,
        'max_freq': max_freq,
        'original_length': len(audio),
        'num_frames': len(frames),
        'spectrogram_shape': Sxx.shape
    }
    
    with open(os.path.join(output_dir, f'{base_name}_metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    return metadata

def process_all_audio(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for file in os.listdir(input_dir):
        if file.endswith('.wav'):
            audio_path = os.path.join(input_dir, file)
            print(f"Procesez: {file}")
            try:
                extract_features(audio_path, output_dir)
            except Exception as e:
                print(f"Eroare la procesarea {file}: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='Directorul cu fișierele audio .wav')
    parser.add_argument('output_dir', help='Directorul pentru salvarea rezultatelor')
    args = parser.parse_args()
    
    process_all_audio(args.input_dir, args.output_dir)
    print("Procesarea completă!")