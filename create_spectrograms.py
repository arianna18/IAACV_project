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
    
    # # Normalizare și conversie la mono dacă e stereo
    # if len(audio.shape) > 1:
    #     audio = np.mean(audio, axis=1)
    # audio = audio / np.max(np.abs(audio))
    
    # Padding cu zerouri pentru ultimul cadru
    pad_length = 12*fs - len(audio)
    # pad_length = 12*fs
    audio = np.pad(audio, (0, pad_length), 'constant')

    # Extragere cadre
    ham_window = hamming(frame_length, sym=False)  # Create the Hamming window once
    
    # Compute the STFT (spectrogram) using the same Hamming window
    f, t, Zxx = stft(
        audio,
        fs=fs,
        window=ham_window, 
        nperseg=frame_length, 
        noverlap=frame_overlap, 
        nfft=nfft
    )

    # Filtrăm doar frecvențele până la 8kHz (primele 257 puncte)
    freq_mask = f <= max_freq
    Zxx = Zxx[freq_mask, :]
    f = f[freq_mask]
    
    # Conversie la dB
    Sxx = 10 * np.log10(np.abs(Zxx)**2 + 1e-10)
    
    # Salvăm rezultatele
    base_name = os.path.basename(audio_path).replace('.wav', '')
    
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
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('input_dir', help='Directorul cu fișierele audio .wav')
    # parser.add_argument('output_dir', help='Directorul pentru salvarea rezultatelor')
    # args = parser.parse_args()
    
    process_all_audio("S:\master poli\sem 2\iaaecv\extracted_speakers","S:\master poli\sem 2\iaaecv\spectrograms")
    print("Procesarea completă!")