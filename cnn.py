import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Functia de incărcare a spectrogramelor și etichetelor
def load_spectrograms_and_labels(spectrogram_dir):
    spectrograms = []
    labels = []
    
    for file in os.listdir(spectrogram_dir):
        if file.endswith('_spectrogram.npy'):
            # Încarcă spectrograma
            spectrogram = np.load(os.path.join(spectrogram_dir, file))
            
            # Verificăm dimensiunea spectrogramei și adăugăm o dimensiune suplimentară pentru canal
            spectrogram = np.expand_dims(spectrogram, axis=-1)  # Adăugăm dimensiunea pentru canal (1)
            
            # Extrage eticheta din numele fișierului (presupunem că este adăugată în nume)
            if 'truth' in file:
                label = 1  # Eticheta pentru "truth"
            elif 'lie' in file:
                label = 0  # Eticheta pentru "lie"
            else:
                continue  # Dacă nu găsești eticheta, sari peste fișier
            
            # Adăugăm spectrograma și eticheta în liste
            spectrograms.append(spectrogram)
            labels.append(label)
    
    # Transformăm listele în arrays numpy
    spectrograms = np.array(spectrograms)
    labels = np.array(labels)
    
    return spectrograms, labels

# Funcția pentru crearea modelului CNN
def create_cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))  # Input layer
    
    # Primul bloc de convoluție
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Al doilea bloc de convoluție
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Al treilea bloc de convoluție
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Aplatizare (Flatten) și adăugare de dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    
    # Strat de ieșire
    model.add(layers.Dense(1, activation='sigmoid'))  # Sigmoid pentru clasificare binară
    
    # Compilarea modelului
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Antrenarea modelului
def train_model(spectrograms, labels):
    # Normalizare spectrograme (scalare la intervalul [0, 1])
    spectrograms = spectrograms / np.max(spectrograms)
    
    # Împărțirea datelor în seturi de antrenament și test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(spectrograms, labels, test_size=0.2, random_state=42)

    # Crearea modelului CNN
    input_shape = X_train.shape[1:]  # Dimensiunea spectrogramelor
    model = create_cnn_model(input_shape)
    
    # Antrenarea modelului
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    # Evaluarea modelului
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Precizia pe setul de test: {test_accuracy * 100:.2f}%')

    return model

# Principal
def main():
    spectrogram_dir = "D:\Iulia\ETTI\Master\Semestrul 2\IAACV_project\spectrograms"
    
    # Încarcă spectrogramele și etichetele
    spectrograms, labels = load_spectrograms_and_labels(spectrogram_dir)
    
    # Verificăm dimensiunea spectrogramelor
    print(f"Dimensiunea spectrogramelor: {spectrograms.shape}")
    
    # Antrenăm modelul
    model = train_model(spectrograms, labels)
    
    # Salvăm modelul antrenat
    model.save('cnn_audio_classifier.h5')
    print("Modelul a fost salvat!")

if __name__ == "__main__":
    main()