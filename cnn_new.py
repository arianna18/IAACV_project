import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import random

# ============================
# CONFIGURARE GLOBALĂ
# ============================
SPECTROGRAM_DIR = "spectrograms"
INPUT_SHAPE = (128, 128, 1)
TEST_SPLIT = 0.1
KFOLD_SPLITS = 5
EPOCHS = 100
BATCH_SIZE = 32
CNN_CONFIGS = [
    (16, 32, 64), (32, 64, 128), (8, 16, 32), (64, 64, 64), (32, 32, 32),
    (16, 16, 32), (32, 64, 64), (64, 128, 128), (8, 32, 64), (16, 64, 128),
    (32, 16, 8), (64, 32, 16), (128, 64, 32), (16, 32, 16), (32, 32, 64),
    (64, 128, 64), (128, 128, 64), (16, 128, 32), (32, 128, 64), (64, 64, 32)
]

# ============================
# FUNCȚII DE ÎNCĂRCARE ȘI PREPROCESARE
# ============================
def extract_info_from_filename(filename):
    parts = filename.split("_")
    label = 1 if parts[1] == "truth" else 0
    speaker_id = parts[2]  # ex: 001
    gender = parts[3][-1]  # F or M
    return label, f"{parts[1]}_{speaker_id}", gender  # speaker key: truth_001

def load_data_and_labels(directory):
    speaker_data = defaultdict(list)
    speaker_genders = dict()

    for file in os.listdir(directory):
        if not file.endswith("_spectrogram.npy"):
            continue
        path = os.path.join(directory, file)
        try:
            data = np.load(path)
            if data.ndim == 2:
                data = np.expand_dims(data, axis=-1)
            data = tf.image.resize(data, [128, 128]).numpy()
        except Exception as e:
            print(f"Eroare la fișierul {file}: {e}")
            continue

        label, speaker_key, gender = extract_info_from_filename(file)
        speaker_data[speaker_key].append((data, label))
        speaker_genders[speaker_key] = gender

    return speaker_data, speaker_genders

# ============================
# MODEL CNN
# ============================
def create_cnn_model(input_shape, filters):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))

    for f in filters:
        model.add(layers.Conv2D(f, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.BatchNormalization())

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ============================
# ÎMPĂRȚIREA PE SPEAKERI
# ============================
def split_speakers(speaker_data, test_ratio=TEST_SPLIT):
    speaker_keys = list(speaker_data.keys())
    random.shuffle(speaker_keys)

    test_size = int(len(speaker_keys) * test_ratio)
    test_speakers = set(speaker_keys[:test_size])
    train_val_speakers = set(speaker_keys[test_size:])

    return test_speakers, train_val_speakers

# ============================
# CONSTRUIREA SETURILOR
# ============================
def build_dataset(speaker_data, speakers):
    X, y = [], []
    for spk in speakers:
        for x, label in speaker_data[spk]:
            X.append(x)
            y.append(label)
    return np.array(X), np.array(y)

# ============================
# LOOP PRINCIPAL
# ============================
def main():
    speaker_data, speaker_genders = load_data_and_labels(SPECTROGRAM_DIR)
    all_speakers = list(speaker_data.keys())

    print(f"Numar total de instante: {sum(len(v) for v in speaker_data.values())}")
    print(f"Numar total de vorbitori unici: {len(all_speakers)}")
    print(f"Distributia genurilor: F: {sum(1 for g in speaker_genders.values() if g == 'F')}, M: {sum(1 for g in speaker_genders.values() if g == 'M')}")

    test_speakers, train_val_speakers = split_speakers(speaker_data)
    X_test, y_test = build_dataset(speaker_data, test_speakers)

    results = []

    for config in tqdm(CNN_CONFIGS, desc="Evaluare configuratii CNN"):
        X_trainval, y_trainval = build_dataset(speaker_data, train_val_speakers)

        kfold = StratifiedKFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=42)
        fold_accuracies = []

        for train_idx, val_idx in kfold.split(X_trainval, y_trainval):
            X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
            y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

            model = create_cnn_model(INPUT_SHAPE, config)
            early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=[early_stop])

            val_preds = model.predict(X_val).flatten() > 0.5
            val_acc = accuracy_score(y_val, val_preds)
            fold_accuracies.append(val_acc)

        # Evaluare pe setul de test
        test_preds = model.predict(X_test).flatten() > 0.5
        test_acc = accuracy_score(y_test, test_preds)

        print(f"Configuratia {config} - Acc. media validare: {np.mean(fold_accuracies):.4f}, Acc. test: {test_acc:.4f}")
        results.append({
            'config': config,
            'val_acc': np.mean(fold_accuracies),
            'test_acc': test_acc
        })

    print("\nRezumat configuratii:")
    for r in results:
        print(f"Config: {r['config']}, Val_Acc: {r['val_acc']:.4f}, Test_Acc: {r['test_acc']:.4f}")

if __name__ == "__main__":
    main()