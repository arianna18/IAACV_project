import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
import tensorflow.image as tf_image
from collections import defaultdict

# Incarca spectrogramele si etichetele

def load_data_and_labels(spectrogram_dir):
    spectrograms = []
    labels = []
    speaker_ids = []
    genders = []

    for file in os.listdir(spectrogram_dir):
        if file.endswith('_spectrogram.npy'):
            spectrogram = np.load(os.path.join(spectrogram_dir, file))
            if spectrogram.ndim == 2:
                spectrogram = np.expand_dims(spectrogram, axis=-1)
            spectrogram = tf_image.resize(spectrogram, [128, 128]).numpy()
            if 'truth' in file:
                label = 1
            elif 'lie' in file:
                label = 0
            else:
                continue

            base = file.split('_spectrogram')[0]
            speaker_id = '_'.join(base.split('_')[1:3])  # ex: lie_005 sau truth_023

            gender = 'F' if 'speakerF' in file else 'M'

            spectrograms.append(spectrogram)
            labels.append(label)
            speaker_ids.append(speaker_id)
            genders.append(gender)

    spectrograms = np.array(spectrograms)
    labels = np.array(labels)
    speaker_ids = np.array(speaker_ids)
    genders = np.array(genders)

    print(f"Numar total de instante: {len(labels)}")
    print(f"Numar total de vorbitori unici: {len(np.unique(speaker_ids))}")
    print(f"Distributie genuri: F: {np.sum(genders == 'F')}, M: {np.sum(genders == 'M')}")

    return spectrograms, labels, speaker_ids, genders

# Creeaza model CNN configurabil

def create_cnn_model(input_shape, conv_filters):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))

    for filters in conv_filters:
        model.add(layers.Conv2D(filters, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 5-fold cross-validation stratificat pe speaker

def train_with_cross_validation(X, y, speaker_ids, conv_configs):
    unique_speakers = np.unique(speaker_ids)
    speaker_to_label = {spk: y[np.where(speaker_ids == spk)[0][0]] for spk in unique_speakers}
    speaker_labels = np.array([speaker_to_label[spk] for spk in unique_speakers])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for conv_filters in conv_configs:
        print(f"\nEvaluare model cu configuratia: {conv_filters}")
        acc_scores = []

        for train_index, val_index in skf.split(unique_speakers, speaker_labels):
            train_speakers = unique_speakers[train_index]
            val_speakers = unique_speakers[val_index]

            train_mask = np.isin(speaker_ids, train_speakers)
            val_mask = np.isin(speaker_ids, val_speakers)

            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]

            model = create_cnn_model(X.shape[1:], conv_filters)

            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=20, batch_size=8, verbose=0, callbacks=[early_stopping])

            _, acc = model.evaluate(X_val, y_val, verbose=0)
            acc_scores.append(acc)

        mean_acc = np.mean(acc_scores)
        print(f"Acc: {mean_acc:.4f}")


# Main

def main():
    spectrogram_dir = r"D:\Iulia\ETTI\Master\Semestrul 2\IAACV_project\spectrograms"
    X, y, speaker_ids, genders = load_data_and_labels(spectrogram_dir)
    X = X / np.max(X)  # Normalizare la [0, 1]

    conv_configurations = [
        [16, 32], [32, 64], [64, 128], [16, 32, 64], [32, 64, 128],
        [32, 32, 64], [64, 64, 128], [16, 32, 32, 64], [32, 64, 64, 128],
        [64, 64, 64, 128], [16, 32, 64, 128], [32, 64, 128, 128],
        [32, 32, 64, 64], [16, 32, 32], [32, 32, 32], [32, 64, 64],
        [64, 64, 64], [32, 32, 64, 64, 128], [16, 32, 64, 64, 128],
        [16, 16, 32, 32, 64], [32, 64, 128, 256]
    ]

    train_with_cross_validation(X, y, speaker_ids, conv_configurations)

if __name__ == "__main__":
    main()