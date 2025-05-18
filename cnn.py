import os
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm
import random
from collections import defaultdict

# Configurații
SPECTROGRAMS_DIR = 'spectrograms'
TEST_SIZE = 0.1
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Încărcarea datelor și organizarea pe vorbitori
def load_data():
    speaker_data = defaultdict(list)
    speaker_labels = defaultdict(list)
    speaker_genders = {}
    
    files = [f for f in os.listdir(SPECTROGRAMS_DIR) if f.endswith('.npy')]
    
    for file in files:
        parts = file.split('_')
        lie_truth = parts[1]  # 'lie' sau 'truth'
        speaker_id = parts[2]  # numărul vorbitorului
        gender = parts[3][0]  # 'F' sau 'M'
        
        # Încărcare spectrogramă
        spectrogram = np.load(os.path.join(SPECTROGRAMS_DIR, file))
        
        # Adăugare la dicționarul vorbitorului
        speaker_data[speaker_id].append(spectrogram)
        speaker_labels[speaker_id].append(0 if lie_truth == 'lie' else 1)
        speaker_genders[speaker_id] = gender
    
    return speaker_data, speaker_labels, speaker_genders

# Procesare date
def prepare_data(speaker_data, speaker_labels):
    X = []
    y = []
    speaker_ids = []
    
    for speaker_id in speaker_data:
        X.extend(speaker_data[speaker_id])
        y.extend(speaker_labels[speaker_id])
        speaker_ids.extend([speaker_id] * len(speaker_data[speaker_id]))
    
    return np.array(X), np.array(y), np.array(speaker_ids)

# Împărțirea datelor în train și test, păstrând vorbitorii separați
def split_data(X, y, speaker_ids, speaker_genders, test_size=0.1):
    unique_speakers = list(set(speaker_ids))
    unique_speakers.sort()
    
    # Asigurăm proporția de gen în setul de test
    female_speakers = [s for s in unique_speakers if speaker_genders[s] == 'F']
    male_speakers = [s for s in unique_speakers if speaker_genders[s] == 'M']
    
    # Calculăm numărul de vorbitori de test pentru fiecare gen
    # Asigurăm că avem cel puțin 1 vorbitor și cel mult numărul disponibil
    n_test_female = max(1, min(len(female_speakers) - 1, 
                             round(len(female_speakers) * test_size)))
    n_test_male = max(1, min(len(male_speakers) - 1, 
                           round(len(male_speakers) * test_size)))
    
    # Dacă nu avem suficiente vorbitori de un gen, ajustăm
    if len(female_speakers) == 0:
        n_test_female = 0
    if len(male_speakers) == 0:
        n_test_male = 0
    
    # Selectăm aleatoriu vorbitori pentru test
    test_speakers = []
    if n_test_female > 0 and len(female_speakers) > 0:
        test_speakers.extend(random.sample(female_speakers, n_test_female))
    if n_test_male > 0 and len(male_speakers) > 0:
        test_speakers.extend(random.sample(male_speakers, n_test_male))
    
    # Dacă nu am putut selecta niciun vorbitor, alegem unul aleatoriu
    if not test_speakers and unique_speakers:
        test_speakers.append(random.choice(unique_speakers))
    
    # Creăm masca pentru setul de test
    test_mask = np.isin(speaker_ids, test_speakers)
    
    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]
    speaker_ids_train = speaker_ids[~test_mask]
    
    return X_train, X_test, y_train, y_test, speaker_ids_train

# Creare model CNN
def create_cnn_model(input_shape, config):
    model = Sequential()
    
    # Strat de normalizare batch
    model.add(BatchNormalization(input_shape=input_shape))
    
    # Straturi convoluționale
    for i in range(config['n_conv_layers']):
        model.add(Conv2D(
            filters=config['filters'][i],
            kernel_size=config['kernel_sizes'][i],
            activation='relu',
            padding='same',
            kernel_regularizer=l2(config['l2_reg']) if config['l2_reg'] > 0 else None
        ))
        model.add(MaxPooling2D(pool_size=config['pool_sizes'][i]))
        model.add(Dropout(config['dropout_rates'][i]))
        if config['use_batch_norm']:
            model.add(BatchNormalization())
    
    model.add(Flatten())
    
    # Straturi dense
    for units in config['dense_units']:
        model.add(Dense(
            units,
            activation='relu',
            kernel_regularizer=l2(config['l2_reg']) if config['l2_reg'] > 0 else None
        ))
        model.add(Dropout(config['dense_dropout']))
        if config['use_batch_norm']:
            model.add(BatchNormalization())
    
    # Strat de ieșire
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = Adam(learning_rate=config['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Configurații CNN de testat
def generate_cnn_configs():
    configs = []
    
    # Configurații de bază
    base_configs = [
        {
            'n_conv_layers': 2,
            'filters': [32, 64],
            'kernel_sizes': [(3, 3), (3, 3)],
            'pool_sizes': [(2, 2), (2, 2)],
            'dropout_rates': [0.25, 0.25],
            'dense_units': [64],
            'dense_dropout': 0.5,
            'learning_rate': 0.001,
            'l2_reg': 0.001,
            'use_batch_norm': True,
            'epochs': 30,
            'batch_size': 32
        },
        {
            'n_conv_layers': 3,
            'filters': [32, 64, 128],
            'kernel_sizes': [(3, 3), (3, 3), (3, 3)],
            'pool_sizes': [(2, 2), (2, 2), (2, 2)],
            'dropout_rates': [0.3, 0.3, 0.3],
            'dense_units': [128, 64],
            'dense_dropout': 0.5,
            'learning_rate': 0.0001,
            'l2_reg': 0.01,
            'use_batch_norm': True,
            'epochs': 50,
            'batch_size': 16
        },
        {
            'n_conv_layers': 2,
            'filters': [64, 128],
            'kernel_sizes': [(5, 5), (3, 3)],
            'pool_sizes': [(2, 2), (2, 2)],
            'dropout_rates': [0.4, 0.4],
            'dense_units': [128],
            'dense_dropout': 0.6,
            'learning_rate': 0.0005,
            'l2_reg': 0.005,
            'use_batch_norm': False,
            'epochs': 40,
            'batch_size': 32
        }
    ]
    
    # Generăm variații
    for base in base_configs:
        # Variații în numărul de filtre
        for filters in [[16, 32], [32, 64], [64, 128], [128, 256]]:
            if len(filters) >= base['n_conv_layers']:
                new_config = base.copy()
                new_config['filters'] = filters[:new_config['n_conv_layers']]
                configs.append(new_config)
        
        # Variații în dimensiunea kernel-ului
        for kernels in [[(3, 3), (3, 3)], [(5, 5), (3, 3)], [(7, 7), (5, 5)]]:
            if len(kernels) >= base['n_conv_layers']:
                new_config = base.copy()
                new_config['kernel_sizes'] = kernels[:new_config['n_conv_layers']]
                configs.append(new_config)
    
    # Eliminăm duplicatele
    unique_configs = []
    seen = set()
    
    for config in configs:
        config_hash = str(sorted(config.items()))
        if config_hash not in seen:
            seen.add(config_hash)
            unique_configs.append(config)
    
    return unique_configs[:20]  # Returnăm doar primele 20 de configurații unice

# Antrenare și evaluare cu cross-validation
def evaluate_model(X_train, y_train, speaker_ids_train, config):
    group_kfold = GroupKFold(n_splits=5)
    accuracies = []
    
    for train_index, val_index in group_kfold.split(X_train, y_train, groups=speaker_ids_train):
        X_tr, X_val = X_train[train_index], X_train[val_index]
        y_tr, y_val = y_train[train_index], y_train[val_index]
        
        # Verificăm dimensiunile spectrogramelor
        input_shape = X_tr.shape[1:]
        
        model = create_cnn_model(input_shape, config)
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            callbacks=[early_stopping],
            verbose=0
        )
        
        y_pred = model.predict(X_val, verbose=0)
        y_pred_class = (y_pred > 0.5).astype(int)
        acc = accuracy_score(y_val, y_pred_class)
        accuracies.append(acc)
    
    return np.mean(accuracies)

# Funcția principală
def main():
    # Încărcare date
    print("Încărcare date...")
    speaker_data, speaker_labels, speaker_genders = load_data()
    X, y, speaker_ids = prepare_data(speaker_data, speaker_labels)
    
    # Adăugare dimensiune de canal pentru CNN (dacă nu există deja)
    if len(X.shape) == 3:  # Dacă e (samples, height, width)
        X = np.expand_dims(X, axis=-1)  # Transformă în (samples, height, width, 1)
    
    # Împărțire date
    print("Împărțire date în train și test...")
    X_train, X_test, y_train, y_test, speaker_ids_train = split_data(
        X, y, speaker_ids, speaker_genders, TEST_SIZE
    )
    
    # Generare configurații
    print("Generare configurații CNN...")
    cnn_configs = generate_cnn_configs()
    print(f"Număr de configurații generate: {len(cnn_configs)}")
    
    # Antrenare și evaluare modele
    print(f"Antrenare și evaluare {len(cnn_configs)} modele CNN...")
    best_acc = 0
    best_config = None
    best_model = None
    
    for config in tqdm(cnn_configs, desc="Modele CNN"):
        try:
            acc = evaluate_model(X_train, y_train, speaker_ids_train, config)
            
            if acc > best_acc:
                best_acc = acc
                best_config = config
                
                # Antrenăm modelul pe întreg setul de antrenare pentru evaluarea finală
                input_shape = X_train.shape[1:]
                best_model = create_cnn_model(input_shape, best_config)
                best_model.fit(
                    X_train, y_train,
                    epochs=best_config['epochs'],
                    batch_size=best_config['batch_size'],
                    verbose=0
                )
                
                print(f"\nConfigurație îmbunătățită - Acuratețe: {acc:.4f}")
                print(f"Configurație: {best_config}")
        except Exception as e:
            print(f"\nEroare la antrenarea configurării {config}: {str(e)}")
            continue
    
    # Evaluare pe setul de test
    if best_model is not None:
        y_pred_test = best_model.predict(X_test, verbose=0)
        y_pred_test_class = (y_pred_test > 0.5).astype(int)
        test_acc = accuracy_score(y_test, y_pred_test_class)
        
        print("\nRezultate finale:")
        print(f"Cea mai bună configurație: {best_config}")
        print(f"Acuratețe medie cross-validation: {best_acc:.4f}")
        print(f"Acuratețe pe setul de test: {test_acc:.4f}")
        
        # Salvăm modelul
        best_model.save('best_cnn_model.h5')
        print("Model salvat ca 'best_cnn_model.h5'")
    else:
        print("Niciun model valid nu a putut fi antrenat.")

if __name__ == "__main__":
    main()
