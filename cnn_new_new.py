import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import random
from sklearn.model_selection import train_test_split

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
    (16, 16, 32), (32, 64, 64), (64, 128, 128), (8, 32, 64), (16, 64, 128)
]

# ============================
# FUNCȚII DE ÎNCĂRCARE ȘI PREPROCESARE
# ============================
def extract_info_from_filename(filename):
    parts = filename.split("_")
    label = 1 if parts[1] == "truth" else 0
    speaker_id = parts[1] + "_" + parts[2]  # ex: truth_001
    gender = parts[3][7]  # F or M
    return label, speaker_id, gender

def load_data_and_labels(directory):
    speaker_data = defaultdict(list)
    speaker_genders = {}
    speaker_labels = {}

    for file in os.listdir(directory):
        if not file.endswith("_spectrogram.npy"):
            continue
            
        try:
            # Încărcare și preprocesare spectrogramă
            spectrogram = np.load(os.path.join(directory, file))
            if spectrogram.ndim == 2:
                spectrogram = np.expand_dims(spectrogram, axis=-1)
            spectrogram = tf.image.resize(spectrogram, [INPUT_SHAPE[0], INPUT_SHAPE[1]]).numpy()
            
            # Extragere metadate
            label, speaker_id, gender = extract_info_from_filename(file)
            
            # Organizare date
            speaker_data[speaker_id].append(spectrogram)
            speaker_genders[speaker_id] = gender
            speaker_labels[speaker_id] = label  # Presupunem că toate instanțele unui speaker au aceeași etichetă
            
        except Exception as e:
            print(f"Eroare la procesarea fișierului {file}: {str(e)}")
            continue

    return speaker_data, speaker_genders, speaker_labels

# ============================
# FUNCȚII DE ÎMPĂRȚIRE STRATIFICATĂ
# ============================
def create_stratified_splits(speaker_genders, speaker_labels, test_size=0.1):
    """Creează împărțire train/test stratificată pe gen și etichetă"""
    # Grupăm speakerii după gen și etichetă
    groups = defaultdict(list)
    for speaker_id in speaker_genders:
        key = (speaker_genders[speaker_id], speaker_labels[speaker_id])
        groups[key].append(speaker_id)
    
    # Selectăm proporțional din fiecare grupă pentru test
    test_speakers = []
    for key in groups:
        speakers = groups[key]
        n_test = max(1, round(len(speakers) * test_size))
        test_speakers.extend(random.sample(speakers, n_test))
    
    return set(test_speakers)

def check_data_distribution(speaker_labels, speaker_genders, test_speakers):
    print("\n=== Distribuția pe seturi ===")
    
    # Calculăm distribuția pentru train+val și test
    for set_name, speakers in [("Train+Val", set(speaker_labels.keys()) - set(test_speakers)), 
                            ("Test", test_speakers)]:
        truth = sum(1 for s in speakers if speaker_labels[s] == 1)
        lie = sum(1 for s in speakers if speaker_labels[s] == 0)
        female = sum(1 for s in speakers if speaker_genders[s] == 'F')
        male = sum(1 for s in speakers if speaker_genders[s] == 'M')
        
        print(f"{set_name}:")
        print(f"  Truth: {truth} ({truth/(truth+lie):.1%}) | Lie: {lie} ({lie/(truth+lie):.1%})")
        print(f"  F: {female} ({female/(female+male):.1%}) | M: {male} ({male/(female+male):.1%})")

def check_fold_distribution(X, y, groups, train_idx, val_idx, fold_num, speaker_labels, speaker_genders):
    """Verifică distribuția claselor și genurilor în fiecare fold"""
    # Extrage speakerii unici pentru fold
    train_speakers = np.unique(groups[train_idx])
    val_speakers = np.unique(groups[val_idx])
    
    # Calculează distribuția
    def get_stats(speakers):
        truth = sum(1 for s in speakers if speaker_labels[s] == 1)
        lie = sum(1 for s in speakers if speaker_labels[s] == 0)
        f = sum(1 for s in speakers if speaker_genders[s] == 'F')
        m = sum(1 for s in speakers if speaker_genders[s] == 'M')
        return truth, lie, f, m
    
    train_truth, train_lie, train_f, train_m = get_stats(train_speakers)
    val_truth, val_lie, val_f, val_m = get_stats(val_speakers)
    
    print(f"\nFold {fold_num} Distribution:")
    print("Train: Truth={} ({:.1%}), Lie={} ({:.1%}) | F={} ({:.1%}), M={} ({:.1%})".format(
        train_truth, train_truth/(train_truth+train_lie), 
        train_lie, train_lie/(train_truth+train_lie),
        train_f, train_f/(train_f+train_m),
        train_m, train_m/(train_f+train_m)
    ))
    print("Val:   Truth={} ({:.1%}), Lie={} ({:.1%}) | F={} ({:.1%}), M={} ({:.1%})".format(
        val_truth, val_truth/(val_truth+val_lie), 
        val_lie, val_lie/(val_truth+val_lie),
        val_f, val_f/(val_f+val_m),
        val_m, val_m/(val_f+val_m)
    ))
# ============================
# MODEL CNN
# ============================
def create_cnn_model(input_shape, filters):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    
    for f in filters:
        model.add(layers.Conv2D(f, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.3))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    return model

# ============================
# FUNCȚII DE EVALUARE
# ============================
def evaluate_model(model, X, y):
    y_pred = (model.predict(X) > 0.5).astype(int)
    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, target_names=['lie', 'truth'], output_dict=True)
    return acc, report

# ============================
# FUNCȚIA PRINCIPALĂ
# ============================
def main():
    # Încărcare date
    speaker_data, speaker_genders, speaker_labels = load_data_and_labels(SPECTROGRAM_DIR)
    
    # Statistici inițiale
    print(f"Număr total de vorbitori: {len(speaker_data)}")
    print(f"Distribuție genuri: F={sum(1 for g in speaker_genders.values() if g == 'F')}, M={sum(1 for g in speaker_genders.values() if g == 'M')}")
    print(f"Distribuție etichete: Truth={sum(1 for l in speaker_labels.values() if l == 1)}, Lie={sum(1 for l in speaker_labels.values() if l == 0)}")
    
    # Creare split stratificat
    test_speakers = create_stratified_splits(speaker_genders, speaker_labels, TEST_SPLIT)
    train_val_speakers = set(speaker_data.keys()) - test_speakers
    check_data_distribution(speaker_labels, speaker_genders, test_speakers)
    
    # Construire seturi de date
    def build_dataset(speakers):
        X, y, groups = [], [], []
        for speaker_id in speakers:
            for spectrogram in speaker_data[speaker_id]:
                X.append(spectrogram)
                y.append(speaker_labels[speaker_id])
                groups.append(speaker_id)
        return np.array(X), np.array(y), np.array(groups)
    
    X_test, y_test, _ = build_dataset(test_speakers)
    X_trainval, y_trainval, groups_trainval = build_dataset(train_val_speakers)
    
    # Normalizare
    X_trainval = X_trainval / 255.0
    X_test = X_test / 255.0
    
    # Antrenare cu cross-validation
    results = []
    group_kfold = GroupKFold(n_splits=KFOLD_SPLITS)
    fold = 0
    for config in tqdm(CNN_CONFIGS, desc="Evaluare configurații"):
        fold_metrics = {'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': []}
        
        for train_idx, val_idx in group_kfold.split(X_trainval, y_trainval, groups_trainval):
            X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
            y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

            check_fold_distribution(X_trainval, y_trainval, groups_trainval, train_idx, val_idx, fold+1, speaker_labels, speaker_genders)

            # Înainte de antrenare, verificați distribuția claselor
            print("\nClass distribution in training set:")
            print(f"Truth: {sum(y_train)} samples")
            print(f"Lie: {len(y_train)-sum(y_train)} samples")

            # Dacă există dezechilibru, calculați class weights
            class_weights = {0: len(y_train)/(2*(len(y_train)-sum(y_train))), 
                            1: len(y_train)/(2*sum(y_train))}
            print("Class weights:", class_weights)
            
            # Creare și antrenare model
            model = create_cnn_model(INPUT_SHAPE, config)
            early_stop = callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            )
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=0,
                callbacks=[early_stop]
            )
            
            # Evaluare
            val_acc, val_report = evaluate_model(model, X_val, y_val)
            fold_metrics['val_acc'].append(val_acc)
            fold_metrics['val_precision'].append(val_report['weighted avg']['precision'])
            fold_metrics['val_recall'].append(val_report['weighted avg']['recall'])
            fold_metrics['val_f1'].append(val_report['weighted avg']['f1-score'])
        
        # Evaluare pe setul de test
        test_acc, test_report = evaluate_model(model, X_test, y_test)
        fold += 1

        # Salvare rezultate
        results.append({
            'config': config,
            'val_acc': np.mean(fold_metrics['val_acc']),
            'val_precision': np.mean(fold_metrics['val_precision']),
            'val_recall': np.mean(fold_metrics['val_recall']),
            'val_f1': np.mean(fold_metrics['val_f1']),
            'test_acc': test_acc,
            'test_precision': test_report['weighted avg']['precision'],
            'test_recall': test_report['weighted avg']['recall'],
            'test_f1': test_report['weighted avg']['f1-score']
        })
        
        print(f"\nConfigurație {config}:")
        print(f"  Validare - Acc: {np.mean(fold_metrics['val_acc']):.4f}, F1: {np.mean(fold_metrics['val_f1']):.4f}")
        print(f"  Test - Acc: {test_acc:.4f}, F1: {test_report['weighted avg']['f1-score']:.4f}")
    
    # Afișare rezultate finale
    print("\nRezultate finale:")
    results_df = pd.DataFrame(results)
    print(results_df.sort_values(by='val_f1', ascending=False).to_string())
    
    # Salvare rezultate
    results_df.to_csv('cnn_results_stratified.csv', index=False)

if __name__ == "__main__":
    main()