import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================
# CONFIGURARE GLOBALĂ
# ============================
SPECTROGRAM_DIR = "spectrograms"
INPUT_SHAPE = (128, 128, 1)
TEST_SPLIT = 0.1
KFOLD_SPLITS = 5
EPOCHS = 200  # Creștem numărul de epoci
BATCH_SIZE = 64
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

def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_callback = callbacks.LearningRateScheduler(lr_scheduler)

def load_data_and_labels(directory):
    speaker_data = defaultdict(list)
    speaker_genders = {}
    speaker_labels = {}

    for file in os.listdir(directory):
        if not file.endswith("_spectrogram.npy"):
            continue
            
        try:
            spectrogram = np.load(os.path.join(directory, file))
            if spectrogram.ndim == 2:
                spectrogram = np.expand_dims(spectrogram, axis=-1)
            spectrogram = tf.image.resize(spectrogram, [INPUT_SHAPE[0], INPUT_SHAPE[1]]).numpy()
            
            label, speaker_id, gender = extract_info_from_filename(file)
            
            speaker_data[speaker_id].append(spectrogram)
            speaker_genders[speaker_id] = gender
            speaker_labels[speaker_id] = label
            
        except Exception as e:
            print(f"Eroare la procesarea fișierului {file}: {str(e)}")
            continue

    return speaker_data, speaker_genders, speaker_labels

# ============================
# FUNCȚII DE ÎMPĂRȚIRE STRATIFICATĂ
# ============================
def create_stratified_splits(speaker_genders, speaker_labels, test_size=0.1):
    groups = defaultdict(list)
    for speaker_id in speaker_genders:
        key = (speaker_genders[speaker_id], speaker_labels[speaker_id])
        groups[key].append(speaker_id)
    
    test_speakers = []
    for key in groups:
        speakers = groups[key]
        n_test = max(1, round(len(speakers) * test_size))
        test_speakers.extend(random.sample(speakers, n_test))
    
    return set(test_speakers)

def check_data_distribution(speaker_labels, speaker_genders, test_speakers):
    print("\n=== Distribuția pe seturi ===")
    
    for set_name, speakers in [("Train+Val", set(speaker_labels.keys()) - set(test_speakers)), 
                            ("Test", test_speakers)]:
        truth = sum(1 for s in speakers if speaker_labels[s] == 1)
        lie = sum(1 for s in speakers if speaker_labels[s] == 0)
        female = sum(1 for s in speakers if speaker_genders[s] == 'F')
        male = sum(1 for s in speakers if speaker_genders[s] == 'M')
        
        print(f"{set_name}:")
        print(f"  Truth: {truth} ({truth/(truth+lie):.1%}) | Lie: {lie} ({lie/(truth+lie):.1%})")
        print(f"  F: {female} ({female/(female+male):.1%}) | M: {male} ({male/(female+male):.1%})")

def calculate_sample_weights(y, genders, gender_weight=0.3):
    class_weights = {0: len(y)/(2*(len(y)-sum(y))), 1: len(y)/(2*sum(y))}
    gender_weights = {'F': 0.5/np.mean([g == 'F' for g in genders]),
                     'M': 0.5/np.mean([g == 'M' for g in genders])}
    
    sample_weights = []
    for label, gender in zip(y, genders):
        weight = (gender_weight * gender_weights[gender]) + ((1-gender_weight) * class_weights[label])
        sample_weights.append(weight)
    
    return np.array(sample_weights)

# ============================
# MODEL CNN ÎMBUNĂTĂȚIT
# ============================
def create_improved_cnn(input_shape, filters):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    
    for f in filters:
        model.add(layers.Conv2D(f, (3, 3), padding='same',
                              kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)))
        model.add(layers.LeakyReLU(alpha=0.1))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.5))
    
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(64, activation='relu',
                         kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(0.6))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

# ============================
# FUNCȚII DE EVALUARE
# ============================
def evaluate_model(model, X, y, genders=None):
    y_pred = (model.predict(X) > 0.5).astype(int)
    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, target_names=['lie', 'truth'], output_dict=True)
    
    metrics = {
        'acc': acc,
        'report': report,
        'f1': report['weighted avg']['f1-score']
    }
    
    if genders is not None:
        gender_metrics = {}
        for gender in ['F', 'M']:
            mask = np.array(genders) == gender
            if sum(mask) > 0:
                gender_report = classification_report(y[mask], y_pred[mask], 
                                                    target_names=['lie', 'truth'], 
                                                    output_dict=True)
                gender_metrics[gender] = {
                    'acc': accuracy_score(y[mask], y_pred[mask]),
                    'f1': gender_report['weighted avg']['f1-score']
                }
        metrics['gender_metrics'] = gender_metrics
    
    return metrics

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
        X, y, groups, genders = [], [], [], []
        for speaker_id in speakers:
            for spectrogram in speaker_data[speaker_id]:
                X.append(spectrogram)
                y.append(speaker_labels[speaker_id])
                groups.append(speaker_id)
                genders.append(speaker_genders[speaker_id])
        return np.array(X), np.array(y), np.array(groups), np.array(genders)
    
    X_test, y_test, test_groups, test_genders = build_dataset(test_speakers)
    X_trainval, y_trainval, groups_trainval, genders_trainval = build_dataset(train_val_speakers)
    
    # Normalizare
    X_trainval = X_trainval / 255.0
    X_test = X_test / 255.0

    all_histories = []
    
    # Antrenare cu cross-validation
    results = []
    group_kfold = StratifiedGroupKFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=42)
    
    for config in tqdm(CNN_CONFIGS, desc="Evaluare configurații"):
        fold_metrics = {'val_acc': [], 'val_f1': [], 'val_auc': []}
        gender_metrics = defaultdict(list)
        
        for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X_trainval, y_trainval, groups_trainval)):
            X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
            y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]
            train_groups = groups_trainval[train_idx]
            val_groups = groups_trainval[val_idx]
            
            # Calcul ponderi
            train_genders = [speaker_genders[g] for g in train_groups]
            sample_weights = calculate_sample_weights(y_train, train_genders)
            
            # Creare și antrenare model
            model = create_improved_cnn(INPUT_SHAPE, config)
            
            early_stop = callbacks.EarlyStopping(
                monitor='val_auc',
                patience=20,
                mode='max',
                restore_best_weights=True
            )
            
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
            
            # Augmentare date
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.15,
                height_shift_range=0.15,
                zoom_range=0.15,
                horizontal_flip=True,
                fill_mode='constant'
            )
            
            history = model.fit(
                datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, sample_weight=sample_weights),
                validation_data=(X_val, y_val),
                epochs=EPOCHS,
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )

            all_histories.append({
                'config': config,
                'fold': fold,
                'accuracy': history.history['accuracy'],
                'val_accuracy': history.history['val_accuracy']
            })

            # Evaluare
            val_metrics = evaluate_model(model, X_val, y_val, [speaker_genders[g] for g in val_groups])
            fold_metrics['val_acc'].append(val_metrics['acc'])
            fold_metrics['val_f1'].append(val_metrics['f1'])
            fold_metrics['val_auc'].append(np.max(history.history['val_auc']))
            
            # Salvare metrici pe gen
            if 'gender_metrics' in val_metrics:
                for gender, metrics in val_metrics['gender_metrics'].items():
                    gender_metrics[f'val_{gender}_acc'].append(metrics['acc'])
                    gender_metrics[f'val_{gender}_f1'].append(metrics['f1'])
        
        # Evaluare pe setul de test
        test_metrics = evaluate_model(model, X_test, y_test, test_genders)
        
        # Salvare rezultate
        result = {
            'config': config,
            'val_acc': np.mean(fold_metrics['val_acc']),
            'val_f1': np.mean(fold_metrics['val_f1']),
            'val_auc': np.mean(fold_metrics['val_auc']),
            'test_acc': test_metrics['acc'],
            'test_f1': test_metrics['f1']
        }
        
        # Adăugare metrici pe gen
        for gender in ['F', 'M']:
            if f'val_{gender}_acc' in gender_metrics:
                result[f'val_{gender}_acc'] = np.mean(gender_metrics[f'val_{gender}_acc'])
                result[f'val_{gender}_f1'] = np.mean(gender_metrics[f'val_{gender}_f1'])
            if 'gender_metrics' in test_metrics and gender in test_metrics['gender_metrics']:
                result[f'test_{gender}_acc'] = test_metrics['gender_metrics'][gender]['acc']
                result[f'test_{gender}_f1'] = test_metrics['gender_metrics'][gender]['f1']
        
        results.append(result)
        
        # Afișare rezultate folduri
        print(f"\nConfigurație {config}:")
        print(f"  Validare - Acc: {result['val_acc']:.4f}, F1: {result['val_f1']:.4f}, AUC: {result['val_auc']:.4f}")
        print(f"  Test - Acc: {result['test_acc']:.4f}, F1: {result['test_f1']:.4f}")
        
        # Afișare performanță pe gen
        for gender in ['F', 'M']:
            if f'val_{gender}_acc' in result:
                print(f"  {gender} - Val Acc: {result[f'val_{gender}_acc']:.4f}, Test Acc: {result.get(f'test_{gender}_acc', np.nan):.4f}")
    
    # Afișare rezultate finale
    print("\nRezultate finale:")
    results_df = pd.DataFrame(results)
    print(results_df.sort_values(by='val_f1', ascending=False).to_string())
    
    # Salvare rezultate
    results_df.to_csv('cnn_results_improved.csv', index=False)

    results_sorted = sorted(results, key=lambda x: x['test_f1'], reverse=True)

    import json
    with open("train_histories.json", "w") as f:
        json.dump(all_histories, f)

    print("\n=== Top 3 Configurații CNN după test F1 ===")
    for i, r in enumerate(results_sorted[:3]):
        print(f"\nConfigurație {i+1}: {r['config']}")
        print(f"  Val Acc:  {r['val_acc']:.4f} | Val F1:  {r['val_f1']:.4f} | Val AUC: {r['val_auc']:.4f}")
        print(f"  Test Acc: {r['test_acc']:.4f} | Test F1: {r['test_f1']:.4f}")
        for gender in ['F', 'M']:
            acc = r.get(f'test_{gender}_acc', None)
            f1 = r.get(f'test_{gender}_f1', None)
            if acc is not None and f1 is not None:
                print(f"    {gender} - Acc: {acc:.4f} | F1: {f1:.4f}")

if __name__ == "__main__":
    main()