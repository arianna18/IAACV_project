import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# Configurație
INPUT_CSV = 'global_norm.csv'  
OUTPUT_CSV = 'SVM_results2.csv'
SVM_KERNELS = ['linear', 'poly', 'rbf']
CS = [1, 0.1, 0.01]

def load_and_prepare_data(filepath):
    """Încarcă datele și pregătește caracteristicile și etichetele"""
    data = pd.read_csv(filepath)
    
    # Pregătește caracteristicile (features NFCC și F0)
    feature_cols = [col for col in data.columns if col.startswith('NFCC') or col.startswith('F0')]
    X = data[feature_cols].values
    
    # Pregătește etichetele
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data['label'])
    Kclass = len(label_encoder.classes_)
    
    # Grupurile (vorbitorii) și genul pentru stratificare
    groups = data['speaker'].values
    gender = data['gender'].values
    
    # Vector de stratificare combinând genul și eticheta
    stratify = np.array([f"{g}_{l}" for g, l in zip(gender, y)])
    
    return X, y, groups, stratify, Kclass

def evaluate_model(model, X_train, y_train, X_val, y_val, Kclass):
    """Evaluează modelul și returnează metrici"""
    # Metrici antrenare
    y_pred_train = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)
    train_bal_acc = balanced_accuracy_score(y_train, y_pred_train)
    
    # Metrici validare
    y_pred_val = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_pred_val)
    val_bal_acc = balanced_accuracy_score(y_val, y_pred_val)
    
    return {
        'train_acc': train_acc,
        'train_bal_acc': train_bal_acc,
        'val_acc': val_acc,
        'val_bal_acc': val_bal_acc
    }

def display_csv_results(csv_path):
    """Afișează conținutul CSV-ului într-un format citibil"""
    try:
        results = pd.read_csv(csv_path)
        print("\n" + "="*50)
        print("REZULTATELE DIN FIȘIERUL CSV:")
        print("="*50)
        print(results.to_string(index=False))
        print("="*50 + "\n")
    except Exception as e:
        print(f"\nEroare la afișarea rezultatelor: {e}")

def run_experiment():
    # Încarcă datele
    X, y, groups, stratify, Kclass = load_and_prepare_data(INPUT_CSV)
    
    # Pregătește stocarea rezultatelor
    results = []
    
    for kernel in SVM_KERNELS:
        for C in CS:
            print(f"\nAntrenare SVM cu kernel={kernel}, C={C}")
            
            fold_metrics = {
                'train_acc': [],
                'train_bal_acc': [],
                'val_acc': [],
                'val_bal_acc': []
            }
            
            # GroupKFold pentru independența vorbitorilor
            group_kfold = GroupKFold(n_splits=5)
            
            for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X, y, groups)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Antrenează modelul
                model = SVC(C=C, kernel=kernel, random_state=42)
                model.fit(X_train, y_train)
                
                # Evaluează
                metrics = evaluate_model(model, X_train, y_train, X_val, y_val, Kclass)
                
                # Stochează metrici pentru fold
                for key in metrics:
                    fold_metrics[key].append(metrics[key])
                
                print(f"Fold {fold+1}: Acc Validare={metrics['val_acc']:.2f}, Acc Echilibrată Validare={metrics['val_bal_acc']:.2f}")
            
            # Calculează metrici medii peste fold-uri
            avg_metrics = {
                'kernel': kernel,
                'C': C,
                'avg_train_acc': np.mean(fold_metrics['train_acc']),
                'avg_train_bal_acc': np.mean(fold_metrics['train_bal_acc']),
                'avg_val_acc': np.mean(fold_metrics['val_acc']),
                'avg_val_bal_acc': np.mean(fold_metrics['val_bal_acc']),
                'std_val_acc': np.std(fold_metrics['val_acc']),
                'std_val_bal_acc': np.std(fold_metrics['val_bal_acc'])
            }
            
            results.append(avg_metrics)
            
            print(f"\nRezumat pentru {kernel}, C={C}:")
            print(f"Acuratețe Antrenare: {avg_metrics['avg_train_acc']:.3f}")
            print(f"Acuratețe Echilibrată Antrenare: {avg_metrics['avg_train_bal_acc']:.3f}")
            print(f"Acuratețe Validare: {avg_metrics['avg_val_acc']:.3f} ± {avg_metrics['std_val_acc']:.3f}")
            print(f"Acuratețe Echilibrată Validare: {avg_metrics['avg_val_bal_acc']:.3f} ± {avg_metrics['std_val_bal_acc']:.3f}")
    
    # Salvează rezultatele în CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nRezultate salvate în {OUTPUT_CSV}")
    
    # Afișează conținutul CSV-ului
    display_csv_results(OUTPUT_CSV)
    
    return results_df

# Rulează experimentul și afișează rezultatele
final_results = run_experiment()