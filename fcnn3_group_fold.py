#!/usr/bin/env python3
"""
Random search over 100 FCNN configurations on MFCC+F0 features, using speaker- and gender-aware StratifiedGroupKFold splits.
- Loads CSV and parses label, speaker, gender.
- Splits speakers into test (10%) and train_val (90%) via StratifiedGroupKFold (groups=speaker, stratify=gender).
- Splits train_val into train (80%) and val (20%) via another StratifiedGroupKFold.
- Standardizes features on train set.
- Samples hyperparameters for each trial, trains model with early stopping.
- Records and prints train, val (best), and test accuracy per trial.
- Saves all results to `hyperparam_results.csv`.
- Plots accuracy curves for the best configuration to `best_accuracy_plot.png`.
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, InputLayer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ---- Settings ----
CSV_PATH    = 'signal_features_global_normalized.csv'
TEST_PCT    = 0.10
VAL_PCT     = 0.20
RANDOM_STATE= 42
N_TRIALS    = 100
EPOCHS      = 100
PATIENCE    = 10
RESULTS_CSV = 'hyperparam_results_group.csv'
BEST_PLOT   = 'best_accuracy_plot_group.png'

# ---- Load and parse metadata ----
df = pd.read_csv(CSV_PATH)
auto_fn = next((c for c in df.columns if 'file' in c.lower() or 'name' in c.lower()), None)
if not auto_fn:
    raise RuntimeError('Filename column not found')
pat_label  = re.compile(r"(?i)(truth|lie)[ _]?(\d{1,3})")
pat_gender = re.compile(r"speaker([MFmf])")
labels, speakers, genders = [], [], []
for fn in df[auto_fn].astype(str):
    m = pat_label.search(fn)
    lab = f"{m.group(1).lower()}_{int(m.group(2)):03d}"
    labels.append(1 if lab.startswith('lie') else 0)
    speakers.append(lab)
    mg = pat_gender.search(fn)
    genders.append(mg.group(1).upper() if mg else 'U')
df['y'] = labels
df['speaker'] = speakers
df['gender']  = genders
meta = {auto_fn, 'y', 'speaker', 'gender'}
feature_cols = [c for c in df.columns if c not in meta and np.issubdtype(df[c].dtype, np.number)]

# ---- Speaker-level split using StratifiedGroupKFold ----
# Prepare unique speakers and their gender labels
gspeaker = df[['speaker','gender']].drop_duplicates()
speakers_arr = gspeaker['speaker'].values
gender_labels = gspeaker['gender'].map({'M':0,'F':1}).values

# 1) Test split: 10% of speakers
gkf_test = StratifiedGroupKFold(n_splits=int(1/TEST_PCT), shuffle=True, random_state=RANDOM_STATE)
_, test_idx = next(gkf_test.split(speakers_arr, gender_labels, groups=speakers_arr))
test_speakers = speakers_arr[test_idx]

# Remaining for train_val
train_val_speakers = np.setdiff1d(speakers_arr, test_speakers)
train_val_labels = gspeaker.set_index('speaker').loc[train_val_speakers,'gender'].map({'M':0,'F':1}).values

# 2) Validation split: 20% of train_val speakers
gkf_val = StratifiedGroupKFold(n_splits=int(1/VAL_PCT), shuffle=True, random_state=RANDOM_STATE)
_, val_idx = next(gkf_val.split(train_val_speakers, train_val_labels, groups=train_val_speakers))
val_speakers   = train_val_speakers[val_idx]
train_speakers = np.setdiff1d(train_val_speakers, val_speakers)

# Slice DataFrame by speaker sets
df_train = df[df['speaker'].isin(train_speakers)]
df_val   = df[df['speaker'].isin(val_speakers)]
df_test  = df[df['speaker'].isin(test_speakers)]

# Extract arrays
X_train, y_train = df_train[feature_cols].values, df_train['y'].values
X_val,   y_val   = df_val[feature_cols].values,   df_val['y'].values
X_test,  y_test  = df_test[feature_cols].values,  df_test['y'].values

# Standardize
scaler   = StandardScaler().fit(X_train)
X_train  = scaler.transform(X_train)
X_val    = scaler.transform(X_val)
X_test   = scaler.transform(X_test)

# ---- Hyperparameter space ----
param_grid = {
    'units1': [32, 64, 128, 256],
    'units2': [16, 32, 64, 128],
    'dropout': [0.2, 0.3, 0.4, 0.5],
    'l2_reg': [1e-4, 1e-3, 1e-2],
    'lr': [1e-4, 3e-4, 1e-3],
    'batch': [16, 32, 64]
}
import numpy.random as rnd
rnd.seed(RANDOM_STATE)

def sample_config():
    return {
        'units1': int(rnd.choice(param_grid['units1'])),
        'units2': int(rnd.choice(param_grid['units2'])),
        'dropout': float(rnd.choice(param_grid['dropout'])),
        'l2_reg': float(rnd.choice(param_grid['l2_reg'])),
        'lr': float(rnd.choice(param_grid['lr'])),
        'batch': int(rnd.choice(param_grid['batch']))
    }

# ---- Random search ----
results = []
best_val = -np.inf
best_history = None
best_cfg = None
for i in range(1, N_TRIALS+1):
    cfg = sample_config()
    # build model
    model = Sequential([
        InputLayer(input_shape=(X_train.shape[1],)),
        Dense(cfg['units1'], activation='relu', kernel_regularizer=l2(cfg['l2_reg'])),
        BatchNormalization(), Dropout(cfg['dropout']),
        Dense(cfg['units2'], activation='relu', kernel_regularizer=l2(cfg['l2_reg'])),
        BatchNormalization(), Dropout(cfg['dropout']),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(cfg['lr']), loss='binary_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=cfg['batch'],
        callbacks=[es],
        verbose=0
    )

    tr_acc  = hist.history['accuracy'][-1]
    val_acc = max(hist.history['val_accuracy'])
    test_acc= model.evaluate(X_test, y_test, verbose=0)[1]

    print(f"Trial {i}: train_acc={tr_acc:.4f}, val_acc={val_acc:.4f}, test_acc={test_acc:.4f}")
    results.append({**cfg, 'train_acc': tr_acc, 'val_acc': val_acc, 'test_acc': test_acc})
    if val_acc > best_val:
        best_val = val_acc
        best_history = hist
        best_cfg = cfg

# Save results
pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
print("Best configuration:", best_cfg, "val_acc=", best_val)

# Plot best
plt.figure()
plt.plot(best_history.history['accuracy'], label='Train')
plt.plot(best_history.history['val_accuracy'], label='Val')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Best Config Accuracy')
plt.legend(); 
plt.savefig(BEST_PLOT)
