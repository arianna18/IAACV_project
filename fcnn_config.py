#!/usr/bin/env python3
"""
Random search over 100 FCNN configurations on MFCC+F0 features.
- Loads CSV and parses label, speaker, gender.
- Splits speakers within each gender into train/val/test balanced.
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
RESULTS_CSV = 'hyperparam_results.csv'
BEST_PLOT   = 'best_accuracy_plot.png'

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
meta_cols = {auto_fn,'y','speaker','gender'}
feature_cols = [c for c in df.columns if c not in meta_cols and np.issubdtype(df[c].dtype, np.number)]

# ---- Split speakers within gender ----
speaker_gend = df[['speaker','gender']].drop_duplicates().set_index('speaker')['gender']
import numpy.random as rnd
rnd.seed(RANDOM_STATE)
def split_speakers(sp_list, test_pct, val_pct):
    n = len(sp_list)
    n_test = max(1, int(n*test_pct))
    test_sp = list(rnd.choice(sp_list, n_test, replace=False))
    rem = [s for s in sp_list if s not in test_sp]
    n_val = max(1, int(len(rem)*val_pct))
    val_sp = list(rnd.choice(rem, n_val, replace=False))
    train_sp = [s for s in rem if s not in val_sp]
    return train_sp, val_sp, test_sp
train_s, val_s, test_s = [], [], []
for g in ['M','F']:
    spk = speaker_gend[speaker_gend==g].index.tolist()
    tr, va, te = split_speakers(spk, TEST_PCT, VAL_PCT)
    train_s += tr; val_s += va; test_s += te

df_train = df[df['speaker'].isin(train_s)]
df_val   = df[df['speaker'].isin(val_s)]
df_test  = df[df['speaker'].isin(test_s)]

X_train, y_train = df_train[feature_cols].values, df_train['y'].values
X_val,   y_val   = df_val[feature_cols].values,   df_val['y'].values
X_test,  y_test  = df_test[feature_cols].values,  df_test['y'].values

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ---- Hyperparameter space ----
param_grid = {
    'units1': [32,64,128,256], 'units2': [16,32,64,128],
    'dropout': [0.2,0.3,0.4,0.5], 'l2_reg': [1e-4,1e-3,1e-2],
    'lr': [1e-4,3e-4,1e-3], 'batch': [16,32,64]
}
def sample_config():
    return {k: float(rnd.choice(v)) if isinstance(v[0], float) else int(rnd.choice(v))
            for k,v in param_grid.items()}

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

    # train
    es = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    hist = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                     epochs=EPOCHS, batch_size=int(cfg['batch']), callbacks=[es], verbose=0)

    # evaluate
    tr_acc  = hist.history['accuracy'][-1]
    val_acc = max(hist.history['val_accuracy'])
    test_acc= model.evaluate(X_test, y_test, verbose=0)[1]

    # record and print
    results.append({**cfg, 'train_acc': tr_acc, 'val_acc': val_acc, 'test_acc': test_acc})
    print(f"Trial {i}: train_acc={tr_acc:.4f}, val_acc={val_acc:.4f}, test_acc={test_acc:.4f}")

    if val_acc > best_val:
        best_val = val_acc
        best_history = hist
        best_cfg = cfg

# save results
pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
print("Best configuration:", best_cfg, "with val_acc=", best_val)

# plot best
plt.figure()
plt.plot(best_history.history['accuracy'], label='Train')
plt.plot(best_history.history['val_accuracy'], label='Val')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Best Config Accuracy')
plt.legend(); plt.savefig(BEST_PLOT)
