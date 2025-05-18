#!/usr/bin/env python3
"""
Train and evaluate FCNN on MFCC+F0 features with speaker- and gender-balanced splits.
- Reads CSV with a filename column, MFCC means & stds, f0_mean, f0_std.
- Extracts label (lie/truth), speaker code, and gender from filename.
- Splits speakers *within each gender* into:
    • Test: 10% of speakers of each gender
    • Validation: 20% of *remaining* speakers of each gender
    • Train: 80% of *remaining* speakers of each gender
  ensuring equal gender distribution across train/val/test and speaker-independence.
- Slices recordings accordingly into df_train, df_val, df_test.
- Standardizes features based on train set.
- Builds FCNN:
    Dense(64, ReLU) → BatchNorm → Dropout
    Dense(32, ReLU) → BatchNorm → Dropout
    Dense(1, sigmoid)
  with L2 regularization, Adam optimizer, binary crossentropy, early stopping.
- Trains on df_train, validates on df_val, evaluates on df_test.
- Saves accuracies to metrics.csv and plots train/val curves and test accuracy to accuracy_plot.png
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, InputLayer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# ---- Parameters ----
CSV_PATH    = 'signal_features_global_normalized.csv'  # input CSV
METRICS_CSV = 'metrics.csv'
PLOT_PNG    = 'accuracy_plot.png'
RANDOM_STATE= 42
TEST_PCT    = 0.10
VAL_PCT     = 0.20  # of remaining after test
BATCH_SIZE  = 32
EPOCHS      = 100
PATIENCE    = 10
L2_REG      = 1e-3
DROP_RATE   = 0.5

# ---- Load and parse metadata ----
df = pd.read_csv(CSV_PATH)
# detect filename column
auto_fn = next((c for c in df.columns if 'file' in c.lower() or 'name' in c.lower()), None)
if not auto_fn:
    raise RuntimeError('No filename column found in CSV')

# regex for label and gender
pat_label  = re.compile(r"(?i)(truth|lie)[ _]?(\d{1,3})")
pat_gender = re.compile(r"speaker([MFmf])")
labels, speakers, genders = [], [], []
for fn in df[auto_fn].astype(str):
    m = pat_label.search(fn)
    if not m:
        raise ValueError(f"Filename '{fn}' lacks truth/lie label")
    lab = f"{m.group(1).lower()}_{int(m.group(2)):03d}"
    labels.append(1 if lab.startswith('lie') else 0)
    speakers.append(lab)
    mg = pat_gender.search(fn)
    genders.append(mg.group(1).upper() if mg else 'U')
# attach
df['y'] = labels
df['speaker'] = speakers
df['gender'] = genders

# features
meta_cols = {auto_fn, 'y', 'speaker', 'gender'}
feature_cols = [c for c in df.columns if c not in meta_cols and np.issubdtype(df[c].dtype, np.number)]

# ---- Speaker-level split by gender ----
speaker_gend = df[['speaker','gender']].drop_duplicates().set_index('speaker')['gender']

def split_speakers(spk_list, test_pct, val_pct, rng):
    n = len(spk_list)
    n_test = max(1, int(np.floor(n * test_pct)))
    test_sp = rng.choice(spk_list, size=n_test, replace=False).tolist()
    rem_sp = [s for s in spk_list if s not in test_sp]
    n_val = max(1, int(np.floor(len(rem_sp) * val_pct)))
    val_sp = rng.choice(rem_sp, size=n_val, replace=False).tolist()
    train_sp = [s for s in rem_sp if s not in val_sp]
    return train_sp, val_sp, test_sp

rng = np.random.default_rng(RANDOM_STATE)
train_spks, val_spks, test_spks = [], [], []
for gender in ['M','F']:
    spk_gender = speaker_gend[speaker_gend == gender].index.tolist()
    tr, va, te = split_speakers(spk_gender, TEST_PCT, VAL_PCT, rng)
    train_spks.extend(tr)
    val_spks.extend(va)
    test_spks.extend(te)

# slice data
df_train = df[df['speaker'].isin(train_spks)]
df_val   = df[df['speaker'].isin(val_spks)]
df_test  = df[df['speaker'].isin(test_spks)]

# sanity check: gender ratios
for name, subset in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
    print(f"{name} female ratio: {subset['gender'].eq('F').mean():.2%}, speakers={subset['speaker'].nunique()}")

# extract arrays
X_train, y_train = df_train[feature_cols].values, df_train['y'].values
X_val,   y_val   = df_val[feature_cols].values,   df_val['y'].values
X_test,  y_test  = df_test[feature_cols].values,  df_test['y'].values

# standardize
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# model builder
def build_model(input_dim):
    m = Sequential([
        InputLayer(input_shape=(input_dim,)),
        Dense(64, activation='relu', kernel_regularizer=l2(L2_REG)),
        BatchNormalization(), Dropout(DROP_RATE),
        Dense(32, activation='relu', kernel_regularizer=l2(L2_REG)),
        BatchNormalization(), Dropout(DROP_RATE),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m

# train
model = build_model(X_train.shape[1])
es = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es],
    verbose=1
)

# evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# save metrics
metrics = {
    'train_accuracy': [history.history['accuracy'][-1]],
    'validation_accuracy': [history.history['val_accuracy'][-1]],
    'test_accuracy': [test_acc]
}
pd.DataFrame(metrics).to_csv(METRICS_CSV, index=False)

# plot
plt.figure()
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.hlines(test_acc, 0, len(history.history['accuracy'])-1, linestyles='--', label='Test')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy')
plt.savefig(PLOT_PNG)
