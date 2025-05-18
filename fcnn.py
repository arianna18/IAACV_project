import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, InputLayer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# ---- Parameters ----
CSV_PATH = 'signal_features_global_normalized.csv'  # Update to your CSV
METRICS_CSV = 'metrics.csv'
PLOT_PNG = 'accuracy_plot.png'
TEST_SPLIT = 0.10
VAL_SPLIT = 0.20
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 10
L2_REG = 1e-3
DROP_RATE = 0.5

# ---- Load data ----
df = pd.read_csv(CSV_PATH)
# 1) Detect filename column
filename_col = next((c for c in df.columns if 'file' in c.lower() or 'name' in c.lower()), None)
if filename_col is None:
    raise RuntimeError("No filename column found (look for 'file' or 'name') in CSV")

# 2) Extract label and speaker from filename
pat = re.compile(r"(?i)(truth|lie)[ _]?(\d{1,3})")
labels = []
speakers = []
for fn in df[filename_col].astype(str):
    m = pat.search(fn)
    if not m:
        raise ValueError(f"Filename '{fn}' does not contain truth/lie label")
    lab, sid = m.groups()
    lab = lab.lower()
    labels.append(f"{lab}_{int(sid):03d}")
    speakers.append(f"{lab}_{int(sid):03d}")
df['label'] = labels
# binary target: lie=1, truth=0
df['y'] = [1 if l.startswith('lie') else 0 for l in labels]
# store speaker code for splitting
df['speaker'] = speakers

# 3) Feature columns are all numeric except meta
meta = {filename_col, 'label', 'y', 'speaker'}
feature_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number) and c not in meta]

# ---- Split speakers for test ----
unique_speakers = df['speaker'].unique()
rng = np.random.default_rng(RANDOM_STATE)
test_speakers = rng.choice(unique_speakers, size=int(len(unique_speakers)*TEST_SPLIT), replace=False)
test_idx = df['speaker'].isin(test_speakers)

df_test = df[test_idx]
df_rest = df[~test_idx]

# ---- Train/val split ----
X_rest = df_rest[feature_cols].values
y_rest = df_rest['y'].values
X_train, X_val, y_train, y_val = train_test_split(
    X_rest, y_rest,
    test_size=VAL_SPLIT,
    stratify=y_rest,
    random_state=RANDOM_STATE
)
X_test = df_test[feature_cols].values
y_test = df_test['y'].values

# ---- Standardize ----
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ---- Build FCNN ----
model = Sequential([
    InputLayer(input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu', kernel_regularizer=l2(L2_REG)),
    BatchNormalization(),
    Dropout(DROP_RATE),
    Dense(32, activation='relu', kernel_regularizer=l2(L2_REG)),
    BatchNormalization(),
    Dropout(DROP_RATE),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ---- Train ----
es = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
hist = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es],
    verbose=1
)

# ---- Evaluate ----
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

# ---- Save metrics ----
metrics_df = pd.DataFrame({
    'train_accuracy': [hist.history['accuracy'][-1]],
    'validation_accuracy': [hist.history['val_accuracy'][-1]],
    'test_accuracy': [test_acc]
})
metrics_df.to_csv(METRICS_CSV, index=False)

# ---- Plot ----
plt.figure()
plt.plot(hist.history['accuracy'], label='Train Acc')
plt.plot(hist.history['val_accuracy'], label='Val Acc')
plt.hlines(test_acc, 0, len(hist.history['accuracy'])-1, colors='r', linestyles='--', label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.savefig(PLOT_PNG)
print(f"Test accuracy: {test_acc:.4f}")
