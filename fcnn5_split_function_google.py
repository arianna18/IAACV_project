import re
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

INPUT_CSV    = '/content/drive/MyDrive/pcdtv/signal_features_with_speaker.csv'
OUTPUT_CSV   = '/content/drive/MyDrive/pcdtv/FCNN_results_final55.csv'

# hyperparameter grid
UNITS1_LIST  = [32, 64, 128, 256]
UNITS2_LIST  = [16, 32, 64, 128]
DROPOUTS     = [0.2, 0.3, 0.4, 0.5]
L2_REGS      = [1e-4, 1e-3, 1e-2]
LRS          = [1e-4, 3e-4, 1e-3]
BATCH_SIZES  = [16, 32, 64]

def code_one_hot(Y_int):
    Y_onehot = np.zeros((len(Y_int), Kclass))
    for i in range(len(Y_int)):
        Y_onehot[i, Y_int[i]] = 1
    return Y_onehot

def get_UA(OUT, TAR):
    K = OUT.shape[1]
    VN = np.sum(TAR, axis=0)
    aux = TAR - OUT
    WN = np.sum((aux + np.abs(aux)) // 2, axis=0)
    CN = VN - WN
    return np.round(np.sum(CN / VN) / K * 100, 1)

def get_WA(OUT, TAR):
    OUT = np.argmax(OUT, axis=1)
    TAR = np.argmax(TAR, axis=1)
    return np.round(np.mean(OUT == TAR) * 100, 1)

def create_stratified_speaker_split(data, test_size=0.1):
    speaker_groups = defaultdict(list)
    for speaker, grp in data.groupby('speaker'):
        label  = grp['label'].iloc[0]
        gender = grp['gender'].iloc[0]
        speaker_groups[(label, gender)].append(speaker)
    test_speakers = []
    for (label, gender), speakers in speaker_groups.items():
        n_test = max(1, round(len(speakers) * test_size))
        test_speakers.extend(np.random.choice(speakers, n_test, replace=False))
    return set(test_speakers)

def display_speaker_results(test_data, y_test, y_pred_test, label_encoder):
    test_data = test_data.copy()
    test_data['predicted_label'] = label_encoder.inverse_transform(y_pred_test)
    speaker_results = test_data.groupby('speaker').agg({
        'gender': 'first',
        'label' : 'first',
        'predicted_label': lambda x: x.mode()[0]
    }).reset_index()
    speaker_results['correct'] = (speaker_results['label'] == speaker_results['predicted_label'])

    print("\nconfusion matrix (speaker‐majority vote)")
    print(pd.crosstab(
        speaker_results['label'],
        speaker_results['predicted_label'],
        rownames=['true'],
        colnames=['predicted']
    ))


df = pd.read_csv(INPUT_CSV)

# extract label, speaker, gender from filename
fn_col = next(c for c in df.columns if re.search(r'file|name', c, re.I))
pat_lab = re.compile(r'(?i)(truth|lie)[ _]?(\d{1,3})')
pat_gen = re.compile(r'speaker([MFmf])')
labels, speakers, genders = [], [], []
for fn in df[fn_col].astype(str):
    m = pat_lab.search(fn)
    lab = f"{m.group(1).lower()}_{int(m.group(2)):03d}"
    labels.append(1 if lab.startswith('lie') else 0)
    speakers.append(lab)
    mg = pat_gen.search(fn)
    genders.append(mg.group(1).upper() if mg else 'U')
df['label']   = labels
df['speaker'] = speakers
df['gender']  = genders

# feature matrix and encoding
meta_cols    = {fn_col, 'label', 'speaker', 'gender'}
feature_cols = [c for c in df.columns if c not in meta_cols and np.issubdtype(df[c].dtype, np.number)]

label_encoder = LabelEncoder().fit(df['label'])
df['label_enc'] = label_encoder.transform(df['label'])
Kclass = len(label_encoder.classes_)

# stratified speaker‐level split
test_speakers = create_stratified_speaker_split(df)
train_df = df[~df['speaker'].isin(test_speakers)]
test_df  = df[ df['speaker'].isin(test_speakers)]

X_train = train_df[feature_cols].values
y_train = train_df['label_enc'].values
X_test  = test_df[ feature_cols].values
y_test  = test_df['label_enc'].values

# scale features
from sklearn.preprocessing import StandardScaler
scaler  = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

# cross‐validation setup
groups      = train_df['speaker'].values
group_kfold = GroupKFold(n_splits=5)

results = []

for units1 in UNITS1_LIST:
    for units2 in UNITS2_LIST:
        for dropout in DROPOUTS:
            for l2_reg in L2_REGS:
                for lr in LRS:
                    for batch in BATCH_SIZES:
                        fold_metrics = {'UA_train':[], 'WA_train':[], 'UA_val':[], 'WA_val':[]}

                        # CV folds
                        for tr_idx, val_idx in group_kfold.split(X_train, y_train, groups):
                            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
                            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

                            model = Sequential([
                                InputLayer(input_shape=(X_tr.shape[1],)),
                                Dense(units1, activation='relu', kernel_regularizer=l2(l2_reg)),
                                BatchNormalization(), Dropout(dropout),
                                Dense(units2, activation='relu', kernel_regularizer=l2(l2_reg)),
                                BatchNormalization(), Dropout(dropout),
                                Dense(Kclass, activation='softmax')
                            ])
                            model.compile(optimizer=Adam(lr), loss='sparse_categorical_crossentropy')
                            es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                            model.fit(X_tr, y_tr,
                                      validation_data=(X_val, y_val),
                                      epochs=100,
                                      batch_size=batch,
                                      callbacks=[es],
                                      verbose=0)

                            y_pred_tr  = np.argmax(model.predict(X_tr, verbose=0), axis=1)
                            y_pred_val = np.argmax(model.predict(X_val, verbose=0), axis=1)

                            fold_metrics['UA_train'].append(get_UA(code_one_hot(y_pred_tr), code_one_hot(y_tr)))
                            fold_metrics['WA_train'].append(get_WA(code_one_hot(y_pred_tr), code_one_hot(y_tr)))
                            fold_metrics['UA_val'].append(get_UA(code_one_hot(y_pred_val), code_one_hot(y_val)))
                            fold_metrics['WA_val'].append(get_WA(code_one_hot(y_pred_val), code_one_hot(y_val)))

                        # retrain on full training set
                        final_model = Sequential([
                            InputLayer(input_shape=(X_train.shape[1],)),
                            Dense(units1, activation='relu', kernel_regularizer=l2(l2_reg)),
                            BatchNormalization(), Dropout(dropout),
                            Dense(units2, activation='relu', kernel_regularizer=l2(l2_reg)),
                            BatchNormalization(), Dropout(dropout),
                            Dense(Kclass, activation='softmax')
                        ])
                        final_model.compile(optimizer=Adam(lr), loss='sparse_categorical_crossentropy')
                        final_model.fit(X_train, y_train,
                                        epochs=100,
                                        batch_size=batch,
                                        verbose=0)

                        y_pred_test = np.argmax(final_model.predict(X_test, verbose=0), axis=1)
                        OUT_test    = code_one_hot(y_pred_test)
                        Y_test_hot  = code_one_hot(y_test)
                        UA_test     = get_UA(OUT_test, Y_test_hot)
                        WA_test     = get_WA(OUT_test, Y_test_hot)

                        print(
                            f"Done: units1={units1}, units2={units2}, dropout={dropout}, "
                            f"l2_reg={l2_reg}, lr={lr}, batch={batch} → "
                            f"WA_val={np.mean(fold_metrics['WA_val']):.1f}%, WA_test={WA_test:.1f}%"
                        )

                        results.append({
                            'units1'      : units1,
                            'units2'      : units2,
                            'dropout'     : dropout,
                            'l2_reg'      : l2_reg,
                            'lr'          : lr,
                            'batch'       : batch,
                            'UA_train_avg': np.mean(fold_metrics['UA_train']),
                            'WA_train_avg': np.mean(fold_metrics['WA_train']),
                            'UA_val_avg'  : np.mean(fold_metrics['UA_val']),
                            'WA_val_avg'  : np.mean(fold_metrics['WA_val']),
                            'UA_test'     : UA_test,
                            'WA_test'     : WA_test
                        })

# --- save and display results ---
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_CSV, index=False)
print("\nfinal CV results:")
print(df_results.round(3))

# plot WA curves
plt.figure(figsize=(10, 6))
x = np.arange(len(df_results))
plt.plot(x, df_results['WA_train_avg'], marker='o', label='Train WA')
plt.plot(x, df_results['WA_val_avg'],   marker='o', label='Val WA')
plt.plot(x, df_results['WA_test'],      marker='o', label='Test WA')
labels = [
    f"{r['units1']}-{r['units2']}-{r['dropout']}-{r['l2_reg']}-{r['lr']}-{r['batch']}"
    for _, r in df_results.iterrows()
]
plt.xticks(x, labels, rotation=90)
plt.xlabel('Experiment (units1-units2-dropout-l2-lr-batch)')
plt.ylabel('WA (%)')
plt.title('WA – antrenare, validare și testare (FCNN)')
plt.legend()
plt.tight_layout()
plt.show()

# best model + confusion matrix
best_idx    = df_results['WA_val_avg'].idxmax()
best_params = df_results.loc[best_idx, [
    'units1','units2','dropout','l2_reg','lr','batch',
    'UA_train_avg','WA_train_avg','UA_val_avg','UA_test','WA_test'
]]

print("\n test evaluation")
print(f"best model: units1={best_params['units1']}, units2={best_params['units2']}, "
      f"dropout={best_params['dropout']}, l2_reg={best_params['l2_reg']}, lr={best_params['lr']}, "
      f"batch={best_params['batch']}, UA_train_avg={best_params['UA_train_avg']}, "
      f"WA_train_avg={best_params['WA_train_avg']}, UA_val_avg={best_params['UA_val_avg']}, "
      f"UA_test={best_params['UA_test']}, WA_test={best_params['WA_test']}")

# retrain best model for confusion matrix
best_model = Sequential([
    InputLayer(input_shape=(X_train.shape[1],)),
    Dense(best_params['units1'], activation='relu', kernel_regularizer=l2(best_params['l2_reg'])),
    BatchNormalization(), Dropout(best_params['dropout']),
    Dense(best_params['units2'], activation='relu', kernel_regularizer=l2(best_params['l2_reg'])),
    BatchNormalization(), Dropout(best_params['dropout']),
    Dense(Kclass, activation='softmax')
])
best_model.compile(optimizer=Adam(best_params['lr']), loss='sparse_categorical_crossentropy')
best_model.fit(X_train, y_train, epochs=100, batch_size=int(best_params['batch']), verbose=0)

y_pred_best = np.argmax(best_model.predict(X_test, verbose=0), axis=1)
cm = pd.crosstab(
    label_encoder.inverse_transform(y_test),
    label_encoder.inverse_transform(y_pred_best),
    rownames=['true'],
    colnames=['predicted']
)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de confuzie pentru best FCNN model')
plt.xlabel('Etichete prezise')
plt.ylabel('Etichete adevărate')
plt.tight_layout()
plt.show()
