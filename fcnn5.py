import re
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

CSV_PATH     = 'signal_features_with_speaker.csv'
RESULTS_CSV  = 'FCNN_results_final55.csv'
TEST_PCT     = 0.10
RANDOM_STATE = 42
N_TRIALS     = 100
EPOCHS       = 100
PATIENCE     = 10

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
        label = grp['label'].iloc[0]
        gender= grp['gender'].iloc[0]
        speaker_groups[(label, gender)].append(speaker)

    test_speakers = []
    for (label, gender), speakers in speaker_groups.items():
            n_test = max(1, round(len(speakers) * test_size))
            test_speakers.extend(np.random.choice(speakers, n_test, replace=False))
    
    # for (label, gender), speakers in speaker_groups.items():
    #     print(f"Group (label={label}, gender={gender}): with {len(speakers)} speakers ")
    #     for spk in speakers:
    #         print(spk)

    return test_speakers


def display_speaker_results(test_data, y_test, y_pred_test, label_encoder):
    test_data = test_data.copy()
    test_data['predicted_label'] = label_encoder.inverse_transform(y_pred_test)
    speaker_results = test_data.groupby('speaker').agg({
        'gender': 'first',
        'label'     : 'first',
        'predicted_label': lambda x: x.mode()[0]
    }).reset_index()

    speaker_results['correct'] = speaker_results['label'] == speaker_results['predicted_label']

    # confusion matrix
    print("\n confusion matrix")
    print(pd.crosstab(
        label_encoder.inverse_transform(y_test),
        label_encoder.inverse_transform(y_pred_test),
        rownames=['True'], colnames=['Predicted']
    ))


df = pd.read_csv(CSV_PATH)
# print("columns:", df.columns.tolist())

fn_col = next(c for c in df if re.search(r'file|name', c, re.I))
pat_lab  = re.compile(r"(?i)(truth|lie)[ _]?(\d{1,3})")
pat_gen  = re.compile(r"speaker([MFmf])")

labels, speakers, genders = [], [], []
for fn in df[fn_col].astype(str):
    m = pat_lab.search(fn)
    lab = f"{m.group(1).lower()}_{int(m.group(2)):03d}"
    labels.append(1 if lab.startswith('lie') else 0)
    speakers.append(lab)
    mg = pat_gen.search(fn)
    genders.append(mg.group(1).upper() if mg else 'U')

df['label']       = labels
df['speaker'] = speakers
df['gender']  = genders

# feature columns
meta_cols    = {fn_col, 'label', 'speaker', 'gender'}
feature_cols = [c for c in df.columns if c not in meta_cols and np.issubdtype(df[c].dtype, np.number)]

# ---- Encode & split out test speakers ----
label_encoder = LabelEncoder().fit(df['label'])
df['label_enc'] = label_encoder.transform(df['label'])
Kclass     = len(label_encoder.classes_)

test_speakers  = create_stratified_speaker_split(df)
print(test_speakers)

is_test    = df['speaker'].isin(test_speakers)
train_df   = df[~is_test]
test_df    = df[ is_test]

X_train = train_df[feature_cols].values
y_train = train_df['label_enc'].values
X_test  = test_df [feature_cols].values
y_test  = test_df ['label_enc'].values


scaler    = StandardScaler().fit(X_train)
X_train   = scaler.transform(X_train)
X_test    = scaler.transform(X_test)

groups      = train_df['speaker'].values
group_kfold = GroupKFold(n_splits=5)

param_grid = {
    'units1': [32, 64, 128, 256],
    'units2': [16, 32, 64, 128],
    'dropout': [0.2, 0.3, 0.4, 0.5],
    'l2_reg': [1e-4, 1e-3, 1e-2],
    'lr': [1e-4, 3e-4, 1e-3],
    'batch': [16, 32, 64]
}
rnd = np.random.default_rng(RANDOM_STATE)
def sample_config():
    return {
        'units1': rnd.choice(param_grid['units1']),
        'units2': rnd.choice(param_grid['units2']),
        'dropout': rnd.choice(param_grid['dropout']),
        'l2_reg': rnd.choice(param_grid['l2_reg']),
        'lr': rnd.choice(param_grid['lr']),
        'batch': rnd.choice(param_grid['batch']),
    }

results   = []
for trial in range(1, N_TRIALS+1):
    cfg = sample_config()
    fold_metrics = {'UA_train':[], 'WA_train':[], 'UA_val':[], 'WA_val':[]}

    for tr_idx, val_idx in group_kfold.split(X_train, y_train, groups):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        model = Sequential([
            InputLayer(input_shape=(X_tr.shape[1],)),
            Dense(cfg['units1'], activation='relu', kernel_regularizer=l2(cfg['l2_reg'])),
            BatchNormalization(), Dropout(cfg['dropout']),
            Dense(cfg['units2'], activation='relu', kernel_regularizer=l2(cfg['l2_reg'])),
            BatchNormalization(), Dropout(cfg['dropout']),
            Dense(Kclass, activation='softmax')
        ])
        model.compile(optimizer=Adam(cfg['lr']), loss='sparse_categorical_crossentropy')
        es = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
        model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                  epochs=EPOCHS, batch_size=cfg['batch'],
                  callbacks=[es], verbose=0)

        # predictions
        y_pred_tr  = np.argmax(model.predict(X_tr, verbose=0), axis=1)
        y_pred_val = np.argmax(model.predict(X_val, verbose=0), axis=1)

        # compute UA/WA
        fold_metrics['UA_train'].append(get_UA(code_one_hot(y_pred_tr), code_one_hot(y_tr)))
        fold_metrics['WA_train'].append(get_WA(code_one_hot(y_pred_tr), code_one_hot(y_tr)))
        fold_metrics['UA_val']  .append(get_UA(code_one_hot(y_pred_val), code_one_hot(y_val)))
        fold_metrics['WA_val']  .append(get_WA(code_one_hot(y_pred_val), code_one_hot(y_val)))

    final_model = Sequential([
        InputLayer(input_shape=(X_train.shape[1],)),
        Dense(cfg['units1'], activation='relu', kernel_regularizer=l2(cfg['l2_reg'])),
        BatchNormalization(), Dropout(cfg['dropout']),
        Dense(cfg['units2'], activation='relu', kernel_regularizer=l2(cfg['l2_reg'])),
        BatchNormalization(), Dropout(cfg['dropout']),
        Dense(Kclass, activation='softmax')
    ])

    final_model.compile(optimizer=Adam(cfg['lr']), loss='sparse_categorical_crossentropy')
    es_final = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    final_model.fit(X_train, y_train, validation_split=0.0,
                    epochs=EPOCHS, batch_size=cfg['batch'],
                    callbacks=[es_final], verbose=0)

    y_pred_test = np.argmax(final_model.predict(X_test, verbose=0), axis=1)
    OUT_test     = code_one_hot(y_pred_test)
    Y_test_hot   = code_one_hot(y_test)
    UA_test      = get_UA(OUT_test, Y_test_hot)
    WA_test      = get_WA(OUT_test, Y_test_hot)

    results.append({
        **cfg,
        'UA_train_avg': np.mean(fold_metrics['UA_train']),
        'WA_train_avg': np.mean(fold_metrics['WA_train']),
        'UA_val_avg'  : np.mean(fold_metrics['UA_val']),
        'WA_val_avg'  : np.mean(fold_metrics['WA_val']),
        'UA_test'     : UA_test,
        'WA_test'     : WA_test,
    })
    print(f"Trial {trial}: UA_test={UA_test:.1f}%, WA_test={WA_test:.1f}%")


df_results = pd.DataFrame(results)
df_results.to_csv(RESULTS_CSV, index=False)
print("\nfinal results:")
print(df_results.round(3))

best_idx = df_results['WA_val_avg'].idxmax()
best_params = df_results.loc[best_idx, ['units1','units2','dropout','l2_reg','lr','batch','UA_train_avg','WA_train_avg','UA_val_avg','UA_test','WA_test']]
print(f"\n test evaluation")
print(f"best config: {best_params.to_dict()}")

best_cfg = best_params.to_dict()
display_speaker_results(test_df, y_test, np.argmax(final_model.predict(X_test, verbose=0), axis=1), label_encoder)


pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
print("\nDone. Results written to", RESULTS_CSV)
