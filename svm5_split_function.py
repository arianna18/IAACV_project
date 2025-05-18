import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import random

INPUT_CSV = 'signal_features_with_speaker.csv'
OUTPUT_CSV = 'SVM_results_final55.csv'

SVM_KERNELS = ['linear', 'poly', 'rbf']
CS          = [100, 10, 1, 0.1, 0.01]

def code_one_hot(Y_int):
    Y_onehot = np.zeros((len(Y_int), Kclass))
    for i in range(len(Y_int)):
        Y_onehot[i, Y_int[i]] = 1
    return Y_onehot

def get_UA(OUT, TAR):
    Kclass = OUT.shape[1]
    VN = np.sum(TAR, axis=0)
    aux = TAR - OUT
    WN = np.sum((aux + np.abs(aux)) // 2, axis=0)
    CN = VN - WN
    UA = np.round(np.sum(CN / VN) / Kclass * 100, decimals=1)
    return UA

def get_WA(OUT, TAR):
    OUT = np.argmax(OUT, axis=1)
    TAR = np.argmax(TAR, axis=1)
    return np.round(np.mean(OUT == TAR) * 100, decimals=1)

def create_stratified_splits(speaker_genders, speaker_labels, test_size=0.1):
    groups = defaultdict(list)
    for spk in speaker_genders:
        key = (speaker_genders[spk], speaker_labels[spk])
        groups[key].append(spk)
    test_speakers = []
    for key in groups:
        spks = groups[key]
        n_test = max(1, round(len(spks) * test_size))
        test_speakers.extend(random.sample(spks, n_test))
    return set(test_speakers)

def display_speaker_results(test_data, y_test, y_pred_test, label_encoder):
    test_data = test_data.copy()
    test_data['predicted_label'] = label_encoder.inverse_transform(y_pred_test)
    speaker_results = test_data.groupby('speaker').agg({
        'gender': 'first',
        'label': 'first',
        'predicted_label': lambda x: x.mode()[0]
    }).reset_index()
    speaker_results['correct'] = speaker_results['label'] == speaker_results['predicted_label']
    # confusion matrix
    # print("\nconfusion matrix")
    # print(pd.crosstab(
    #     label_encoder.inverse_transform(y_test),
    #     label_encoder.inverse_transform(y_pred_test),
    #     rownames=['true'],
    #     colnames=['predicted']
    # ))

data = pd.read_csv(INPUT_CSV)
label_encoder = LabelEncoder()
data['label_enc'] = label_encoder.fit_transform(data['label'])
Kclass = len(label_encoder.classes_)
print(Kclass)

# stratified speaker split
speaker_genders = data.groupby('speaker')['gender'].first().to_dict()
speaker_labels  = data.groupby('speaker')['label'].first().to_dict()
test_speakers   = create_stratified_splits(speaker_genders, speaker_labels, test_size=0.1)

is_test    = data['speaker'].isin(test_speakers)
train_data = data[~is_test]
test_data  = data[is_test]

feature_cols = [c for c in data.columns if c.startswith('MFCC') or c.startswith('F0')]
X_train = train_data[feature_cols].values
y_train = train_data['label_enc'].values
print('y train', len(y_train))

X_test  = test_data[feature_cols].values
y_test  = test_data['label_enc'].values
print('y test', len(y_test))

groups = train_data['speaker'].values
group_kfold = GroupKFold(n_splits=5)

results = []

for kernel in SVM_KERNELS:
    for C in CS:
        fold_metrics = {'UA_train': [], 'WA_train': [], 'UA_val': [], 'WA_val': []}
        for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X_train, y_train, groups)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model = SVC(C=C, kernel=kernel, probability=False)
            model.fit(X_tr, y_tr)

            y_pred_tr  = model.predict(X_tr)
            y_pred_val = model.predict(X_val)

            fold_metrics['UA_train'].append(get_UA(code_one_hot(y_pred_tr), code_one_hot(y_tr)))
            fold_metrics['WA_train'].append(get_WA(code_one_hot(y_pred_tr), code_one_hot(y_tr)))
            fold_metrics['UA_val'].append(get_UA(code_one_hot(y_pred_val), code_one_hot(y_val)))
            fold_metrics['WA_val'].append(get_WA(code_one_hot(y_pred_val), code_one_hot(y_val)))

        final_model = SVC(C=C, kernel=kernel, probability=False)
        final_model.fit(X_train, y_train)
        y_pred_test = final_model.predict(X_test)
        OUT_test = code_one_hot(y_pred_test)
        Y_test_hot = code_one_hot(y_test)
        UA_test = get_UA(OUT_test, Y_test_hot)
        WA_test = get_WA(OUT_test, Y_test_hot)

        results.append({
            'kernel'       : kernel,
            'C'            : C,
            'UA_train_avg' : np.mean(fold_metrics['UA_train']),
            'WA_train_avg' : np.mean(fold_metrics['WA_train']),
            'UA_val_avg'   : np.mean(fold_metrics['UA_val']),
            'WA_val_avg'   : np.mean(fold_metrics['WA_val']),
            'UA_test'      : UA_test,
            'WA_test'      : WA_test,
        })

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
labels = [f"{row['kernel']}-{row['C']}" for _, row in df_results.iterrows()]
plt.xticks(x, labels, rotation=90)
plt.xlabel('Experiment (kernel-C)')
plt.ylabel('WA (%)')
plt.title('WA – antrenare, validare și testare')
plt.legend()
plt.tight_layout()
plt.show()

# best model + confusion matrix
best_idx    = df_results['WA_val_avg'].idxmax()
best_params = df_results.loc[best_idx, ['kernel','C','UA_train_avg','WA_train_avg','UA_val_avg','WA_val_avg','UA_test','WA_test']]
print("\n test evaluation")
print(f"best model: kernel={best_params['kernel']}, C={best_params['C']}, "
      f"UA_train_avg={best_params['UA_train_avg']}, WA_train_avg={best_params['WA_train_avg']}, "
      f"UA_val_avg={best_params['UA_val_avg']}, UA_test={best_params['UA_test']}, WA_test={best_params['WA_test']}")

# refit and plot confusion matrix for best model
best_model = SVC(C=best_params['C'], kernel=best_params['kernel'], probability=False)
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

cm = pd.crosstab(
    label_encoder.inverse_transform(y_test),
    label_encoder.inverse_transform(y_pred_best),
    rownames=['true'],
    colnames=['predicted']
)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de confuzie pentru best model')
plt.xlabel('Etichete prezise')
plt.ylabel('Etichete adevărate')
plt.tight_layout()
plt.show()
