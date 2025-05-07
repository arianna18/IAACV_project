import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.metrics import make_scorer

# === Dataset ===
X = DATASET.drop('label', axis=1)
Y = DATASET['label']
Kclass = Y.nunique()

# === One-hot encoding pentru metrici personalizate ===
def codeOneHot(Y_int, Kclass):
    DB_size = Y_int.shape[0]
    Y_onehot = np.zeros((DB_size, Kclass))
    for i in range(DB_size):
        Y_onehot[i, Y_int[i]] = 1
    return Y_onehot

def getWA(y_true, y_pred):
    return np.mean(y_true == y_pred)

def getUA(y_true, y_pred):
    y_true_oh = codeOneHot(y_true, Kclass)
    y_pred_oh = codeOneHot(y_pred, Kclass)
    K = y_true_oh.shape[1]
    VN = np.sum(y_true_oh, axis=0)
    aux = y_true_oh - y_pred_oh
    WN = np.sum((aux + np.abs(aux)) // 2, axis=0)
    CN = VN - WN
    UA = np.sum(CN / VN) / K
    return UA

# === Hiperparametri pentru Grid Search ===
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [None, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt'],
    'max_samples': [None, 0.7]
}

from itertools import product

# === Generăm toate combinațiile posibile ===
keys, values = zip(*param_grid.items())
all_configs = [dict(zip(keys, v)) for v in product(*values)]

# === Cross-validation + salvare scoruri ===
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for config in all_configs:
    print(f"Evaluating: {config}")
    ua_scores = []
    wa_scores = []

    for train_idx, val_idx in cv.split(X, Y):
        X_train, Y_train = X.iloc[train_idx], Y.iloc[train_idx]
        X_val, Y_val = X.iloc[val_idx], Y.iloc[val_idx]

        model = RandomForestClassifier(**config, random_state=42)
        model.fit(X_train, Y_train)

        y_pred = model.predict(X_val)
        wa_scores.append(getWA(Y_val.to_numpy(), y_pred))
        ua_scores.append(getUA(Y_val.to_numpy(), y_pred))

    avg_ua = round(np.mean(ua_scores) * 100, 2)
    avg_wa = round(np.mean(wa_scores) * 100, 2)

    result = config.copy()
    result.update({
        'UA (%)': avg_ua,
        'WA (%)': avg_wa
    })
    results.append(result)

# === Salvare rezultate în CSV ===
results_df = pd.DataFrame(results)
results_df.to_csv("rf_gridsearch_results.csv", index=False)
print("\n✅ Rezultatele au fost salvate în 'rf_gridsearch_results.csv'")
