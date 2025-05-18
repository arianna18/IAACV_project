# import re
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import StratifiedGroupKFold
# import numpy.random as rnd

# # ---- User parameters ----
# CSV_PATH       = "signal_features_global_normalized.csv"
# RESULTS_CSV    = "rf_results.csv"
# BEST_PLOT_PNG  = "best_rf_accuracy.png"
# RANDOM_STATE   = 42
# TEST_PCT       = 0.10
# VAL_PCT        = 0.20  # of the remaining 90%
# N_TRIALS       = 10

# # ---- 1) Load and parse metadata ----
# df = pd.read_csv(CSV_PATH)
# # auto-detect filename column
# fn_col = next((c for c in df.columns if "file" in c.lower() or "name" in c.lower()), None)
# if not fn_col:
#     raise RuntimeError("No filename column found in CSV")

# # extract label, speaker, gender from filename
# pat_label  = re.compile(r"(?i)(truth|lie)[ _]?(\d{1,3})")
# pat_gender = re.compile(r"speaker([MFmf])")

# labels, speakers, genders = [], [], []
# for fn in df[fn_col].astype(str):
#     m = pat_label.search(fn)
#     if not m:
#         raise ValueError(f"Filename '{fn}' lacks truth/lie label")
#     lab = f"{m.group(1).lower()}_{int(m.group(2)):03d}"
#     labels.append(1 if lab.startswith("lie") else 0)
#     speakers.append(lab)
#     mg = pat_gender.search(fn)
#     genders.append(mg.group(1).upper() if mg else "U")

# df["y"] = labels
# df["speaker"] = speakers
# df["gender"]  = genders

# # feature columns (all numeric except meta)
# meta = {fn_col, "y", "speaker", "gender"}
# feature_cols = [c for c in df.columns
#                 if c not in meta and np.issubdtype(df[c].dtype, np.number)]

# # ---- 2) Speaker-level splits via StratifiedGroupKFold ----
# # first: carve out TEST (10% of speakers), stratified on gender
# sp_df = df[["speaker","gender"]].drop_duplicates().reset_index(drop=True)
# speakers_arr   = sp_df["speaker"].values
# gender_labels  = sp_df["gender"].map({"M":0,"F":1}).values

# # use n_splits = 1/TEST_PCT to get 1 fold as test
# n_splits_test = int(1.0/TEST_PCT)
# sgkf_test = StratifiedGroupKFold(n_splits=n_splits_test,
#                                  shuffle=True, random_state=RANDOM_STATE)
# # take the first fold: train_val_idx, test_idx
# _, test_idx = next(sgkf_test.split(speakers_arr, gender_labels, groups=speakers_arr))
# test_speakers = set(speakers_arr[test_idx])

# # remaining speakers
# train_val_speakers = np.setdiff1d(speakers_arr, speakers_arr[test_idx])
# tv_gender = sp_df.set_index("speaker").loc[train_val_speakers, "gender"]\
#                    .map({"M":0,"F":1}).values

# # second: split train_val into TRAIN and VAL (VAL_PCT of train_val)
# n_splits_val = int(1.0/VAL_PCT)
# sgkf_val = StratifiedGroupKFold(n_splits=n_splits_val,
#                                 shuffle=True, random_state=RANDOM_STATE)
# _, val_idx = next(sgkf_val.split(train_val_speakers, tv_gender, groups=train_val_speakers))
# val_speakers   = set(train_val_speakers[val_idx])
# train_speakers = set(train_val_speakers) - val_speakers

# # slice DataFrame
# df_train = df[df["speaker"].isin(train_speakers)]
# df_val   = df[df["speaker"].isin(val_speakers)]
# df_test  = df[df["speaker"].isin(test_speakers)]

# X_train, y_train = df_train[feature_cols].values, df_train["y"].values
# X_val,   y_val   = df_val[feature_cols].values,   df_val["y"].values
# X_test,  y_test  = df_test[feature_cols].values,  df_test["y"].values

# # ---- 3) Standardize features on TRAIN only ----
# scaler = StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_val   = scaler.transform(X_val)
# X_test  = scaler.transform(X_test)

# # ---- 4) Hyperparameter grid & random sampling ----
# param_grid = {
#     "n_estimators":    [50, 100, 200, 500],
#     "max_depth":       [None, 5, 10, 20],
#     "min_samples_leaf":[1, 2, 5, 10],
#     "max_features":    ["sqrt","log2", None]
# }
# rnd.seed(RANDOM_STATE)

# def sample_cfg():
#     return {
#         "n_estimators":     int(rnd.choice(param_grid["n_estimators"])),
#         "max_depth":        rnd.choice(param_grid["max_depth"]),
#         "min_samples_leaf": int(rnd.choice(param_grid["min_samples_leaf"])),
#         "max_features":     rnd.choice(param_grid["max_features"])
#     }

# # ---- 5) Random search over N_TRIALS ----
# results = []
# best_val = -np.inf
# best_cfg = None
# best_scores = None

# for trial in range(1, N_TRIALS+1):
#     cfg = sample_cfg()
#     rf = RandomForestClassifier(
#         n_estimators=cfg["n_estimators"],
#         max_depth=cfg["max_depth"],
#         min_samples_leaf=cfg["min_samples_leaf"],
#         max_features=cfg["max_features"],
#         random_state=RANDOM_STATE
#     )
#     rf.fit(X_train, y_train)

#     # Predictions
#     pred_tr  = rf.predict(X_train)
#     pred_val = rf.predict(X_val)
#     pred_te  = rf.predict(X_test)

#     # Accuracies
#     acc_tr  = accuracy_score(y_train, pred_tr)
#     acc_val = accuracy_score(y_val,   pred_val)
#     acc_te  = accuracy_score(y_test,  pred_te)

#     # Store & print
#     results.append({
#         **cfg,
#         "train_acc":  acc_tr,
#         "val_acc":    acc_val,
#         "test_acc":   acc_te
#     })
#     print(f"Trial {trial}: train={acc_tr:.3f}, val={acc_val:.3f}, test={acc_te:.3f}")

#     # track best
#     if acc_val > best_val:
#         best_val = acc_val
#         best_cfg = cfg
#         best_scores = (acc_tr, acc_val, acc_te)

# # ---- 6) Save all results ----
# pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
# print("\nBest configuration:", best_cfg, f"with val_acc={best_val:.3f}")

# # ---- 7) Plot best config accuracies ----
# tr, va, te = best_scores
# plt.figure()
# plt.bar(["Train","Val","Test"], [tr, va, te], color=["C0","C1","C2"])
# plt.ylim(0,1)
# plt.ylabel("Accuracy")
# plt.title("Best RF Configuration Performance")
# plt.savefig(BEST_PLOT_PNG)



import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
import numpy.random as rnd

# ---- User parameters ----
CSV_PATH       = "signal_features_global_normalized.csv"
RESULTS_CSV    = "rf_results.csv"
BEST_PLOT_PNG  = "best_rf_accuracy.png"
RANDOM_STATE   = 42
TEST_PCT       = 0.10
VAL_PCT        = 0.20  # of the remaining 90%
N_TRIALS       = 10
CV_FOLDS       = 5     # Number of folds for cross-validation

# ---- 1) Load and parse metadata ----
df = pd.read_csv(CSV_PATH)
# auto-detect filename column
fn_col = next((c for c in df.columns if "file" in c.lower() or "name" in c.lower()), None)
if not fn_col:
    raise RuntimeError("No filename column found in CSV")

# extract label, speaker, gender from filename
pat_label  = re.compile(r"(?i)(truth|lie)[ _]?(\d{1,3})")
pat_gender = re.compile(r"speaker([MFmf])")

labels, speakers, genders = [], [], []
for fn in df[fn_col].astype(str):
    m = pat_label.search(fn)
    if not m:
        raise ValueError(f"Filename '{fn}' lacks truth/lie label")
    lab = f"{m.group(1).lower()}_{int(m.group(2)):03d}"
    labels.append(1 if lab.startswith("lie") else 0)
    speakers.append(lab)
    mg = pat_gender.search(fn)
    genders.append(mg.group(1).upper() if mg else "U")

df["y"] = labels
df["speaker"] = speakers
df["gender"]  = genders

# feature columns (all numeric except meta)
meta = {fn_col, "y", "speaker", "gender"}
feature_cols = [c for c in df.columns
                if c not in meta and np.issubdtype(df[c].dtype, np.number)]

# ---- 2) Speaker-level splits via StratifiedGroupKFold ----
# first: carve out TEST (10% of speakers), stratified on gender
sp_df = df[["speaker","gender"]].drop_duplicates().reset_index(drop=True)
speakers_arr   = sp_df["speaker"].values
gender_labels  = sp_df["gender"].map({"M":0,"F":1}).values

# use n_splits = 1/TEST_PCT to get 1 fold as test
n_splits_test = int(1.0/TEST_PCT)
sgkf_test = StratifiedGroupKFold(n_splits=n_splits_test,
                                 shuffle=True, random_state=RANDOM_STATE)
# take the first fold: train_val_idx, test_idx
_, test_idx = next(sgkf_test.split(speakers_arr, gender_labels, groups=speakers_arr))
test_speakers = set(speakers_arr[test_idx])

# remaining speakers
train_val_speakers = np.setdiff1d(speakers_arr, speakers_arr[test_idx])
tv_gender = sp_df.set_index("speaker").loc[train_val_speakers, "gender"]\
                   .map({"M":0,"F":1}).values

# second: split train_val into TRAIN and VAL (VAL_PCT of train_val)
n_splits_val = int(1.0/VAL_PCT)
sgkf_val = StratifiedGroupKFold(n_splits=n_splits_val,
                                shuffle=True, random_state=RANDOM_STATE)
_, val_idx = next(sgkf_val.split(train_val_speakers, tv_gender, groups=train_val_speakers))
val_speakers   = set(train_val_speakers[val_idx])
train_speakers = set(train_val_speakers) - val_speakers

# slice DataFrame
df_train = df[df["speaker"].isin(train_speakers)]
df_val   = df[df["speaker"].isin(val_speakers)]
df_test  = df[df["speaker"].isin(test_speakers)]

X_train, y_train = df_train[feature_cols].values, df_train["y"].values
X_val,   y_val   = df_val[feature_cols].values,   df_val["y"].values
X_test,  y_test  = df_test[feature_cols].values,  df_test["y"].values

# ---- 3) Standardize features on TRAIN only ----
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ---- 4) Hyperparameter grid & random sampling ----
param_grid = {
    "n_estimators":    [50, 100, 200, 500],
    "max_depth":       [None, 5, 10, 20],
    "min_samples_leaf":[1, 2, 5, 10],
    "max_features":    ["sqrt","log2", None]
}
rnd.seed(RANDOM_STATE)

def sample_cfg():
    return {
        "n_estimators":     int(rnd.choice(param_grid["n_estimators"])),
        "max_depth":        rnd.choice(param_grid["max_depth"]),
        "min_samples_leaf": int(rnd.choice(param_grid["min_samples_leaf"])),
        "max_features":     rnd.choice(param_grid["max_features"])
    }

# ---- 5) Random search with cross-validation ----
results = []
best_val = -np.inf
best_cfg = None
best_scores = None

# Prepare for cross-validation on training data
cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

for trial in range(1, N_TRIALS+1):
    cfg = sample_cfg()
    
    # Cross-validation scores
    cv_scores = []
    cv_train_scores = []
    
    # Get groups for CV (speakers in training set)
    train_speakers_cv = df_train["speaker"].values
    train_gender_cv = df_train["gender"].map({"M":0,"F":1}).values
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train, groups=train_speakers_cv)):
        X_tr, X_v = X_train[train_idx], X_train[val_idx]
        y_tr, y_v = y_train[train_idx], y_train[val_idx]
        
        rf = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
            max_features=cfg["max_features"],
            random_state=RANDOM_STATE
        )
        rf.fit(X_tr, y_tr)
        
        # Store fold scores
        cv_train_scores.append(accuracy_score(y_tr, rf.predict(X_tr)))
        cv_scores.append(accuracy_score(y_v, rf.predict(X_v)))
    
    # Average CV scores
    avg_train_acc = np.mean(cv_train_scores)
    avg_val_acc = np.mean(cv_scores)
    
    # Final evaluation on hold-out validation set
    rf_final = RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        min_samples_leaf=cfg["min_samples_leaf"],
        max_features=cfg["max_features"],
        random_state=RANDOM_STATE
    )
    rf_final.fit(X_train, y_train)
    
    # Predictions
    pred_tr  = rf_final.predict(X_train)
    pred_val = rf_final.predict(X_val)
    pred_te  = rf_final.predict(X_test)
    
    # Accuracies
    acc_tr  = accuracy_score(y_train, pred_tr)
    acc_val = accuracy_score(y_val,   pred_val)
    acc_te  = accuracy_score(y_test,  pred_te)
    
    # Store & print
    results.append({
        **cfg,
        "cv_train_acc": avg_train_acc,
        "cv_val_acc":   avg_val_acc,
        "train_acc":    acc_tr,
        "val_acc":      acc_val,
        "test_acc":     acc_te
    })
    print(f"Trial {trial}: CV train={avg_train_acc:.3f}, CV val={avg_val_acc:.3f}, "
          f"Holdout val={acc_val:.3f}, test={acc_te:.3f}")
    
    # track best based on CV validation score
    if avg_val_acc > best_val:
        best_val = avg_val_acc
        best_cfg = cfg
        best_scores = (avg_train_acc, avg_val_acc, acc_tr, acc_val, acc_te)

# ---- 6) Save all results ----
pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
print("\nBest configuration:", best_cfg, f"with CV val_acc={best_val:.3f}")

# ---- 7) Plot best config accuracies ----
cv_tr, cv_va, tr, va, te = best_scores
plt.figure(figsize=(10, 5))
plt.bar(["CV Train", "CV Val", "Train", "Val", "Test"], 
        [cv_tr, cv_va, tr, va, te], 
        color=["C0", "C1", "C0", "C1", "C2"])
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Best RF Configuration Performance")
plt.savefig(BEST_PLOT_PNG)