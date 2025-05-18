# import re
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import StratifiedGroupKFold
# import numpy.random as rnd

# # ---- User parameters ----
# CSV_PATH      = "signal_features_global_normalized.csv"
# RESULTS_CSV   = "svm_results.csv"
# BEST_PLOT_PNG = "best_svm_accuracy.png"
# RANDOM_STATE  = 42
# TEST_PCT      = 0.10
# VAL_PCT       = 0.20  # of the remaining 90%
# N_TRIALS      = 10

# # ---- 1) Load data & parse metadata ----
# df = pd.read_csv(CSV_PATH)
# fn_col = next((c for c in df.columns if "file" in c.lower() or "name" in c.lower()), None)
# if fn_col is None:
#     raise RuntimeError("No filename column found in CSV")

# pat_label  = re.compile(r"(?i)(truth|lie)[ _]?(\d{1,3})")
# pat_gender = re.compile(r"speaker([MFmf])")

# ys, spks, gds = [], [], []
# for fn in df[fn_col].astype(str):
#     m = pat_label.search(fn)
#     if not m:
#         raise ValueError(f"Filename '{fn}' lacks truth/lie label")
#     lab = f"{m.group(1).lower()}_{int(m.group(2)):03d}"
#     ys.append(1 if lab.startswith("lie") else 0)
#     spks.append(lab)
#     mg = pat_gender.search(fn)
#     gds.append(mg.group(1).upper() if mg else "U")

# df["y"] = ys
# df["speaker"] = spks
# df["gender"]  = gds

# meta = {fn_col, "y", "speaker", "gender"}
# feature_cols = [c for c in df.columns
#                 if c not in meta and np.issubdtype(df[c].dtype, np.number)]

# # ---- 2) Speaker-level split via StratifiedGroupKFold ----
# # Prepare unique speaker list & their gender labels
# sp_df = df[["speaker","gender"]].drop_duplicates().reset_index(drop=True)
# speakers_arr  = sp_df["speaker"].values
# gender_labels = sp_df["gender"].map({"M":0,"F":1}).values

# # 2a) TEST split (10% of speakers)
# n_splits_test = int(1.0/TEST_PCT)
# sgkf_test = StratifiedGroupKFold(n_splits=n_splits_test,
#                                  shuffle=True, random_state=RANDOM_STATE)
# _, test_idx = next(sgkf_test.split(speakers_arr, gender_labels, groups=speakers_arr))
# test_speakers = set(speakers_arr[test_idx])

# # Remaining TRAIN_VAL speakers
# train_val_speakers = np.setdiff1d(speakers_arr, speakers_arr[test_idx])
# tv_genders = sp_df.set_index("speaker").loc[train_val_speakers,"gender"]\
#                    .map({"M":0,"F":1}).values

# # 2b) VAL split (20% of TRAIN_VAL)
# n_splits_val = int(1.0/VAL_PCT)
# sgkf_val = StratifiedGroupKFold(n_splits=n_splits_val,
#                                 shuffle=True, random_state=RANDOM_STATE)
# _, val_idx = next(sgkf_val.split(train_val_speakers, tv_genders,
#                                  groups=train_val_speakers))
# val_speakers   = set(train_val_speakers[val_idx])
# train_speakers = set(train_val_speakers) - val_speakers

# # Slice into DataFrames
# df_train = df[df["speaker"].isin(train_speakers)]
# df_val   = df[df["speaker"].isin(val_speakers)]
# df_test  = df[df["speaker"].isin(test_speakers)]

# X_train, y_train = df_train[feature_cols].values, df_train["y"].values
# X_val,   y_val   = df_val[feature_cols].values,   df_val["y"].values
# X_test,  y_test  = df_test[feature_cols].values,  df_test["y"].values

# # ---- 3) Standardize on TRAIN only ----
# scaler = StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_val   = scaler.transform(X_val)
# X_test  = scaler.transform(X_test)

# # ---- 4) Hyperparameter search space & sampling ----
# param_grid = {
#     "kernel":  ["linear", "rbf", "poly"],
#     "C":       [0.1, 1, 10, 100],
#     "gamma":   ["scale", "auto"],
#     "degree":  [2, 3, 4]  # only used if kernel='poly'
# }
# rnd.seed(RANDOM_STATE)

# def sample_cfg():
#     cfg = {
#         "kernel": rnd.choice(param_grid["kernel"]),
#         "C":      float(rnd.choice(param_grid["C"])),
#         "gamma":  rnd.choice(param_grid["gamma"])
#     }
#     # only sample degree if poly
#     cfg["degree"] = int(rnd.choice(param_grid["degree"])) if cfg["kernel"]=="poly" else 3
#     return cfg

# # ---- 5) Random search over N_TRIALS ----
# results    = []
# best_val   = -np.inf
# best_cfg   = None
# best_scores= None

# for i in range(1, N_TRIALS+1):
#     cfg = sample_cfg()
#     svc = SVC(kernel=cfg["kernel"],
#               C=cfg["C"],
#               gamma=cfg["gamma"],
#               degree=cfg["degree"],
#               random_state=RANDOM_STATE)
#     svc.fit(X_train, y_train)

#     # evaluate
#     tr_acc  = accuracy_score(y_train, svc.predict(X_train))
#     val_acc = accuracy_score(y_val,   svc.predict(X_val))
#     te_acc  = accuracy_score(y_test,  svc.predict(X_test))

#     # log & print
#     results.append({**cfg,
#                     "train_acc": tr_acc,
#                     "val_acc":   val_acc,
#                     "test_acc":  te_acc})
#     print(f"Trial {i}: train={tr_acc:.3f}, val={val_acc:.3f}, test={te_acc:.3f}")

#     if val_acc > best_val:
#         best_val    = val_acc
#         best_cfg    = cfg
#         best_scores = (tr_acc, val_acc, te_acc)

# # ---- 6) Save all trials ----
# pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
# print("\nBest configuration:", best_cfg, f"with val_acc={best_val:.3f}")

# # ---- 7) Plot best configuration ----
# tr, va, te = best_scores
# plt.figure()
# plt.bar(["Train","Val","Test"], [tr, va, te], color=["C0","C1","C2"])
# plt.ylim(0,1)
# plt.ylabel("Accuracy")
# plt.title("Best SVM Configuration Performance")
# plt.savefig(BEST_PLOT_PNG)


import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
import numpy.random as rnd

# ---- Helper functions ----
def codeOneHot(Y_int):
    DB_size = Y_int.shape[0]
    Y_onehot = np.zeros((DB_size, Kclass))
    for i in range(0, DB_size):
        Y_onehot[i, Y_int[i]] = 1
    return Y_onehot

def getUA(OUT, TAR):
    Kclass = OUT.shape[1]
    VN = np.sum(TAR, axis=0)
    aux = TAR - OUT
    WN = np.sum((aux + np.absolute(aux)) // 2, axis=0)
    CN = VN - WN
    UA = np.round(np.sum(CN / VN) / Kclass * 100, decimals=1)
    return UA

def getWA(OUT, TAR):
    DB_size = OUT.shape[0]
    OUT = np.argmax(OUT, axis=1)
    TAR = np.argmax(TAR, axis=1)
    hits = np.sum(OUT == TAR)
    WA = np.round(hits / DB_size * 100, decimals=1)
    return WA

# ---- User parameters ----
CSV_PATH      = "signal_features_global_normalized.csv"  # Update with your CSV path
RESULTS_CSV   = "svm_config_results_1.csv"
BEST_PLOT_PNG = "best_svm_accuracy.png"
RANDOM_STATE  = 42
TEST_PCT      = 0.10
VAL_PCT       = 0.20  # of the remaining 90%
N_TRIALS      = 10
CV_FOLDS      = 5     # Number of cross-validation folds

# ---- 1) Load data & parse metadata ----
df = pd.read_csv(CSV_PATH)  # Assuming tab-separated based on your sample
print(df.columns)

# Extract labels and convert to binary (0 for truth, 1 for lie)
df['y'] = df['label'].apply(lambda x: 1 if x.strip().lower() == 'lie' else 0)
df['gender'] = df['gender'].str.strip().str.upper()

# Set global variables for helper functions
Kclass = len(df['y'].unique())
Nclass = np.sum(codeOneHot(df['y'].values), axis=0)

# Feature columns (all MFCC and F0 features)
feature_cols = [col for col in df.columns if 'MFCC' in col or 'F0' in col]

# ---- 2) Speaker-level split via StratifiedGroupKFold ----
# Since we don't have speaker IDs, we'll use gender for stratification
# If you have speaker info, replace 'gender' with speaker column
groups = df['gender'].values
gender_labels = df['gender'].map({"M":0,"F":1}).values

# 2a) TEST split (10% of data), stratified on gender
n_splits_test = int(1.0/TEST_PCT)
sgkf_test = StratifiedGroupKFold(n_splits=n_splits_test,
                               shuffle=True, random_state=RANDOM_STATE)
_, test_idx = next(sgkf_test.split(df, gender_labels, groups=groups))
test_mask = np.zeros(len(df), dtype=bool)
test_mask[test_idx] = True
df_test = df[test_mask]
df_train_val = df[~test_mask]

# 2b) VAL split (20% of TRAIN_VAL)
n_splits_val = int(1.0/VAL_PCT)
sgkf_val = StratifiedGroupKFold(n_splits=n_splits_val,
                              shuffle=True, random_state=RANDOM_STATE)
_, val_idx = next(sgkf_val.split(df_train_val, 
                               df_train_val['gender'].map({"M":0,"F":1}).values,
                               groups=df_train_val['gender'].values))
val_mask = np.zeros(len(df_train_val), dtype=bool)
val_mask[val_idx] = True
df_val = df_train_val[val_mask]
df_train = df_train_val[~val_mask]

# Prepare data splits
X_train, y_train = df_train[feature_cols].values, df_train['y'].values
X_val, y_val = df_val[feature_cols].values, df_val['y'].values
X_test, y_test = df_test[feature_cols].values, df_test['y'].values

# Class counts
Nclass_train = np.sum(codeOneHot(y_train), axis=0)
Nclass_val = np.sum(codeOneHot(y_val), axis=0)
Nclass_test = np.sum(codeOneHot(y_test), axis=0)

# ---- 3) Standardize on TRAIN only ----
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ---- 4) Hyperparameter search space & sampling ----
param_grid = {
    "kernel":  ["linear", "rbf", "poly"],
    "C":       [0.1, 1, 10, 100],
    "gamma":   ["scale", "auto"],
    "degree":  [2, 3, 4]  # only used if kernel='poly'
}
rnd.seed(RANDOM_STATE)

def sample_cfg():
    cfg = {
        "kernel": rnd.choice(param_grid["kernel"]),
        "C":      float(rnd.choice(param_grid["C"])),
        "gamma":  rnd.choice(param_grid["gamma"])
    }
    cfg["degree"] = int(rnd.choice(param_grid["degree"])) if cfg["kernel"]=="poly" else 3
    return cfg

# ---- 5) Random search with cross-validation ----
results = []
best_val_wa = -np.inf
best_cfg = None
best_scores = None
best_cv_metrics = None

for trial in range(1, N_TRIALS+1):
    cfg = sample_cfg()
    print(f"\nTrial {trial} Configuration: {cfg}")
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr, X_v = X_train[train_idx], X_train[val_idx]
        y_tr, y_v = y_train[train_idx], y_train[val_idx]
        
        svc = SVC(kernel=cfg["kernel"],
                 C=cfg["C"],
                 gamma=cfg["gamma"],
                 degree=cfg["degree"],
                 random_state=RANDOM_STATE)
        svc.fit(X_tr, y_tr)
        
        # Predictions
        pred_tr = svc.predict(X_tr)
        pred_v = svc.predict(X_v)
        
        # Convert to one-hot for metrics
        y_tr_oh = codeOneHot(y_tr)
        y_v_oh = codeOneHot(y_v)
        pred_tr_oh = codeOneHot(pred_tr)
        pred_v_oh = codeOneHot(pred_v)
        
        # Calculate metrics
        ua_tr = getUA(pred_tr_oh, y_tr_oh)
        wa_tr = getWA(pred_tr_oh, y_tr_oh)
        ua_v = getUA(pred_v_oh, y_v_oh)
        wa_v = getWA(pred_v_oh, y_v_oh)
        cv_metrics.append([ua_tr, wa_tr, ua_v, wa_v])
        
        # Print fold results
        print(f"\nFold {fold+1}:")
        print('Class count:')
        print('-> TRAIN:  ', end='')
        for i in range(0, Kclass):
            print('%d:%3d' % (i, np.sum(codeOneHot(y_tr)[:,i]), end==''))
            if i < Kclass-1:
                print(' | ', end='')
        print('')
        print('-> VAL:    ', end='')
        for i in range(0, Kclass):
            print('%d:%3d' % (i, np.sum(codeOneHot(y_v)[:,i]), end==''))
            if i < Kclass-1:
                print(' | ', end='')
        print('\n')
        print('Metrics:')
        print(f'-> UA (train) = {ua_tr:.1f}%')
        print(f'-> WA (train) = {wa_tr:.1f}%')
        print(f'-> UA (val) = {ua_v:.1f}%')
        print(f'-> WA (val) = {wa_v:.1f}%')
    
    # Calculate average CV metrics
    cv_metrics = np.array(cv_metrics)
    avg_ua_tr, avg_wa_tr = np.mean(cv_metrics[:,0]), np.mean(cv_metrics[:,1])
    avg_ua_v, avg_wa_v = np.mean(cv_metrics[:,2]), np.mean(cv_metrics[:,3])
    
    # Train final model on full training set
    svc_final = SVC(kernel=cfg["kernel"],
                   C=cfg["C"],
                   gamma=cfg["gamma"],
                   degree=cfg["degree"],
                   random_state=RANDOM_STATE)
    svc_final.fit(X_train, y_train)
    
    # Evaluate on all sets
    pred_tr = svc_final.predict(X_train)
    pred_val = svc_final.predict(X_val)
    pred_te = svc_final.predict(X_test)
    
    # Convert to one-hot for metrics
    y_tr_oh = codeOneHot(y_train)
    y_val_oh = codeOneHot(y_val)
    y_te_oh = codeOneHot(y_test)
    pred_tr_oh = codeOneHot(pred_tr)
    pred_val_oh = codeOneHot(pred_val)
    pred_te_oh = codeOneHot(pred_te)
    
    # Calculate metrics
    ua_tr = getUA(pred_tr_oh, y_tr_oh)
    wa_tr = getWA(pred_tr_oh, y_tr_oh)
    ua_val = getUA(pred_val_oh, y_val_oh)
    wa_val = getWA(pred_val_oh, y_val_oh)
    ua_te = getUA(pred_te_oh, y_te_oh)
    wa_te = getWA(pred_te_oh, y_te_oh)
    
    # Store results
    results.append({
        **cfg,
        "cv_ua_train": avg_ua_tr,
        "cv_wa_train": avg_wa_tr,
        "cv_ua_val": avg_ua_v,
        "cv_wa_val": avg_wa_v,
        "train_ua": ua_tr,
        "train_wa": wa_tr,
        "val_ua": ua_val,
        "val_wa": wa_val,
        "test_ua": ua_te,
        "test_wa": wa_te
    })
    
    # Print overall results
    print("\nOverall Class count:")
    print('-> ALL:    ', end='')
    for i in range(0, Kclass):
        print('%d:%3d' % (i, Nclass[i]), end='')
        if i < Kclass-1:
            print(' | ', end='')
    print('')
    print('-> TRAIN:  ', end='')
    for i in range(0, Kclass):
        print('%d:%3d' % (i, Nclass_train[i]), end='')
        if i < Kclass-1:
            print(' | ', end='')
    print('')
    print('-> VAL:    ', end='')
    for i in range(0, Kclass):
        print('%d:%3d' % (i, Nclass_val[i]), end='')
        if i < Kclass-1:
            print(' | ', end='')
    print('')
    print('-> TEST:   ', end='')
    for i in range(0, Kclass):
        print('%d:%3d' % (i, Nclass_test[i]), end='')
        if i < Kclass-1:
            print(' | ', end='')
    print('\n')
    
    print('Cross-validation averages:')
    print(f'-> UA (train) = {avg_ua_tr:.1f}%')
    print(f'-> WA (train) = {avg_wa_tr:.1f}%')
    print(f'-> UA (val) = {avg_ua_v:.1f}%')
    print(f'-> WA (val) = {avg_wa_v:.1f}%')
    
    print('\nFull set metrics:')
    print(f'-> UA (train) = {ua_tr:.1f}%')
    print(f'-> WA (train) = {wa_tr:.1f}%')
    print(f'-> UA (val) = {ua_val:.1f}%')
    print(f'-> WA (val) = {wa_val:.1f}%')
    print(f'-> UA (test) = {ua_te:.1f}%')
    print(f'-> WA (test) = {wa_te:.1f}%')

    # Track best based on CV validation WA
    if avg_wa_v > best_val_wa:
        best_val_wa = avg_wa_v
        best_cfg = cfg
        best_scores = (wa_tr, wa_val, wa_te)
        best_cv_metrics = (avg_ua_tr, avg_wa_tr, avg_ua_v, avg_wa_v)

# ---- 6) Save all results ----
pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
print("\nBest configuration:", best_cfg, f"with CV val_wa={best_val_wa:.1f}%")

# ---- 7) Plot best config accuracies ----
tr, va, te = best_scores
plt.figure(figsize=(10, 5))
plt.bar(["Train","Val","Test"], [tr, va, te], color=["C0","C1","C2"])
plt.ylim(0,100)
plt.ylabel("Weighted Accuracy (%)")
plt.title("Best SVM Configuration Performance")
plt.savefig(BEST_PLOT_PNG)

# ---- 8) Print final cross-validation results ----
print("\n" + 50*'=' + '\n' +
      'Best Configuration Cross-validation Results\n' +
      50*'=')
print('')
print('Metrics:')
print(f'-> UA (train) = {best_cv_metrics[0]:.1f}%')
print(f'-> WA (train) = {best_cv_metrics[1]:.1f}%')
print(f'-> UA (val) = {best_cv_metrics[2]:.1f}%')
print(f'-> WA (val) = {best_cv_metrics[3]:.1f}%')
print('')