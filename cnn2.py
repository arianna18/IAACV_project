#!/usr/bin/env python3
"""
Random search over 10 CNN configurations on spectrograms.
- Loads per-speaker .npy spectrograms and labels
- One-hot encodes gender
- Splits speakers: 10% test, then 5-fold StratifiedGroupKFold on remaining for train/val
- Standardizes per-spectrogram features? (Not needed for CNN)
- Samples 10 random configs from generate_cnn_configs()
- For each: runs 5-fold StratifiedGroupKFold CV on train_val speakers, records mean CV acc
- Trains best config on full train_val, evaluates on test
- Saves all configs + mean CV acc to `cnn_results.csv`
- Plots best config CV curve and test acc to `best_cnn_accuracy.png`
"""
import os, re, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# Config
SPECTRO_DIR = 'spectrograms'
RANDOM_STATE= 42
TEST_PCT    = 0.10
NFOLDS      = 5
N_TRIALS    = 10
RESULTS_CSV = 'cnn_results.csv'
BEST_PLOT   = 'best_cnn_accuracy.png'
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# Load data
def load_spectrograms():
    speaker_data, speaker_labels, speaker_genders = defaultdict(list), {}, {}
    for f in os.listdir(SPECTRO_DIR):
        if not f.endswith('.npy'): continue
        parts = f.split('_')
        lab = parts[1]  # 'lie' or 'truth'
        spk = parts[2]
        gender = parts[3][0].upper()
        spec = np.load(os.path.join(SPECTRO_DIR, f))
        speaker_data[spk].append(spec)
        speaker_labels.setdefault(spk, []).append(1 if lab=='lie' else 0)
        speaker_genders[spk] = gender
    return speaker_data, speaker_labels, speaker_genders

# Flatten
def build_arrays(speaker_data, speaker_labels, speaker_genders):
    X, y, spk, gender = [], [], [], []
    for s, specs in speaker_data.items():
        labs = speaker_labels[s]
        for spec, lab in zip(specs, labs):
            X.append(spec)
            y.append(lab)
            spk.append(s)
            gender.append(speaker_genders[s])
    X = np.expand_dims(np.array(X), -1)
    return X, np.array(y), np.array(spk), np.array(gender)

# Generate configs
def generate_cnn_configs():
    bases = [
        {'n_conv':2,'filters':[32,64],'kernels':[(3,3),(3,3)],'pools':[(2,2),(2,2)],
         'drop':[0.25,0.25],'dense':[64],'dd':0.5,'lr':1e-3,'l2':1e-3,'bn':True,'epochs':20,'bs':32},
        {'n_conv':3,'filters':[32,64,128],'kernels':[(3,3)]*3,'pools':[(2,2)]*3,
         'drop':[0.3]*3,'dense':[128,64],'dd':0.5,'lr':1e-4,'l2':1e-2,'bn':True,'epochs':30,'bs':16},
    ]
    configs=[]
    for b in bases:
        configs.append(b)
    return configs[:N_TRIALS]

# Build model
def build_cnn(input_shape, cfg):
    m=Sequential([InputLayer(input_shape=input_shape)])
    for i in range(cfg['n_conv']):
        m.add(Conv2D(cfg['filters'][i],cfg['kernels'][i],'relu','same',kernel_regularizer=l2(cfg['l2'])))
        m.add(MaxPooling2D(cfg['pools'][i]))
        m.add(Dropout(cfg['drop'][i]))
        if cfg['bn']: m.add(BatchNormalization())
    m.add(Flatten())
    for u in cfg['dense']:
        m.add(Dense(u,'relu',kernel_regularizer=l2(cfg['l2'])))
        m.add(Dropout(cfg['dd']))
        if cfg['bn']: m.add(BatchNormalization())
    m.add(Dense(1,'sigmoid'))
    m.compile(Adam(cfg['lr']),'binary_crossentropy',['accuracy'])
    return m

# Main
speaker_data, spk_labels, spk_genders=load_spectrograms()
X,y,spk,gen=build_arrays(speaker_data, spk_labels, spk_genders)
# speaker-level arrays
unique_spk, idx = np.unique(spk, return_index=True)
gender_lab = np.array([1 if gen[idx[i]]=='F' else 0 for i in range(len(unique_spk))])
# test split
sgkf=StratifiedGroupKFold(n_splits=int(1/TEST_PCT),shuffle=True,random_state=RANDOM_STATE)
_,test_i=next(sgkf.split(unique_spk,gender_lab,groups=unique_spk))
test_s=unique_spk[test_i]
trainval_s=np.setdiff1d(unique_spk,test_s)
# CV folds on trainval
sgkf2=StratifiedGroupKFold(n_splits=NFOLDS,shuffle=True,random_state=RANDOM_STATE)
# Random search
test_mask=np.isin(spk,test_s)
X_test,y_test=X[test_mask],y[test_mask]
results=[]
best_val=-1;best_cfg=None;best_hist=None
for cfg in generate_cnn_configs():
    fold_acc=[]
    for tr_i,val_i in sgkf2.split(trainval_s,gid:=[1 if spk_genders[s]=='F' else 0 for s in trainval_s],groups=trainval_s):
        tr_s,va_s=trainval_s[tr_i],trainval_s[val_i]
        mask_tr=np.isin(spk,tr_s)
        mask_va=np.isin(spk,va_s)
        X_tr,y_tr=X[mask_tr],y[mask_tr]
        X_va,y_va=X[mask_va],y[mask_va]
        m=build_cnn(X_tr.shape[1:],cfg)
        es=EarlyStopping('val_loss',patience=5,restore_best_weights=True)
        hist=m.fit(X_tr,y_tr,validation_data=(X_va,y_va),epochs=cfg['epochs'],batch_size=cfg['bs'],callbacks=[es],verbose=0)
        p=(m.predict(X_va)>0.5).astype(int)
        fold_acc.append(accuracy_score(y_va,p))
    mean_val=np.mean(fold_acc)
    results.append({**cfg,'val_acc':mean_val})
    if mean_val>best_val: best_val, best_cfg=mean_val,cfg; best_hist=hist
# final test
m=build_cnn(X.shape[1:],best_cfg)
X_trval=X[~test_mask];y_trval=y[~test_mask]
m.fit(X_trval,y_trval,epochs=best_cfg['epochs'],batch_size=best_cfg['bs'],verbose=0)
p_test=(m.predict(X_test)>0.5).astype(int)
results_df=pd.DataFrame(results)
results_df['test_acc']=accuracy_score(y_test,p_test)
results_df.to_csv(RESULTS_CSV,index=False)
print("Best config",best_cfg,best_val)
plt.plot(best_hist.history['val_accuracy']);plt.title('Best CV fold accuracy');plt.savefig(BEST_PLOT)
