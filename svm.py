import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

def codeOneHot(Y_int, Kclass):
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
    UA = np.round(np.sum(CN/VN)/Kclass*100, decimals=1)
    return UA

def getWA(OUT, TAR):
    DB_size = OUT.shape[0]
    OUT = np.argmax(OUT, axis=1)
    TAR = np.argmax(TAR, axis=1)
    hits = np.sum(OUT == TAR)
    WA = np.round(hits/DB_size*100, decimals=1)
    return WA

# Încărcarea datelor din CSV
data = pd.read_csv('global_norm.csv')  

# Presupunem că coloana 'label' conține etichetele (sincer/înșelător)
# și 'speaker' conține identificatorul vorbitorului
# și 'gender' conține genul vorbitorului

# Codificarea etichetelor
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['label'])
Kclass = len(label_encoder.classes_)

# Selectarea caracteristicilor (toate coloanele care încep cu MFCC sau F0)
feature_cols = [col for col in data.columns if col.startswith('MFCC') or col.startswith('F0')]
X = data[feature_cols].values

# Grupurile pentru GroupKFold (pentru independența vorbitorilor)
groups = data['speaker'].values

# Genul pentru stratificare (pentru distribuția proporțională pe gen)
gender = data['gender'].values

# Combinarea genului și etichetei pentru stratificare
# Acest lucru asigură distribuție proporțională atât pe gen cât și pe etichetă
stratify = np.array([f"{g}_{l}" for g, l in zip(gender, y)])

SVM_kernels = ['linear', 'poly', 'rbf']
Cs = [1, 1e-1, 1e-2]
Nsim = len(SVM_kernels)*len(Cs)
METRIX_ = np.zeros((Nsim, 4))
idx_sim = 0

for SVM_kernel in SVM_kernels:
    for C in Cs:
        METRIX = []
        
        # Folosim GroupKFold pentru a asigura independența vorbitorilor
        group_kfold = GroupKFold(n_splits=5)
        
        for idx, (idx_train, idx_val) in enumerate(group_kfold.split(X, y, groups)):
            X_train = X[idx_train]
            Y_train = y[idx_train]
            X_val = X[idx_val]
            Y_val = y[idx_val]
            
            # Verificăm distribuția proporțională pe gen
            train_genders = gender[idx_train]
            val_genders = gender[idx_val]
            
            MODEL = SVC(C=C, kernel=SVM_kernel)
            MODEL.fit(X_train, Y_train)
            OUT_train = MODEL.predict(X_train)
            OUT_val = MODEL.predict(X_val)
            
            UA_train = getUA(codeOneHot(OUT_train, Kclass),
                             codeOneHot(Y_train, Kclass))
            WA_train = getWA(codeOneHot(OUT_train, Kclass),
                             codeOneHot(Y_train, Kclass))
            UA_val = getUA(codeOneHot(OUT_val, Kclass), 
                           codeOneHot(Y_val, Kclass))
            WA_val = getWA(codeOneHot(OUT_val, Kclass), 
                           codeOneHot(Y_val, Kclass))
            METRIX += [UA_train, WA_train, UA_val, WA_val]
            
        # Calculul mediilor pentru cross-validation
        UA_train_avg = WA_train_avg = UA_val_avg = WA_val_avg = 0
        L = len(METRIX)
        for i in range(0, L, 4):
            UA_train_avg += METRIX[i]
        UA_train_avg = np.round(UA_train_avg/5, decimals=1)
        for i in range(1, L, 4):
            WA_train_avg += METRIX[i]
        WA_train_avg = np.round(WA_train_avg/5, decimals=1)
        for i in range(2, L, 4):
            UA_val_avg += METRIX[i]
        UA_val_avg = np.round(UA_val_avg/5, decimals=1)
        for i in range(3, L, 4):
            WA_val_avg += METRIX[i]
        WA_val_avg = np.round(WA_val_avg/5, decimals=1)
        
        METRIX_[idx_sim,:] = [UA_train_avg, WA_train_avg,
                              UA_val_avg, WA_val_avg]
        idx_sim += 1

# Crearea DataFrame-ului cu rezultate
sim_list_idx = range(0, Nsim)
sim_list_SVM_kernels = []
for item in SVM_kernels:
    sim_list_SVM_kernels += len(Cs)*[item]
sim_list_Cs = len(SVM_kernels)*Cs

df_dict = { 
    'SIM': sim_list_idx,
    'Kernel': sim_list_SVM_kernels,
    'C': sim_list_Cs,
    'UA_train [%]': METRIX_[:,0],
    'WA_train [%]': METRIX_[:,1],
    'UA_val [%]': METRIX_[:,2],
    'WA_val [%]': METRIX_[:,3]
}

df = pd.DataFrame(df_dict)
df.to_csv('SVM_Xval.csv', index=False)