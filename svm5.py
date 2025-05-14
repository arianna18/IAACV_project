import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

INPUT_CSV = 'signal_features_with_speaker.csv'
OUTPUT_CSV = 'SVM_results_final55.csv'
SVM_KERNELS = ['linear', 'poly', 'rbf']
CS = [10, 1, 0.1, 0.01]


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

def create_stratified_speaker_split(data, test_size=0.1):
    """Create train/test split maintaining speaker independence and balanced proportions"""
    speaker_groups = defaultdict(list)
    for speaker, group in data.groupby('speaker'):
        label = group['label'].iloc[0]
        gender = group['gender'].iloc[0]
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
    
    # Get most frequent prediction for each speaker
    speaker_results = test_data.groupby('speaker').agg({
        'gender': 'first',
        'label': 'first',
        'predicted_label': lambda x: x.mode()[0]
    }).reset_index()
    
    speaker_results['correct'] = speaker_results['label'] == speaker_results['predicted_label']
    
    # print("\n=== Detailed Speaker Results ===")
    # print(speaker_results.to_string(index=False))
    
    # Print accuracy per gender
    # print("\n=== Accuracy by Gender ===")
    # gender_acc = speaker_results.groupby('gender')['correct'].mean()
    # print(gender_acc.to_string())
    
    # Print confusion matrix
    print("\n=== Confusion Matrix ===")
    print(pd.crosstab(
        label_encoder.inverse_transform(y_test),
        label_encoder.inverse_transform(y_pred_test),
        rownames=['True'],
        colnames=['Predicted']
    ))

# def run_experiment():

# Load and prepare data
data = pd.read_csv(INPUT_CSV)
# print("Columns in CSV:", data.columns.tolist())
label_encoder = LabelEncoder()
data['label_enc'] = label_encoder.fit_transform(data['label'])
Kclass = len(label_encoder.classes_)
print(Kclass)
# Create stratified speaker split
test_speakers = create_stratified_speaker_split(data)
print(test_speakers)
is_test = data['speaker'].isin(test_speakers)

# Split data maintaining speaker independence
train_data = data[~is_test]
test_data = data[is_test]

# Prepare features and labels
feature_cols = [col for col in data.columns if col.startswith('MFCC') or col.startswith('F0')]
X_train = train_data[feature_cols].values
y_train = train_data['label_enc'].values
print('y train', len(y_train))
X_test = test_data[feature_cols].values
y_test = test_data['label_enc'].values
print('y train', len(y_test))


# Cross-validation setup
groups = train_data['speaker'].values
group_kfold = GroupKFold(n_splits=5)

results = []

for kernel in SVM_KERNELS:
    for C in CS:        
        fold_metrics = {
            'UA_train': [], 'WA_train': [],
            'UA_val': [], 'WA_val': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(group_kfold.split(X_train, y_train, groups)):
            # Split maintaining speaker independence
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            # Train model
            model = SVC(C=C, kernel=kernel, probability=False)
            model.fit(X_tr, y_tr)
            
            # Evaluate on validation
            y_pred_val = model.predict(X_val)
            
            # Store metrics

            fold_metrics['UA_train'].append(get_UA(code_one_hot(model.predict(X_tr)), code_one_hot(y_tr)))
            fold_metrics['WA_train'].append(get_WA(code_one_hot(model.predict(X_tr)), code_one_hot(y_tr)))
            fold_metrics['UA_val'].append(get_UA(code_one_hot(y_pred_val), code_one_hot(y_val)))
            fold_metrics['WA_val'].append(get_WA(code_one_hot(y_pred_val),code_one_hot(y_val)))
            
            # Evaluate on test set for this fold
            y_pred_test = model.predict(X_test)
            # print(f"\nFold {fold+1} Test Set Results:")
            # display_speaker_results(test_data, y_test, y_pred_test, label_encoder)
        
        # Aggregate results
        result = {
            'kernel': kernel,
            'C': C,
            'UA_train_avg': np.mean(fold_metrics['UA_train']),
            'WA_train_avg': np.mean(fold_metrics['WA_train']),
            'UA_val_avg': np.mean(fold_metrics['UA_val']),
            'WA_val_avg': np.mean(fold_metrics['WA_val']),
        }
        results.append(result)

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_CSV, index=False)
print("\nFinal Results:")
print(df_results.round(3))

# Final evaluation with best model
best_idx = df_results['WA_val_avg'].idxmax()
best_model = SVC(C=df_results.iloc[best_idx]['C'], 
                kernel=df_results.iloc[best_idx]['kernel'])
best_model.fit(X_train, y_train)

y_pred_test = best_model.predict(X_test)
OUT_test = code_one_hot(y_pred_test)
Y_test_hot = code_one_hot(y_test)

UA_test = get_UA(OUT_test, Y_test_hot)
WA_test = get_WA(OUT_test, Y_test_hot)

print("\n=== Final Test Evaluation ===")
print(f"Best Model: kernel={df_results.iloc[best_idx]['kernel']}, C={df_results.iloc[best_idx]['C']}")
print(f"UA_test: {UA_test:.1f}%")
print(f"WA_test: {WA_test:.1f}%")

# Display final per-speaker results
display_speaker_results(test_data, y_test, y_pred_test, label_encoder)

# return df_results

# Run the experiment
# final_results = run_experiment()