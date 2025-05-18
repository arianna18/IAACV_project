import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

# ==================== 1. Încărcarea datelor ====================
def load_spectrogram_data(spectrograms_dir):
    """Încarcă datele spectrogramelor și construiește un DataFrame"""
    data = {'spectrogram': [], 'label': [], 'speaker_id': [], 'gender': []}
    
    for filename in os.listdir(spectrograms_dir):
        if filename.endswith('.npy'):
            parts = filename.split('_')
            
            # Extrage informații din numele fișierului
            label = 0 if parts[1] == 'lie' else 1  # 0=minciună, 1=adevăr
            speaker_id = parts[2]
            gender = parts[3][7]  # 'F' sau 'M'
            
            # Încarcă spectrograma și ajustează dimensiunile
            spectrogram = np.load(os.path.join(spectrograms_dir, filename))
            if len(spectrogram.shape) == 2:  # Dacă e (H, W)
                spectrogram = np.expand_dims(spectrogram, axis=0)  # Transformă în (1, H, W)
            
            data['spectrogram'].append(spectrogram)
            data['label'].append(label)
            data['speaker_id'].append(speaker_id)
            data['gender'].append(gender)
    
    return pd.DataFrame(data)

# Încărcare date
spectrograms_dir = 'spectrograms'  # Schimbă cu calea ta
df = load_spectrogram_data(spectrograms_dir)

# ==================== 2. Definirea modelului CNN ====================
class SimpleCNN(nn.Module):
    def __init__(self, in_ch, n_classes):
        super(SimpleCNN, self).__init__()
        self.conv_sequence = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveMaxPool2d(1)
        self.dense_sequence = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x):
        z = self.conv_sequence(x)
        z = self.global_pool(z).view(z.shape[0], -1)
        return self.dense_sequence(z)

# ==================== 3. Dataset și utilitare ====================
class SincerityDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        spectrogram = torch.tensor(self.data.iloc[idx]['spectrogram'], dtype=torch.float32)
        label = torch.tensor(self.data.iloc[idx]['label'], dtype=torch.long)
        return spectrogram, label

def combine_stratification_keys(df):
    """Creează chei de stratificare combinate speaker + gen"""
    if df['label'] == 0:
        return 'lie' + "_" + df['speaker_id'].astype(str) + '_' + df['gender'].astype(str)
    return 'truth' + "_" + df['speaker_id'].astype(str) + '_' + df['gender'].astype(str)

# ==================== 4. Pregătire date ====================
# Adaugă chei de stratificare
df['strat_key'] = combine_stratification_keys(df)

# Împarte în train/test (90%/10%)
train_df, test_df = train_test_split(df, test_size=0.1, stratify=df['strat_key'], random_state=42)

# Reset strat key pentru cross-validation
train_df['strat_key'] = combine_stratification_keys(train_df)

# ==================== 5. Antrenare cu cross-validation ====================
# Configurații
batch_size = 64
lr = 1e-3
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Folosim device: {device}")

# Cross-validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(skf.split(train_df, train_df['strat_key']))

all_fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(splits):
    print(f"\n--- Fold {fold + 1} ---")
    
    # Inițializare model
    model = SimpleCNN(in_ch=1, n_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Pregătire dataloaders
    fold_train = train_df.iloc[train_idx].reset_index(drop=True)
    fold_val = train_df.iloc[val_idx].reset_index(drop=True)

    train_loader = DataLoader(SincerityDataset(fold_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SincerityDataset(fold_val), batch_size=batch_size)

    # Antrenare
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            correct += (preds.argmax(1) == y).sum().item()
            total += y.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}: Train Loss = {train_loss/total:.4f}, Accuracy = {acc:.4f}")

    # Evaluare pe validare
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            correct += (preds.argmax(1) == y).sum().item()
            total += y.size(0)
    
    val_acc = correct / total
    print(f"Fold {fold+1} Validation Accuracy: {val_acc:.4f}")
    all_fold_accuracies.append(val_acc)

# ==================== 6. Evaluare pe test set ====================
test_loader = DataLoader(SincerityDataset(test_df), batch_size=batch_size)
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        correct += (preds.argmax(1) == y).sum().item()
        total += y.size(0)

test_acc = correct / total
print(f"\nFinal Test Accuracy: {test_acc:.4f}")
print(f"Cross-Validation Accuracies: {all_fold_accuracies}")
print(f"Average CV Accuracy: {np.mean(all_fold_accuracies):.4f}")