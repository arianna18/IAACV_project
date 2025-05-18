import os
import copy
import random
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------- CONFIGURABLE PARAMS --------------------------- #
SPECTROGRAMS_DIR = "spectrograms"        # directory with .npy files
TEST_SIZE = 0.10                          # 10 % of speakers used as TEST (hold‑out)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# -------------------------------------------------------------------------- #
#                         1.  DATA  LOADING                                  #
# -------------------------------------------------------------------------- #

def load_data():
    """Parcurge directorul cu spectrograme *.npy și le organizează pe vorbitori.

    Denumire fișier așteptată:  <prefix>_<lie|truth>_<speakerID>_<gender>.npy
    Exemplu:  utt_lie_001_F.npy
    """
    speaker_data: dict[str, list[np.ndarray]] = defaultdict(list)
    speaker_labels: dict[str, list[int]] = defaultdict(list)
    speaker_genders: dict[str, str] = {}

    for file in os.listdir(SPECTROGRAMS_DIR):
        if not file.endswith(".npy"):
            continue
        parts = file.split("_")
        if len(parts) < 4:
            raise ValueError(
                f"Nume de fișier invalid: {file}. Așteptat: <prefix>_<lie|truth>_<speakerID>_<gender>.npy"
            )
        lie_truth = parts[1].lower()            # "lie" sau "truth"
        speaker_id = parts[2]                   # ex. "001"
        gender = parts[3][0].upper()            # "F" sau "M" (din "F.npy")

        spectrogram = np.load(os.path.join(SPECTROGRAMS_DIR, file))

        speaker_data[speaker_id].append(spectrogram)
        speaker_labels[speaker_id].append(0 if lie_truth == "lie" else 1)
        speaker_genders[speaker_id] = gender

    return speaker_data, speaker_labels, speaker_genders


def prepare_data(speaker_data: dict, speaker_labels: dict):
    """Concatenăm spectrogramele într‑un singur array X și etichetele în y."""
    X, y, speaker_ids = [], [], []
    for spk in speaker_data:
        X.extend(speaker_data[spk])
        y.extend(speaker_labels[spk])
        speaker_ids.extend([spk] * len(speaker_data[spk]))

    return np.asarray(X), np.asarray(y), np.asarray(speaker_ids)


# -------------------------------------------------------------------------- #
#                      2.  TRAIN / TEST  SPLIT (speaker‑level)               #
# -------------------------------------------------------------------------- #

def split_data(X, y, speaker_ids, speaker_genders, test_size=0.10, seed=42):
    """Împarte setul în TRAIN+VAL vs TEST păstrând proporția F/M la nivel de vorbitori."""
    rnd = random.Random(seed)
    speakers = list(set(speaker_ids))
    female = [s for s in speakers if speaker_genders[s] == "F"]
    male = [s for s in speakers if speaker_genders[s] == "M"]

    n_test_f = max(1, round(len(female) * test_size)) if female else 0
    n_test_m = max(1, round(len(male) * test_size)) if male else 0

    test_speakers = rnd.sample(female, n_test_f) + rnd.sample(male, n_test_m)
    if not test_speakers:                       # fallback
        test_speakers.append(rnd.choice(speakers))

    test_mask = np.isin(speaker_ids, test_speakers)
    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]
    speaker_ids_train = speaker_ids[~test_mask]

    return X_train, X_test, y_train, y_test, speaker_ids_train


# -------------------------------------------------------------------------- #
#                           3.  MODEL  FACTORY                               #
# -------------------------------------------------------------------------- #

def create_cnn_model(input_shape: tuple, cfg: dict):
    """Construiește un model CNN conform configurației specificate."""

    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))

    # --- straturi convoluționale ---
    for i in range(cfg["n_conv_layers"]):
        model.add(
            Conv2D(
                filters=cfg["filters"][i],
                kernel_size=cfg["kernel_sizes"][i],
                activation="relu",
                padding="same",
                kernel_regularizer=l2(cfg["l2_reg"]) if cfg["l2_reg"] else None,
            )
        )
        model.add(MaxPooling2D(pool_size=cfg["pool_sizes"][i]))
        if cfg["use_batch_norm"]:
            model.add(BatchNormalization())
        model.add(Dropout(cfg["dropout_rates"][i]))

    model.add(Flatten())

    # --- fully‑connected ---
    for units in cfg["dense_units"]:
        model.add(
            Dense(
                units,
                activation="relu",
                kernel_regularizer=l2(cfg["l2_reg"]) if cfg["l2_reg"] else None,
            )
        )
        if cfg["use_batch_norm"]:
            model.add(BatchNormalization())
        model.add(Dropout(cfg["dense_dropout"]))

    # --- ieșire binară ---
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(learning_rate=cfg["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# -------------------------------------------------------------------------- #
#                       4.  CONFIG  GENERATION (>=20)                        #
# -------------------------------------------------------------------------- #

def generate_cnn_configs(n_configs: int = 20, seed: int = 42):
    """Generează minimum 20 configurații CNN distincte, variind număr straturi,
    filtrele, kernel‑urile, dropout‑ul etc., conform cerințelor.
    """
    rnd = np.random.RandomState(seed)
    configs: list[dict] = []
    kernel_space = [(3, 3), (5, 5), (7, 7)]
    filter_space = [16, 32, 64, 128, 256]
    while len(configs) < n_configs:
        n_conv = rnd.randint(2, 5)                     # 2‑4 straturi conv.
        filters = rnd.choice(filter_space, size=n_conv, replace=True).tolist()
        kernels = [kernel_space[i] for i in rnd.choice(len(kernel_space), size=n_conv)]
        drop_conv = rnd.uniform(0.20, 0.50, size=n_conv).round(2).tolist()
        cfg = {
            "n_conv_layers": n_conv,
            "filters": filters,
            "kernel_sizes": kernels,
            "pool_sizes": [(2, 2)] * n_conv,
            "dropout_rates": drop_conv,
            "dense_units": rnd.choice([64, 128, 256], size=rnd.randint(1, 3), replace=False).tolist(),
            "dense_dropout": round(rnd.uniform(0.30, 0.60), 2),
            "learning_rate": float(rnd.choice([1e-3, 5e-4, 1e-4])),
            "l2_reg": float(rnd.choice([0.0, 0.001, 0.005, 0.01])),
            "use_batch_norm": bool(rnd.choice([True, False])),
            "epochs": int(rnd.choice([30, 40, 50])),
            "batch_size": int(rnd.choice([16, 32])),
        }
        # eliminăm duplicatele (hash‑uim dictul după sortare)
        cfg_hash = str(sorted(cfg.items()))
        if cfg_hash not in {str(sorted(c.items())) for c in configs}:
            configs.append(copy.deepcopy(cfg))
    return configs


# -------------------------------------------------------------------------- #
#                      5.  TRAIN &  CROSS‑VALIDATION                         #
# -------------------------------------------------------------------------- #

def evaluate_model(X, y, speaker_ids, cfg):
    """5‑fold GroupKFold (speaker‑independent). Returnează accuracy medie."""
    gkf = GroupKFold(n_splits=5)
    accs = []
    for tr_idx, val_idx in gkf.split(X, y, groups=speaker_ids):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        model = create_cnn_model(X_tr.shape[1:], cfg)
        es = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
        model.fit(
            X_tr,
            y_tr,
            validation_data=(X_val, y_val),
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            callbacks=[es],
            verbose=0,
        )
        y_hat = (model.predict(X_val, verbose=0) > 0.5).astype(int)
        accs.append(accuracy_score(y_val, y_hat))
    return float(np.mean(accs))


# -------------------------------------------------------------------------- #
#                              6.  MAIN                                      #
# -------------------------------------------------------------------------- #

def main():
    print("\n=== Încărcare spectrograme ===")
    spk_data, spk_labels, spk_genders = load_data()
    X, y, spk_ids = prepare_data(spk_data, spk_labels)
    if X.ndim == 3:                      # (samples, H, W)  -> (samples, H, W, 1)
        X = X[..., np.newaxis]

    X_train, X_test, y_train, y_test, spk_train_ids = split_data(
        X, y, spk_ids, spk_genders, test_size=TEST_SIZE, seed=RANDOM_SEED
    )

    print("\n=== Generare 20 configurații CNN distincte ===")
    cnn_cfgs = generate_cnn_configs(20, seed=RANDOM_SEED)
    print(f"Generat {len(cnn_cfgs)} configurații.")

    best_acc, best_cfg, best_model = 0.0, None, None

    print("\n=== 5‑fold CV pe fiecare configurație ===")
    for cfg in tqdm(cnn_cfgs, desc="Configurații CNN"):
        try:
            cv_acc = evaluate_model(X_train, y_train, spk_train_ids, cfg)
        except Exception as e:
            print(f"\n[WARN] Config skip din cauza erorii: {e}")
            continue
        if cv_acc > best_acc:
            best_acc, best_cfg = cv_acc, cfg
            # retrenăm pe tot TRAIN pentru test‑set
            best_model = create_cnn_model(X_train.shape[1:], best_cfg)
            best_model.fit(
                X_train,
                y_train,
                epochs=best_cfg["epochs"],
                batch_size=best_cfg["batch_size"],
                verbose=0,
            )
            print(f"\nNou cel mai bun ➜ acc={best_acc:.4f}\nCfg={best_cfg}\n")

    if best_model is None:
        raise RuntimeError("Niciun model valid nu a fost antrenat cu succes.")

    print("\n=== Evaluare pe TEST (speaker‑independent) ===")
    y_hat_test = (best_model.predict(X_test, verbose=0) > 0.5).astype(int)
    test_acc = accuracy_score(y_test, y_hat_test)

    print("\n================ REZULTATE FINALE ================")
    print(f"Configurație optimă: {best_cfg}")
    print(f"Accuratețe medie 5‑fold CV : {best_acc:.4f}")
    print(f"Accuratețe pe TEST        : {test_acc:.4f}")

    best_model.save("best_cnn_model.h5")
    print("Model salvat în 'best_cnn_model.h5'.")


if __name__ == "__main__":
    main()
