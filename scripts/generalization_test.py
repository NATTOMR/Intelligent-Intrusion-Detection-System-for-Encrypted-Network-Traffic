"""
Phase 8: Generalization Testing on UNSW-NB15 dataset.

This script evaluates the trained Hybrid IDS (XGBoost + Autoencoder)
on the UNSW-NB15 dataset, which has completely different feature names
and attack categories than CICIDS2017. The key challenge here is feature
alignment: we map the overlapping semantic features and use only those
to demonstrate real-world generalization.

Expected input files in data/raw/:
  - UNSW_NB15_training-set.csv
  - UNSW_NB15_testing-set.csv  (or just one combined file)
"""
import pandas as pd
import numpy as np
import joblib
import json
import os
import glob
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# ── Autoencoder definition (must match training) ──────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ── Feature mapping: UNSW-NB15 column → CICIDS2017 equivalent ───────────────
#  Only semantically equivalent features are mapped. All others are dropped.
FEATURE_MAP = {
    'dur':            'Flow Duration',
    'sbytes':         'Total Length of Fwd Packets',
    'dbytes':         'Total Length of Bwd Packets',
    'spkts':          'Total Fwd Packets',
    'dpkts':          'Total Backward Packets',
    'smean':          'Fwd Packet Length Mean',
    'dmean':          'Bwd Packet Length Mean',
    'sttl':           'Fwd Header Length',
    'dttl':           'Bwd Header Length',
    'synack':         'Flow IAT Mean',
    'ackdat':         'Flow IAT Std',
    'tcprtt':         'Flow IAT Max',
    'sjit':           'Fwd IAT Mean',
    'djit':           'Bwd IAT Mean',
    'ct_srv_src':     'Fwd Packets/s',
    'ct_srv_dst':     'Bwd Packets/s',
    'ct_dst_ltm':     'Active Mean',
    'ct_src_ltm':     'Active Std',
    'ct_src_dport_ltm': 'Idle Mean',
    'ct_dst_sport_ltm': 'Idle Std',
    'Ltime':          'Flow IAT Min',
    'Stime':          'Fwd IAT Total',
    'dwin':           'Init_Win_bytes_backward',
    'swin':           'Init_Win_bytes_forward',
}

def load_unsw_nb15(raw_dir):
    """Load all UNSW-NB15 CSV files found in raw_dir."""
    patterns = [
        os.path.join(raw_dir, 'UNSW_NB15*.csv'),
        os.path.join(raw_dir, 'unsw*.csv'),
        os.path.join(raw_dir, 'UNSW*.csv'),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = [f for f in files if 'combine' not in f.lower()]   # skip CICIDS

    if not files:
        raise FileNotFoundError(
            "No UNSW-NB15 CSV files found in data/raw/.\n"
            "Please download from https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15\n"
            "and place the CSV files in data/raw/ before running this script."
        )

    print(f"Found UNSW-NB15 files: {[os.path.basename(f) for f in files]}")
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    return pd.concat(dfs, ignore_index=True)

def run_generalization_test():
    models_dir  = 'models'
    results_dir = 'results'

    # 1. Load UNSW-NB15
    print("Loading UNSW-NB15 dataset …")
    df = load_unsw_nb15(os.path.join('data', 'raw'))
    df.columns = df.columns.str.strip()
    print(f"UNSW-NB15 shape: {df.shape}")
    print(f"Columns (sample): {list(df.columns[:10])}")

    # 2. Identify label column (UNSW-NB15 uses 'label' or 'attack_cat')
    label_col = None
    for candidate in ['label', 'Label', 'attack_cat', 'class']:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        raise ValueError("Could not find label column in UNSW-NB15 dataset.")

    print(f"Using label column: '{label_col}'")
    print(f"Label distribution:\n{df[label_col].value_counts()}")

    # Binary encode: 0 = normal, 1 = attack
    if df[label_col].dtype == object:
        df['Label_bin'] = (df[label_col].str.strip().str.lower() != 'normal').astype(int)
    else:
        df['Label_bin'] = (df[label_col] != 0).astype(int)

    y_true = df['Label_bin'].values

    # 3. Load models
    print("\nLoading trained models …")
    xgb_model = joblib.load(os.path.join(models_dir, 'xgboost_baseline.pkl'))
    ae_scaler  = joblib.load(os.path.join(models_dir, 'ae_scaler.pkl'))
    ae_features_trained = ae_scaler.feature_names_in_  # CICIDS2017 feature names

    with open(os.path.join(results_dir, 'ae_threshold.json')) as f:
        ae_threshold = json.load(f)['threshold']

    ae_model = Autoencoder(len(ae_features_trained))
    ae_model.load_state_dict(torch.load(
        os.path.join(models_dir, 'autoencoder.pth'), map_location='cpu'))
    ae_model.eval()

    # 4. Feature alignment for XGBoost (needs all 78 CICIDS2017 features)
    xgb_features_trained = xgb_model.get_booster().feature_names

    # Map UNSW columns → CICIDS equivalents
    df_mapped = df.rename(columns=FEATURE_MAP)

    # Build aligned feature set for XGBoost — fill missing cols with 0
    X_xgb = pd.DataFrame(0.0, index=df_mapped.index, columns=xgb_features_trained)
    for col in xgb_features_trained:
        if col in df_mapped.columns:
            X_xgb[col] = pd.to_numeric(df_mapped[col], errors='coerce').fillna(0)

    # 5. Feature alignment for Autoencoder (needs 39 optimized CICIDS2017 features)
    X_ae = pd.DataFrame(0.0, index=df_mapped.index, columns=ae_features_trained)
    for col in ae_features_trained:
        if col in df_mapped.columns:
            X_ae[col] = pd.to_numeric(df_mapped[col], errors='coerce').fillna(0)

    # 6. Run Hybrid IDS
    print("\nRunning Stage 1 — XGBoost …")
    xgb_preds = xgb_model.predict(X_xgb)

    print("Running Stage 2 — Autoencoder on XGBoost-benign traffic …")
    benign_idx = np.where(xgb_preds == 0)[0]
    final_preds = np.copy(xgb_preds)

    if len(benign_idx) > 0:
        X_ae_sub = ae_scaler.transform(X_ae.iloc[benign_idx]).astype(np.float32)
        tensor = torch.from_numpy(X_ae_sub)
        with torch.no_grad():
            recon = ae_model(tensor)
            mse   = torch.mean((tensor - recon) ** 2, dim=1).numpy()
        anomaly_flags = (mse > ae_threshold).astype(int)
        print(f"  Autoencoder caught {anomaly_flags.sum()} extra anomalies.")
        final_preds[benign_idx] = anomaly_flags

    # 7. Metrics
    print("\n=== Generalization Results on UNSW-NB15 ===")
    acc  = accuracy_score(y_true, final_preds)
    prec = precision_score(y_true, final_preds, zero_division=0)
    rec  = recall_score(y_true, final_preds, zero_division=0)
    f1   = f1_score(y_true, final_preds, zero_division=0)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nFull Classification Report:")
    print(classification_report(y_true, final_preds, target_names=['Normal', 'Attack']))

    # 8. Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, final_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title('Hybrid IDS — Generalization Test (UNSW-NB15)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'generalization_confusion_matrix.png'))
    plt.close()

    # Save metrics
    gen_metrics = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}
    with open(os.path.join(results_dir, 'generalization_metrics.json'), 'w') as f:
        json.dump(gen_metrics, f, indent=4)

    print(f"\nGeneralization test complete! Results saved to {results_dir}/")

if __name__ == "__main__":
    run_generalization_test()
