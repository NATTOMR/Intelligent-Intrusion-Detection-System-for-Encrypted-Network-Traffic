import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)

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

def get_ae_scores(ae_model, ae_scaler, X, ae_features, device):
    """Returns reconstruction MSE (used as anomaly score) for every row."""
    X_sub = ae_scaler.transform(X[ae_features]).astype(np.float32)
    tensor = torch.from_numpy(X_sub).to(device)
    ae_model.eval()
    with torch.no_grad():
        recon = ae_model(tensor)
        mse = torch.mean((tensor - recon) ** 2, dim=1).cpu().numpy()
    return mse

def run_evaluation():
    models_dir  = 'models'
    results_dir = 'results'

    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading full dataset …")
    df_full = pd.read_csv(os.path.join('data', 'processed', 'cleaned_data.csv'))
    X_full  = df_full.drop(columns=['Label'])
    y_full  = df_full['Label']
    _, X_test_full, _, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)

    # ── Load models ──────────────────────────────────────────────────────────
    print("Loading models …")
    rf_model  = joblib.load(os.path.join(models_dir, 'random_forest_baseline.pkl'))
    xgb_model = joblib.load(os.path.join(models_dir, 'xgboost_baseline.pkl'))
    ae_scaler = joblib.load(os.path.join(models_dir, 'ae_scaler.pkl'))
    ae_features = ae_scaler.feature_names_in_

    with open(os.path.join(results_dir, 'ae_threshold.json')) as f:
        ae_threshold = json.load(f)['threshold']

    device = torch.device("cpu")
    ae_model = Autoencoder(len(ae_features))
    ae_model.load_state_dict(torch.load(
        os.path.join(models_dir, 'autoencoder.pth'), map_location=device))

    # ── Predictions & scores ─────────────────────────────────────────────────
    print("Running Random Forest …")
    rf_preds  = rf_model.predict(X_test_full)
    rf_proba  = rf_model.predict_proba(X_test_full)[:, 1]

    print("Running XGBoost …")
    xgb_preds = xgb_model.predict(X_test_full)
    xgb_proba = xgb_model.predict_proba(X_test_full)[:, 1]

    print("Running Autoencoder …")
    ae_scores = get_ae_scores(ae_model, ae_scaler, X_test_full, ae_features, device)
    ae_preds  = (ae_scores > ae_threshold).astype(int)
    # Normalise MSE to [0,1] for ROC curve
    ae_proba  = (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min() + 1e-9)

    print("Running Hybrid IDS …")
    hybrid_preds = np.copy(xgb_preds)
    benign_idx   = np.where(xgb_preds == 0)[0]
    ae_sub_scores = get_ae_scores(ae_model, ae_scaler,
                                   X_test_full.iloc[benign_idx],
                                   ae_features, device)
    hybrid_preds[benign_idx] = (ae_sub_scores > ae_threshold).astype(int)
    # Hybrid score = max of XGBoost prob and normalised AE score
    hybrid_proba = np.copy(xgb_proba)
    ae_sub_norm  = (ae_sub_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min() + 1e-9)
    hybrid_proba[benign_idx] = np.maximum(xgb_proba[benign_idx], ae_sub_norm)

    # ── Metrics table ────────────────────────────────────────────────────────
    def metrics(y_true, y_pred, y_score):
        return {
            'Accuracy':  round(accuracy_score(y_true, y_pred),  4),
            'Precision': round(precision_score(y_true, y_pred), 4),
            'Recall':    round(recall_score(y_true, y_pred),    4),
            'F1-Score':  round(f1_score(y_true, y_pred),        4),
            'ROC-AUC':   round(roc_auc_score(y_true, y_score),  4),
        }

    results = {
        'Random Forest': metrics(y_test, rf_preds,     rf_proba),
        'XGBoost':       metrics(y_test, xgb_preds,    xgb_proba),
        'Autoencoder':   metrics(y_test, ae_preds,      ae_proba),
        'Hybrid IDS':    metrics(y_test, hybrid_preds,  hybrid_proba),
    }

    with open(os.path.join(results_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)

    print("\n=== Evaluation Results ===")
    results_df = pd.DataFrame(results).T
    print(results_df.to_string())

    # ── ROC Curves ───────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    fig = plt.figure(figsize=(18, 6))
    gs  = gridspec.GridSpec(1, 2, figure=fig)

    # ROC curves
    ax1 = fig.add_subplot(gs[0, 0])
    models_roc = {
        'Random Forest': (rf_proba,     '#4CAF50'),
        'XGBoost':       (xgb_proba,    '#2196F3'),
        'Autoencoder':   (ae_proba,     '#FF9800'),
        'Hybrid IDS':    (hybrid_proba, '#9C27B0'),
    }
    for name, (score, color) in models_roc.items():
        fpr, tpr, _ = roc_curve(y_test, score)
        auc_val = roc_auc_score(y_test, score)
        ax1.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.4f})', color=color, linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax1.set_title('ROC-AUC Curves — All Models', fontsize=13, fontweight='bold')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3)

    # Metrics bar chart
    ax2  = fig.add_subplot(gs[0, 1])
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    x    = np.arange(len(metrics_to_plot))
    width = 0.2
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
    model_names = list(results.keys())
    for i, (model, color) in enumerate(zip(model_names, colors)):
        vals = [results[model][m] for m in metrics_to_plot]
        bars = ax2.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(metrics_to_plot, fontsize=10)
    ax2.set_ylim([0.88, 1.01])
    ax2.set_title('Metrics Comparison — All Models', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_comparison.png'), dpi=120)
    plt.close()
    print("Evaluation complete! Plots saved to results/")

if __name__ == "__main__":
    run_evaluation()
