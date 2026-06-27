"""
Phase 9: SHAP Explainability Visualization.

Generates three SHAP plots that explain the XGBoost model's decisions:
  1. Summary Bar Plot  – global mean |SHAP| importance across all features
  2. Beeswarm Plot     – distribution of SHAP values per feature (shows direction)
  3. Waterfall Plot    – local explanation for a single attack prediction
"""
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

def run_shap():
    models_dir  = 'models'
    results_dir = 'results'

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data …")
    df = pd.read_csv(os.path.join('data', 'processed', 'cleaned_data.csv'))
    X  = df.drop(columns=['Label'])
    y  = df['Label']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Use a small balanced sample to keep SHAP fast on CPU
    # 500 benign + 500 attack rows
    n_sample = 500
    idx_benign = y_test[y_test == 0].index[:n_sample]
    idx_attack = y_test[y_test == 1].index[:n_sample]
    sample_idx = idx_benign.append(idx_attack)
    X_sample   = X_test.loc[sample_idx]
    y_sample   = y_test.loc[sample_idx]

    # ── Load XGBoost model ────────────────────────────────────────────────────
    print("Loading XGBoost model …")
    model = joblib.load(os.path.join(models_dir, 'xgboost_baseline.pkl'))

    # ── Compute SHAP values ───────────────────────────────────────────────────
    print("Computing SHAP values (TreeExplainer — fast) …")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)   # returns shap.Explanation object

    # For binary XGBoost the Explanation already contains values for class 1
    # shap_values.values shape: (n_samples, n_features)

    # ── Plot 1: Summary Bar Plot (mean |SHAP|) ────────────────────────────────
    print("Generating Plot 1 — Summary Bar …")
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.title("SHAP Feature Importance — Mean |SHAP| (Top 20)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'shap_summary_bar.png'), dpi=120, bbox_inches='tight')
    plt.close()

    # ── Plot 2: Beeswarm Plot ──────────────────────────────────────────────────
    print("Generating Plot 2 — Beeswarm …")
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title("SHAP Beeswarm — Feature Impact Distribution (Top 20)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'shap_beeswarm.png'), dpi=120, bbox_inches='tight')
    plt.close()

    # ── Plot 3: Waterfall Plot (single attack explanation) ─────────────────────
    print("Generating Plot 3 — Waterfall (single attack prediction) …")
    # Pick the first attack sample
    attack_local_idx = int(np.where(y_sample.values == 1)[0][0])
    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(shap_values[attack_local_idx], max_display=15, show=False)
    plt.title("SHAP Waterfall — Why this packet was flagged as ATTACK", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'shap_waterfall.png'), dpi=120, bbox_inches='tight')
    plt.close()

    print(f"\nPhase 9 complete! All SHAP plots saved to {results_dir}/")

if __name__ == "__main__":
    run_shap()
