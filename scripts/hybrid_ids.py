import pandas as pd
import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Define Autoencoder class exactly as trained
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8), 
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def run_hybrid_ids():
    models_dir = 'models'
    results_dir = 'results'
    data_path_full = os.path.join('data', 'processed', 'cleaned_data.csv')
    
    print("Loading full dataset for XGBoost...")
    df = pd.read_csv(data_path_full)
    X = df.drop(columns=['Label'])
    y = df['Label']
    
    # We must use the same split as before to evaluate fairly on the test set
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\n--- 1. Loading Models ---")
    xgb_model = joblib.load(os.path.join(models_dir, 'xgboost_baseline.pkl'))
    ae_scaler = joblib.load(os.path.join(models_dir, 'ae_scaler.pkl'))
    ae_features = ae_scaler.feature_names_in_
    
    with open(os.path.join(results_dir, 'ae_threshold.json'), 'r') as f:
        threshold = json.load(f)['threshold']
        
    device = torch.device("cpu")
    input_dim = len(ae_features)
    ae_model = Autoencoder(input_dim)
    ae_model.load_state_dict(torch.load(os.path.join(models_dir, 'autoencoder.pth'), map_location=device))
    ae_model.eval()
    
    print("\n--- 2. Executing Hybrid Decision Engine ---")
    
    # Step A: XGBoost predicts all traffic (Known Attacks)
    print("Running Stage 1 (XGBoost Classifier)...")
    xgb_preds = xgb_model.predict(X_test)
    
    # Track final predictions
    final_preds = np.copy(xgb_preds)
    
    # Find which packets XGBoost labeled as BENIGN (0)
    # These are the ones we need to double check with the Autoencoder
    benign_indices = np.where(xgb_preds == 0)[0]
    
    print(f"XGBoost flagged {np.sum(xgb_preds == 1)} attacks.")
    print(f"XGBoost labeled {len(benign_indices)} packets as benign. Passing these to Autoencoder...")
    
    if len(benign_indices) > 0:
        # Step B: Autoencoder checks the "Benign" traffic for Zero-Day anomalies
        # Subset to the features the Autoencoder was trained on
        X_test_benign_subset = X_test.iloc[benign_indices][ae_features]
        
        # Scale
        X_subset_scaled = ae_scaler.transform(X_test_benign_subset).astype(np.float32)
        X_tensor = torch.from_numpy(X_subset_scaled)
        
        print("Running Stage 2 (Autoencoder Anomaly Detection)...")
        with torch.no_grad():
            reconstruction = ae_model(X_tensor)
            mse = torch.mean((X_tensor - reconstruction) ** 2, dim=1).numpy()
            
        # Flag anomalies where MSE > threshold
        anomaly_flags = (mse > threshold).astype(int)
        anomalies_found = np.sum(anomaly_flags)
        print(f"Autoencoder caught {anomalies_found} unknown anomalies hiding in the benign traffic!")
        
        # Step C: Update Final Predictions
        # If anomaly_flags is 1, override the 0 in final_preds to 1
        final_preds[benign_indices] = anomaly_flags

    print("\n--- 3. Hybrid Evaluation ---")
    accuracy = accuracy_score(y_test, final_preds)
    precision = precision_score(y_test, final_preds)
    recall = recall_score(y_test, final_preds)
    f1 = f1_score(y_test, final_preds)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, final_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
    plt.title('Hybrid IDS Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'hybrid_confusion_matrix.png'))
    plt.close()
    
    print(f"\nHybrid testing complete. Confusion matrix saved to {results_dir}")

if __name__ == "__main__":
    run_hybrid_ids()
