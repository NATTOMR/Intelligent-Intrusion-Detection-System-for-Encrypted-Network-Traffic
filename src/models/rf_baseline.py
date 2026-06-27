import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
import joblib
import shap
import matplotlib.pyplot as plt

def load_data(data_dir, sample_frac=0.1):
    """Load and sample data for faster prototyping."""
    print(f"Loading data from {data_dir}...")
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError("No CSV files found in the data directory. Please download the dataset.")
    
    df_list = []
    for file in files:
        filepath = os.path.join(data_dir, file)
        # Load a sample directly if files are too large
        df = pd.read_csv(filepath, sample_frac=sample_frac) if hasattr(pd.read_csv, 'sample_frac') else pd.read_csv(filepath).sample(frac=sample_frac, random_state=42)
        df_list.append(df)
        
    full_df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(full_df)} rows.")
    return full_df

def preprocess_data(df):
    """Clean data and split into features and labels."""
    print("Preprocessing data...")
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Check for Label column
    label_col = 'Label'
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset.")
        
    # Drop features that cause 'shortcut learning' like IPs and Ports if they exist
    # Note: adjust these based on the actual columns in the dataset version you download
    drop_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Timestamp']
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    
    X = df.drop(columns=[label_col] + cols_to_drop)
    y = df[label_col]
    
    # Convert labels to binary (Benign = 0, Malicious = 1)
    y_binary = (y != 'Benign').astype(int)
    
    # Handle missing values and infinity
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    return X_imputed, y_binary, y

def train_baseline(X_train, y_train):
    print("Training Random Forest baseline model...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf

def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC:  {roc_auc_score(y_test, y_prob):.4f}")

def compute_feature_importance(model, X_train):
    print("Computing feature importance using SHAP...")
    # Use a small sample for SHAP to save time
    X_sample = shap.sample(X_train, 100)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    os.makedirs('reports/figures', exist_ok=True)
    
    # For binary classification, shap_values might be a list
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values
        
    shap.summary_plot(shap_values_to_plot, X_sample, show=False)
    plt.savefig('reports/figures/rf_shap_summary.png', bbox_inches='tight')
    plt.close()
    print("SHAP summary plot saved to reports/figures/rf_shap_summary.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/raw', help='Path to raw data directory')
    args = parser.parse_args()
    
    try:
        df = load_data(args.data_dir, sample_frac=0.1) # Use 10% for baseline prototyping
        X, y_binary, y_multi = preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        
        # Train and evaluate
        model = train_baseline(X_train_scaled, y_train)
        evaluate_model(model, X_test_scaled, y_test)
        
        # Feature Importance
        compute_feature_importance(model, X_train_scaled)
        
        # Save model and scaler
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/rf_baseline.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')
        print("Model and scaler saved to 'models/' directory.")
        
    except Exception as e:
        print(f"Error during baseline training: {e}")

if __name__ == "__main__":
    main()
