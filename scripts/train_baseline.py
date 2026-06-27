import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
import joblib
import os
import json

def train_and_evaluate():
    data_path = os.path.join('data', 'processed', 'cleaned_data.csv')
    models_dir = 'models'
    results_dir = 'results'
    
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Separate features and target
    X = df.drop(columns=['Label'])
    y = df['Label']
    
    print("Splitting data into 80% train and 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    results = {}

    # --- Random Forest ---
    print("\nTraining Random Forest...")
    # Using n_estimators=50 and n_jobs=-1 for faster baseline training
    rf = RandomForestClassifier(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    
    print("Evaluating Random Forest...")
    rf_preds = rf.predict(X_test)
    results['Random_Forest'] = {
        'Accuracy': accuracy_score(y_test, rf_preds),
        'Precision': precision_score(y_test, rf_preds),
        'Recall': recall_score(y_test, rf_preds),
        'F1_Score': f1_score(y_test, rf_preds)
    }
    
    print("Saving Random Forest model...")
    joblib.dump(rf, os.path.join(models_dir, 'random_forest_baseline.pkl'))

    # --- XGBoost ---
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    print("Evaluating XGBoost...")
    xgb_preds = xgb_model.predict(X_test)
    results['XGBoost'] = {
        'Accuracy': accuracy_score(y_test, xgb_preds),
        'Precision': precision_score(y_test, xgb_preds),
        'Recall': recall_score(y_test, xgb_preds),
        'F1_Score': f1_score(y_test, xgb_preds)
    }
    
    print("Saving XGBoost model...")
    joblib.dump(xgb_model, os.path.join(models_dir, 'xgboost_baseline.pkl'))

    # --- Save Results ---
    print("\nTraining Complete! Saving results...")
    with open(os.path.join(results_dir, 'baseline_metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\n--- Baseline Results ---")
    for model, metrics in results.items():
        print(f"\n{model}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

if __name__ == "__main__":
    train_and_evaluate()
