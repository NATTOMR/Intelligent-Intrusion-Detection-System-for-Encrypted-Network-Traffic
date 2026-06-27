import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def optimize_features():
    data_path = os.path.join('data', 'processed', 'cleaned_data.csv')
    model_path = os.path.join('models', 'xgboost_baseline.pkl')
    output_path = os.path.join('data', 'processed', 'optimized_data.csv')
    results_dir = 'results'
    
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['Label'])
    
    print(f"Loading trained model from {model_path}...")
    model = joblib.load(model_path)
    
    print("Extracting feature importances...")
    importances = model.feature_importances_
    feature_names = X.columns
    
    # Create DataFrame of importances
    imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    imp_df = imp_df.sort_values(by='Importance', ascending=False)
    
    # Calculate cumulative importance
    imp_df['Cumulative_Importance'] = imp_df['Importance'].cumsum()
    
    # Plot feature importances
    print("Plotting feature importances...")
    plt.figure(figsize=(12, 8))
    
    # Plot top 30 individual features for readability
    top_30 = imp_df.head(30)
    sns.barplot(x='Importance', y='Feature', data=top_30, palette='viridis')
    plt.title('Top 30 Feature Importances (XGBoost)')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_importance.png'))
    plt.close()
    
    # Plot Cumulative Importance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(imp_df) + 1), imp_df['Cumulative_Importance'], marker='o')
    plt.axhline(y=0.99, color='r', linestyle='--', label='99% Explained Importance')
    plt.title('Cumulative Feature Importance')
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Importance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'cumulative_importance.png'))
    plt.close()
    
    # Select features that contribute up to 99% cumulative importance
    # Or at least select a minimum number of features to avoid pruning too much
    threshold = 0.99
    selected_features = imp_df[imp_df['Cumulative_Importance'] <= threshold]['Feature'].tolist()
    
    # In case 99% is reached by just 1 or 2 features (which can happen with XGBoost),
    # let's ensure we keep at least the top 15 features to maintain robust models later.
    if len(selected_features) < 15:
        selected_features = imp_df['Feature'].head(15).tolist()
        
    print(f"\nOriginal feature count: {len(feature_names)}")
    print(f"Selected feature count: {len(selected_features)}")
    print("Top selected features:")
    for f in selected_features[:10]:
        print(f" - {f}")
        
    # Reduce dataset to optimized features + Label
    print(f"\nSaving optimized dataset to {output_path}...")
    optimized_df = df[selected_features + ['Label']]
    optimized_df.to_csv(output_path, index=False)
    print("Feature Optimization Complete!")

if __name__ == "__main__":
    optimize_features()
