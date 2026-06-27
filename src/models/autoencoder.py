import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.impute import SimpleImputer
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import matplotlib.pyplot as plt

def load_and_preprocess_data(data_dir, sample_frac=0.1):
    """Load, clean and prepare data for autoencoder."""
    print(f"Loading data from {data_dir}...")
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError("No CSV files found in the data directory.")
    
    df_list = []
    for file in files:
        filepath = os.path.join(data_dir, file)
        df = pd.read_csv(filepath).sample(frac=sample_frac, random_state=42)
        df_list.append(df)
        
    df = pd.concat(df_list, ignore_index=True)
    df.columns = df.columns.str.strip()
    
    label_col = 'Label'
    drop_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Timestamp']
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    
    X = df.drop(columns=[label_col] + cols_to_drop)
    y = df[label_col]
    y_binary = (y != 'Benign').astype(int)
    
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    return X_imputed, y_binary

def build_autoencoder(input_dim):
    """Build a shallow autoencoder."""
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoder = Dense(64, activation='relu')(input_layer)
    encoder = Dropout(0.1)(encoder)
    encoder = Dense(32, activation='relu')(encoder)
    
    # Decoder
    decoder = Dense(64, activation='relu')(encoder)
    decoder = Dropout(0.1)(decoder)
    decoder = Dense(input_dim, activation='linear')(decoder)
    
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

def train_autoencoder(model, X_train_benign, X_val_benign):
    print("Training Autoencoder on BENIGN data only...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train_benign, X_train_benign,
        epochs=50,
        batch_size=256,
        validation_data=(X_val_benign, X_val_benign),
        callbacks=[early_stopping],
        verbose=1
    )
    return history

def plot_reconstruction_error(mse_benign, mse_malicious, threshold):
    plt.figure(figsize=(10, 6))
    plt.hist(mse_benign, bins=50, alpha=0.6, label='Benign', color='blue')
    plt.hist(mse_malicious, bins=50, alpha=0.6, label='Malicious', color='red')
    plt.axvline(threshold, color='black', linestyle='dashed', linewidth=2, label=f'Threshold: {threshold:.4f}')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Count')
    plt.title('Reconstruction Error Distribution')
    plt.legend()
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig('reports/figures/autoencoder_mse_dist.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/raw', help='Path to raw data directory')
    args = parser.parse_args()
    
    try:
        X, y = load_and_preprocess_data(args.data_dir, sample_frac=0.1)
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # We only train autoencoder on Benign data (y == 0)
        X_train_benign = X_train[y_train == 0]
        
        # Split benign training data into train and validation for early stopping
        X_train_benign, X_val_benign = train_test_split(X_train_benign, test_size=0.1, random_state=42)
        
        # Scale features based on benign training data
        scaler = StandardScaler()
        X_train_benign_scaled = scaler.fit_transform(X_train_benign)
        X_val_benign_scaled = scaler.transform(X_val_benign)
        X_test_scaled = scaler.transform(X_test)
        
        # Build and train model
        input_dim = X_train_benign_scaled.shape[1]
        autoencoder = build_autoencoder(input_dim)
        history = train_autoencoder(autoencoder, X_train_benign_scaled, X_val_benign_scaled)
        
        # Evaluation
        print("Evaluating autoencoder...")
        # Get reconstruction errors for test set
        reconstructions = autoencoder.predict(X_test_scaled)
        mse = np.mean(np.power(X_test_scaled - reconstructions, 2), axis=1)
        
        # Determine threshold using Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, mse)
        # Find threshold that gives best F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        print(f"Optimal Threshold (based on F1): {best_threshold:.4f}")
        
        y_pred = (mse > best_threshold).astype(int)
        
        print("\n--- Autoencoder Classification Report ---")
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC: {roc_auc_score(y_test, mse):.4f}")
        
        # Plot distribution
        mse_benign = mse[y_test == 0]
        mse_malicious = mse[y_test == 1]
        plot_reconstruction_error(mse_benign, mse_malicious, best_threshold)
        
        # Save model
        os.makedirs('models', exist_ok=True)
        autoencoder.save('models/autoencoder.keras')
        joblib.dump(scaler, 'models/ae_scaler.pkl')
        print("Autoencoder model and scaler saved.")
        
    except Exception as e:
        print(f"Error during Autoencoder training: {e}")

if __name__ == "__main__":
    main()
