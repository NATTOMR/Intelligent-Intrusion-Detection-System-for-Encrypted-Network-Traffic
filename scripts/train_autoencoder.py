import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import os
import json

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8), # Bottleneck
            nn.ReLU()
        )
        # Decoder
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

def train_and_evaluate_autoencoder():
    data_path = os.path.join('data', 'processed', 'optimized_data.csv')
    models_dir = 'models'
    results_dir = 'results'
    
    print(f"Loading optimized dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Split into normal and anomalous
    normal_df = df[df['Label'] == 0].drop(columns=['Label'])
    anomaly_df = df[df['Label'] == 1].drop(columns=['Label'])
    
    print(f"Normal samples: {len(normal_df)}, Anomaly samples: {len(anomaly_df)}")
    
    # Split normal data into train and test
    X_train_normal, X_test_normal = train_test_split(normal_df, test_size=0.2, random_state=42)
    
    # Scale data
    print("Scaling data...")
    scaler = MinMaxScaler()
    X_train_normal_scaled = scaler.fit_transform(X_train_normal).astype(np.float32)
    X_test_normal_scaled = scaler.transform(X_test_normal).astype(np.float32)
    X_test_anomaly_scaled = scaler.transform(anomaly_df).astype(np.float32)
    
    input_dim = X_train_normal_scaled.shape[1]
    
    # Create DataLoaders
    batch_size = 512
    train_dataset = TensorDataset(torch.from_numpy(X_train_normal_scaled))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = Autoencoder(input_dim).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training Autoencoder (this may take a minute)...")
    epochs = 5
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.6f}")
        
    # Evaluate
    print("Calculating reconstruction errors...")
    model.eval()
    
    def get_reconstruction_error(data_tensor):
        with torch.no_grad():
            data_tensor = data_tensor.to(device)
            reconstruction = model(data_tensor)
            # MSE per sample
            mse = torch.mean((data_tensor - reconstruction) ** 2, dim=1).cpu().numpy()
        return mse
        
    mse_normal = get_reconstruction_error(torch.from_numpy(X_test_normal_scaled))
    mse_anomaly = get_reconstruction_error(torch.from_numpy(X_test_anomaly_scaled))
    
    # Set threshold at 99th percentile of normal training errors
    threshold = np.percentile(mse_normal, 99)
    print(f"Calculated Anomaly Threshold: {threshold}")
    
    # Plot reconstruction error
    print("Plotting reconstruction errors...")
    plt.figure(figsize=(10, 6))
    plt.hist(mse_normal, bins=50, alpha=0.6, color='blue', label='Normal (Test)', log=True)
    plt.hist(mse_anomaly, bins=50, alpha=0.6, color='red', label='Anomaly', log=True)
    plt.axvline(x=threshold, color='black', linestyle='--', label='Threshold')
    plt.title('Reconstruction Error Distribution (PyTorch Autoencoder)')
    plt.xlabel('Mean Squared Error (MSE)')
    plt.ylabel('Log Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'reconstruction_error.png'))
    plt.close()
    
    # Save Model and Scaler
    print(f"Saving Autoencoder and Scaler to {models_dir}...")
    torch.save(model.state_dict(), os.path.join(models_dir, 'autoencoder.pth'))
    joblib.dump(scaler, os.path.join(models_dir, 'ae_scaler.pkl'))
    
    # Save threshold
    with open(os.path.join(results_dir, 'ae_threshold.json'), 'w') as f:
        json.dump({'threshold': float(threshold)}, f)
        
    print("Phase 5 Autoencoder Training Complete!")

if __name__ == "__main__":
    train_and_evaluate_autoencoder()
