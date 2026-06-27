import pandas as pd
import numpy as np
import os
import gc

def preprocess_data(input_path, output_path):
    print(f"Loading data from {input_path}...")
    
    # Read the dataset
    df = pd.read_csv(input_path)
    print(f"Initial shape: {df.shape}")
    
    # 1. Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()
    
    # 2. Remove unnecessary columns if they exist
    cols_to_drop = ['Flow ID', 'Timestamp']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    # 3. Handle missing and infinite values
    print("Handling missing and infinite values...")
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows with NaN values
    df.dropna(inplace=True)
    print(f"Shape after dropping NaNs and Infs: {df.shape}")
    
    # 4. Encode labels
    print("Encoding labels (BENIGN = 0, Attack = 1)...")
    if 'Label' in df.columns:
        # Check unique labels before encoding
        print("Label distribution before encoding:")
        print(df['Label'].value_counts())
        
        # Binary encoding
        df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
        
        print("Label distribution after encoding:")
        print(df['Label'].value_counts())
    else:
        print("Warning: 'Label' column not found!")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the processed data
    print(f"Saving processed data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Data preprocessing complete!")

if __name__ == "__main__":
    input_file = os.path.join('data', 'raw', 'combine.csv')
    output_file = os.path.join('data', 'processed', 'cleaned_data.csv')
    
    if os.path.exists(input_file):
        preprocess_data(input_file, output_file)
    else:
        print(f"Error: Could not find input file at {input_file}")
