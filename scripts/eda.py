import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(input_path, results_dir):
    print(f"Loading data from {input_path} for EDA...")
    df = pd.read_csv(input_path)
    
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Class Distribution
    print("Plotting class distribution...")
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='Label', data=df, palette='Set2')
    plt.title('Class Distribution (0 = Benign, 1 = Attack)')
    plt.xlabel('Label')
    plt.ylabel('Count')
    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='baseline', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'class_distribution.png'))
    plt.close()
    
    # 2. Correlation Analysis
    print("Calculating correlations...")
    # Calculate correlation matrix with the target variable 'Label'
    corr_matrix = df.corr()
    
    # Get the top 20 features most correlated with 'Label' (absolute value)
    target_corr = corr_matrix['Label'].abs().sort_values(ascending=False)
    # Drop the Label itself
    top_features = target_corr.index[1:21]
    
    # Create a smaller correlation matrix for just these top 20 + Label
    cols_to_plot = list(top_features) + ['Label']
    small_corr_matrix = df[cols_to_plot].corr()
    
    print("Plotting correlation heatmap for top 20 features...")
    plt.figure(figsize=(16, 12))
    sns.heatmap(small_corr_matrix, annot=False, cmap='coolwarm', fmt=".2f",
                linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap (Top 20 Features vs Label)')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'correlation_heatmap.png'))
    plt.close()
    
    # 3. Feature Histograms for top 5 features
    print("Plotting histograms for top 5 features...")
    top_5_features = top_features[:5]
    
    # Create subplots
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for i, feature in enumerate(top_5_features):
        # We use a log scale because network traffic features often have extreme outliers
        sns.histplot(data=df, x=feature, hue='Label', bins=50, ax=axes[i], palette='Set2', log_scale=(False, True))
        axes[i].set_title(f'{feature} Distribution')
        axes[i].set_ylabel('Log Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'top_features_histograms.png'))
    plt.close()
    
    print(f"EDA complete! Plots saved to {results_dir}")

if __name__ == "__main__":
    input_file = os.path.join('data', 'processed', 'cleaned_data.csv')
    results_folder = 'results'
    
    if os.path.exists(input_file):
        run_eda(input_file, results_folder)
    else:
        print(f"Error: Processed data not found at {input_file}. Please run preprocess.py first.")
