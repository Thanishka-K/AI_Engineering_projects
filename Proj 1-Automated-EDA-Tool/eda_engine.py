import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def run_eda(file_path):
    # 1. Load Data
    df = pd.read_csv(file_path)
    print(f"--- Analysis for {file_path} ---")
    
    # 2. Basic Audit
    print("\n[+] Data Info:")
    print(df.info())
    
    print("\n[+] Missing Values:")
    print(df.isnull().sum())
    
    # 3. Create Output Directory
    output_dir = "eda_reports"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 4. Visualizations
    print("\n[+] Generating Visuals...")
    
    # Correlation Heatmap (Only for numeric columns)
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['number'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()

    # Distribution Plots
    for col in numeric_df.columns[:5]: # Limit to first 5 columns for speed
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.savefig(f"{output_dir}/{col}_dist.png")
        plt.close()

    print(f"\n[SUCCESS] Reports saved in /{output_dir}")

if __name__ == "__main__":
    # Change 'sample_data.csv' to your actual file name
    # run_eda("sample_data.csv") 
    print("Engine ready. Provide a CSV to begin.")
