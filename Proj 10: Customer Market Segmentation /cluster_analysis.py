import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def run_market_segmentation():
    # 1. Generate Local Synthetic Customer Data (Income vs Spending Score)
    np.random.seed(42)
    income = np.concatenate([np.random.normal(30, 5, 50), np.random.normal(70, 8, 50), np.random.normal(100, 10, 50)])
    spending = np.concatenate([np.random.normal(20, 5, 50), np.random.normal(60, 8, 50), np.random.normal(80, 5, 50)])
    
    df = pd.DataFrame({
        'Annual_Income_k': income,
        'Spending_Score': spending
    })

    # 2. Initialize and Train K-Means Model
    # We choose 3 clusters to segment customers into Low, Medium, and High value tiers
    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df[['Annual_Income_k', 'Spending_Score']])
    centroids = kmeans.cluster_centers_

    print("📊 Training Unsupervised K-Means Model...")
    print(f"✅ Grouped data into {num_clusters} distinct customer segments.\n")

    # 3. Print Cluster Summary Metrics
    for i in range(num_clusters):
        cluster_data = df[df['Cluster'] == i]
        print(f"🔹 Cluster {i+1} Size: {len(cluster_data)} customers")
        print(f"   Average Income: ${cluster_data['Annual_Income_k'].mean():.1f}k")
        print(f"   Average Spending Score: {cluster_data['Spending_Score'].mean():.1f}/100\n")

    # 4. Generate and Save Cluster Visual Chart
    plt.figure(figsize=(8, 6))
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
    
    for i in range(num_clusters):
        cluster_set = df[df['Cluster'] == i]
        plt.scatter(cluster_set['Annual_Income_k'], cluster_set['Spending_Score'], 
                    c=colors[i], label=f'Segment {i+1}', s=50, alpha=0.7)
        
    # Plot the calculated mathematical midpoints (Centroids)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='Centroids')
    
    plt.title('Customer Market Segmentation (K-Means)')
    plt.xlabel('Annual Income ($k)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save chart locally to folder
    plt.savefig('customer_clusters.png', dpi=300)
    print("💾 Visualization saved locally as 'customer_clusters.png'")

if __name__ == "__main__":
    run_market_segmentation()
  
