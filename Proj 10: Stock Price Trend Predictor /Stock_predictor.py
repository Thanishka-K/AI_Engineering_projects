import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def predict_stock_trends():
    # Generate Synthetic Rolling Time-Series Data (Simulated Stock Price)
    np.random.seed(42)
    days = 100
    time = np.arange(days)
    # Create a baseline upward trend with random realistic market noise
    price = 100 + (time * 0.5) + np.random.normal(0, 3, days)
    
    df = pd.DataFrame({'Day': time, 'Price': price})
    
    # Create a 1-Day Lag (Yesterday's price predicts Today's)
    df['Yesterday_Price'] = df['Price'].shift(1)
    df = df.dropna() # Remove the first row since it won't have a lag value

    X = df[['Yesterday_Price']]
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_test_split=0.2, shuffle=False)

    model = LinearRegression()
    print("📈 Training Time-Series Regression Model...")
    model.fit(X_train, y_train)
    print("✅ Model training complete.")

    df['Predicted_Price'] = model.predict(X)

    plt.figure(figsize=(10, 5))
    plt.plot(df['Day'], df['Price'], label='Actual Market Price', color='#1f77b4', alpha=0.8)
    plt.plot(df['Day'], df['Predicted_Price'], label='AI Trend Line', color='#d62728', linestyle='--')
    
    plt.title('Stock Price Trend Prediction Matrix')
    plt.xlabel('Timeline (Days)')
    plt.ylabel('Asset Value ($)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig('stock_predictions.png', dpi=300)
    print("💾 Analysis visualization saved as 'stock_predictions.png'")

if __name__ == "__main__":
    predict_stock_trends()
  
