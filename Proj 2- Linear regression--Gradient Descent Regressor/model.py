import numpy as np

class LinearRegressor:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.n_iters = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # 1. Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 2. Gradient Descent loop
        for _ in range(self.n_iters):
            # Prediction (y_hat = Xw + b)
            y_predicted = np.dot(X, self.weights) + self.bias

            # Calculate gradients (The "direction" to move)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias