import numpy as np
from scaler import StandardScaler
from gradient_descent import train_linear_regression, online_update

class LinearModel:
  """From-scratch Linear Regression with automatic feature scaling."""
  def __init__(self, learning_rate=0.01):
    self.learning_rate = learning_rate
    self.weights = None
    self.bias = None
    self.scaler = StandardScaler()
    self.is_fitted = False
  
  def fit(self, X, y, iterations=1000):
    """Fit the linear model to the training data."""
    X_scaled = self.scaler.fit_transform(X)
    self.weights, self.bias, _ = train_linear_regression(
      X_scaled, y, self.learning_rate, iterations
    )
    self.is_fitted = True
    return self
  
  def predict(self, X):
    """Predict using scaled features."""
    if not self.is_fitted:
        raise RuntimeError("Model not fitted yet. Call fit() first.")
    X_scaled = self.scaler.transform(X)
    return X_scaled.dot(self.weights) + self.bias
  
  def partial_fit(self, X_sample, y_sample):
    """Online update after single match (real-time adaptation)."""
    if not self.is_fitted:
        raise RuntimeError("Model not fitted yet. Call fit() first.")
    X_scaled = self.scaler.transform_single(X_sample)
    self.weights, self.bias = online_update(
        X_scaled, y_sample, self.weights, self.bias, self.learning_rate
    )
    return self