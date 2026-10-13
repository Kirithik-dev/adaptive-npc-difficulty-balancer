import numpy as np
from .cost_function import compute_mse_gradient, compute_bias_gradient

def train_linear_regression(X, y, learning_rate=0.01, iterations=1000):
  """Train a linear regression model using gradient descent."""
  m, n = X.shape
  weights = np.zeros(n)
  bias = 0.0
  cost_history = []
  for i in range(iterations):
    y_pred = X.dot(weights) + bias
    dw = compute_mse_gradient(X, y, y_pred)
    wb = compute_bias_gradient(y, y_pred)
    weights -= learning_rate * dw
    bias -= learning_rate * wb
    cost_history.append(np.mean((y - y_pred) ** 2))
  return weights, bias, cost_history
  
def online_update(X_sample, y_sample, weights, bias, learning_rate=0.01):
  """Update weights after a single match (real-time adaptation)."""
  X_batch = X_sample.reshape(1, -1)
  y_batch = np.array([y_sample])
  
  y_pred = X_batch.dot(weights) + bias
  dw = compute_mse_gradient(X_batch, y_batch, y_pred)
  db = compute_bias_gradient(y_batch, y_pred)
  
  weights -= learning_rate * dw
  bias -= learning_rate * db  
  return weights, bias