import numpy as np

def compute_mse(y_true,y_pred):
  """Compute Mean Squared Error cost."""
  return float(np.mean((y_true - y_pred) ** 2))

def compute_mse_gradient(X, y_true, y_pred):
  """Compute gradient of MSE w.r.t. weights."""
  m=len(y_true)
  error = y_pred - y_true
  return (2/m) * np.dot(X.T, error)

def compute_bias_gradient(y_true, y_pred):
  """Compute gradient of MSE w.r.t. bias."""
  m=len(y_true)
  error = y_pred - y_true
  return (2/m) * np.sum(error)