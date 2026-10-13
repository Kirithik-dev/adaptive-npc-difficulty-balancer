import numpy as np

class StandardScaler:
  """Normalize features to mean=0, std=1 for stable gradient descent."""
  def __init__(self):
    self.mean = None
    self.std = None
  def fit_transform(self, X):
    """Learn stats from training data AND scale it immediately."""
    self.mean = np.mean(X, axis=0)
    self.std = np.std(X, axis=0)
    # Prevent division by zero if a feature has no variance
    self.std[self.std == 0] = 1.0
    return (X - self.mean) / self.std
  def transform(self, X):
    """Scale new batch using previously learned stats."""
    return (X - self.mean) / self.std
  def transform_single(self, x):
    """Scale one sample without recomputing stats."""
    return (x - self.mean) / self.std