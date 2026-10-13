import numpy as np


class PlayerTracker:
    """Records player performance metrics after each match."""

    def __init__(self):
        self.history = []  # List of dicts: {win_rate, avg_dmg_taken, time_to_kill}

    def record_match(self, win_rate: float, avg_dmg_taken: float, time_to_kill: float):
        """Log a single match's stats (all values should be raw/unnormalized)."""
        self.history.append({
            "win_rate": win_rate,
            "avg_dmg_taken": avg_dmg_taken,
            "time_to_kill": time_to_kill
        })

    def get_latest_features(self) -> np.ndarray:
        """Return most recent match as feature vector [win_rate, dmg, ttk]."""
        if not self.history:
            raise RuntimeError("No matches recorded yet.")
        latest = self.history[-1]
        return np.array([latest["win_rate"], latest["avg_dmg_taken"], latest["time_to_kill"]])

    def get_batch_features(self) -> np.ndarray:
        """Return all historical matches as 2D array for initial training."""
        if not self.history:
            raise RuntimeError("No matches recorded yet.")
        return np.array([[m["win_rate"], m["avg_dmg_taken"], m["time_to_kill"]] 
                         for m in self.history])