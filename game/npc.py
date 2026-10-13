import numpy as np


class AdaptiveNPC:
    """NPC whose stats are dynamically set by the ML model."""

    BASE_HP = 100
    BASE_ATTACK = 15

    def __init__(self, model):
        self.model = model          # Your LinearModel instance
        self.current_hp = self.BASE_HP
        self.current_attack = self.BASE_ATTACK

    def adjust_difficulty(self, player_features: np.ndarray):
        """Predict optimal HP multiplier from player stats and apply it."""
        prediction = self.model.predict(player_features.reshape(1, -1))[0]
        # Clamp prediction to reasonable range [0.5x, 3.0x base HP]
        hp_multiplier = np.clip(prediction, 0.5, 3.0)
        self.current_hp = int(self.BASE_HP * hp_multiplier)
        self.current_attack = int(self.BASE_ATTACK * (0.8 + 0.4 * hp_multiplier))
        return self.current_hp, self.current_attack

    def get_stats(self) -> dict:
        return {"hp": self.current_hp, "attack": self.current_attack}