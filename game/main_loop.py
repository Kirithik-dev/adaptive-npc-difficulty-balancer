from src import LinearModel
from game.player_tracker import PlayerTracker
from game.npc import AdaptiveNPC
from game.arena import simulate_match

# Initialize
model = LinearModel(learning_rate=0.01)
tracker = PlayerTracker()
npc = AdaptiveNPC(model)

# Generate 50 fake matches for initial training
for _ in range(50):
    skill = np.random.rand()
    result = simulate_match(skill, npc.current_hp, npc.current_attack)
    tracker.record_match(result["win_rate"], result["avg_dmg_taken"], result["time_to_kill"])

# Train initially
X_batch = tracker.get_batch_features()
y_batch = np.array([0.5] * len(X_batch))  # Target: perfect fairness = 0.5
model.fit(X_batch, y_batch, iterations=500)

# Play 5 live matches with real-time adaptation
for i in range(5):
    skill = np.random.rand()
    features = tracker.get_latest_features() if tracker.history else np.array([0.5, 100, 10])
    npc.adjust_difficulty(features)
    result = simulate_match(skill, npc.current_hp, npc.current_attack)
    tracker.record_match(result["win_rate"], result["avg_dmg_taken"], result["time_to_kill"])
    model.partial_fit(tracker.get_latest_features(), result["fairness_score"])
    print(f"Match {i+1}: NPC HP={npc.current_hp}, Fairness={result['fairness_score']:.3f}")