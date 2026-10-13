import numpy as np


def simulate_match(player_skill: float, npc_hp: int, npc_attack: int) -> dict:
    """
    Simulate a single fight and return outcome + fairness signal.

    Args:
        player_skill: 0.0-1.0 (higher = better player)
        npc_hp: Current NPC health
        npc_attack: Current NPC attack power

    Returns:
        Dict with win_rate, avg_dmg_taken, time_to_kill, fairness_score
    """
    # Simple deterministic combat sim with noise
    player_dps = 10 + 20 * player_skill
    effective_npc_hp = npc_hp * (1 - 0.3 * player_skill)  # Skilled players deal more effective dmg
    time_to_kill = max(1.0, effective_npc_hp / player_dps)
    
    npc_damage_per_sec = npc_attack * (0.7 + 0.6 * np.random.rand())
    total_dmg_taken = npc_damage_per_sec * time_to_kill
    
    # Win condition: kill NPC before taking >80% of max possible damage
    max_tolerable_dmg = 200 * (0.5 + 0.5 * player_skill)
    won = total_dmg_taken < max_tolerable_dmg
    win_rate = 1.0 if won else 0.0
    
    # Fairness score: 0 = perfectly balanced, 1 = completely unfair
    # Perfect balance = 50% win rate AND moderate damage taken
    ideal_dmg = 100
    dmg_deviation = abs(total_dmg_taken - ideal_dmg) / ideal_dmg
    fairness_score = 0.5 * abs(win_rate - 0.5) + 0.5 * min(dmg_deviation, 1.0)
    
    return {
        "win_rate": win_rate,
        "avg_dmg_taken": total_dmg_taken,
        "time_to_kill": time_to_kill,
        "fairness_score": fairness_score
    }