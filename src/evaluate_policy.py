# evaluate_policy.py

import sys
import os

# Projektstruktur für Import anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Drittanbieter
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Lokale Module
from config import ENV_MODE, EPISODES, MAX_STEPS, LOOP_THRESHOLD, LOOP_PENALTY, REWARDS
from navigation.environment.grid_environment import GridEnvironment
from navigation.environment.container_environment import ContainerShipEnv

# Laden der Q-Tabelle und Setup der Umgebung
Q = np.load("q_table.npy")
print("Q-Tabelle geladen.")

# Umgebung initialisieren
env = ContainerShipEnv() if ENV_MODE == "container" else GridEnvironment(mode=ENV_MODE)

results = defaultdict(int)
rewards_all = []

for ep in range(EPISODES):
    obs, _ = env.reset()
    state = obs[0] * env.grid_size + obs[1] if ENV_MODE == "container" else obs
    episode_reward = 0
    visited_states = {}
    steps = 0
    success = False
    cause = "Timeout"

    for _ in range(MAX_STEPS):
        action = np.argmax(Q[state])
        obs, reward, terminated, _, _ = env.step(action)
        state = obs[0] * env.grid_size + obs[1] if ENV_MODE == "container" else obs
        episode_reward += reward
        steps += 1

        visited_states[state] = visited_states.get(state, 0) + 1
        if visited_states[state] >= LOOP_THRESHOLD:
            cause = "Schleifenabbruch"
            episode_reward += LOOP_PENALTY
            break

        if terminated:
            if reward == REWARDS["goal"] or reward == REWARDS["dropoff"]:
                cause = "Ziel erreicht"
                success = True
            elif reward == REWARDS["obstacle"]:
                cause = "Hindernis-Kollision"
            break

    if not success and steps >= MAX_STEPS:
        cause = "Timeout"

    results[cause] += 1
    rewards_all.append(episode_reward)

avg_reward = np.mean(rewards_all)

print(f"\nAuswertung ({EPISODES} Episoden, Modus: {ENV_MODE}):")
for k, v in results.items():
    print(f"{v}/{EPISODES} Episoden ({(v / EPISODES) * 100:.1f}%): {k}")
print(f"\nDurchschnittlicher Reward: {avg_reward:.2f}")

# Balkendiagramm mit Text
plt.figure(figsize=(8, 5))
bars = plt.bar(results.keys(), results.values(), color='steelblue')
plt.title(f"Ausgang der Episoden ({ENV_MODE}-Modus)")
plt.xlabel(f"Episoden-Endtyp (insg. {EPISODES} Episoden)")
plt.ylabel("Anzahl")
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.5, f'{int(height)}', ha='center', va='bottom')
plt.tight_layout()
plt.show()

# Reward-Histogramm
plt.figure(figsize=(8, 5))
plt.hist(rewards_all, bins=20, edgecolor='black')
plt.title(f"Verteilung der Episoden-Rewards ({ENV_MODE}-Modus)")
plt.xlabel("Reward")
plt.ylabel("Häufigkeit")
plt.tight_layout()
plt.show()
