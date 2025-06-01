# evaluate_policy.py
# Ziel: Wie oft schafft der Agent es zum Ziel? → Test über z. B. 100 zufällige Karten.

import numpy as np
from config import ENV_MODE
from navigation.environment.grid_environment import GridEnvironment
import os
from collections import deque

# Q-Tabelle laden (optional nach ENV_MODE benannt)
q_path = f"q_table_{ENV_MODE}.npy"
if os.path.exists(q_path):
    Q = np.load(q_path)
    print(f"Q-Tabelle geladen: {q_path}")
else:
    Q = np.load("q_table.npy")
    print("Q-Tabelle geladen: q_table.npy")

num_test_envs = 100
max_steps_per_episode = 50  # Abbruchbedingung nach x Schritten

successes = 0
total_rewards = []

for i in range(num_test_envs):
    env = GridEnvironment(mode=ENV_MODE)
    state, _ = env.reset()
    done = False
    episode_reward = 0
    steps = 0
    position_history = deque(maxlen=10)  # Zur Erkennung von Loops

    while not done and steps < max_steps_per_episode:
        action = np.argmax(Q[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        state = next_state
        steps += 1

        # Position verfolgen
        pos = divmod(state, env.grid_size)
        position_history.append(pos)

        # Schleifenverhalten erkennen und bestrafen
        if list(position_history).count(pos) > 6:
            # Strafbewertung beim Loop-Abbruch, rein für die Auswertung,
            # nicht fürs Lernen (weil evaluate_policy.py ja nicht trainiert)
            reward -= 15
            episode_reward += reward
            done = True
            print(f"Episode {i+1}: Abbruch wegen erkennbarer Schleife.")

    total_rewards.append(episode_reward)
    if reward == 10:
        successes += 1

print(f"Modus: {ENV_MODE}")
print(f"Erfolgreiche Zielerreichung: {successes}/{num_test_envs} ({(successes/num_test_envs)*100:.1f}%)")
print(f"Durchschnittlicher Reward: {np.mean(total_rewards):.2f}")
