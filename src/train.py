# train.py

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from config import ENV_MODE, EPISODES, ALPHA, GAMMA, EPSILON

# Projektstruktur für Imports anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Umgebung je nach Modus laden
if ENV_MODE == "container":
    from navigation.environment.container_environment import ContainerShipEnv as Env
else:
    from navigation.environment.grid_environment import GridEnvironment as Env

# Umgebung initialisieren
env = Env(mode=ENV_MODE) if ENV_MODE != "container" else Env()
grid_size = env.grid_size
n_states = env.observation_space.n if hasattr(env.observation_space, 'n') else grid_size * grid_size
n_actions = env.action_space.n

# Q-Tabelle initialisieren
Q = np.zeros((n_states, n_actions))
rewards_per_episode = []
success_per_episode = []

# Training starten
for ep in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False
    success = 0

    while not done:
        if np.random.rand() < EPSILON:
            action = np.random.choice(n_actions)
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        Q[state, action] += ALPHA * (reward + GAMMA * np.max(Q[next_state]) - Q[state, action])
        state = next_state
        total_reward += reward

        if terminated and reward in [10, 20]:  # Je nach Umgebung
            success = 1

    rewards_per_episode.append(total_reward)
    success_per_episode.append(success)

# Lernkurve mit Moving Average
window_size = 20
plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode, alpha=0.3, label="Raw Reward", color='blue')
if len(rewards_per_episode) >= window_size:
    moving_avg = np.convolve(rewards_per_episode, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size - 1, EPISODES), moving_avg, label=f"Moving Average ({window_size})", color='red')
plt.xlabel("Episode")
plt.ylabel("Gesamtreward")
plt.title(f"Lernkurve ({ENV_MODE}-Modus)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Erfolgskurve
plt.figure(figsize=(10, 3))
plt.plot(success_per_episode, label="Ziel erreicht", color='green', alpha=0.5)
plt.xlabel("Episode")
plt.ylabel("Erfolg (0/1)")
plt.title(f"Zielerreichung pro Episode ({ENV_MODE}-Modus)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Q-Tabelle speichern
np.save("q_table.npy", Q)
print("Q-Tabelle gespeichert als q_table.npy")

# Erfolgsstatistik
total_successes = sum(success_per_episode)
print(f"Erfolgreiche Zielerreichung in {total_successes}/{EPISODES} Episoden ({(total_successes/EPISODES)*100:.1f}%)")