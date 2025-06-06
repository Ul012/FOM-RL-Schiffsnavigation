# train.py

import sys
import os

# Projektstruktur für Imports anpassen (muss VOR Projektimporten stehen)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Drittanbieter
import numpy as np
import matplotlib.pyplot as plt

# Lokale Module
from config import ENV_MODE, EPISODES, MAX_STEPS, LOOP_THRESHOLD, ALPHA, GAMMA, EPSILON, ACTIONS, REWARDS
from navigation.environment.grid_environment import GridEnvironment
from navigation.environment.container_environment import ContainerShipEnv

# Umgebung initialisieren
env = ContainerShipEnv() if ENV_MODE == "container" else GridEnvironment(mode=ENV_MODE)
grid_size = env.grid_size

# Zustandscodierung je nach Umgebung
def obs_to_state(obs):
    if ENV_MODE == "container":
        return obs[0] * grid_size + obs[1] + (grid_size * grid_size) * obs[2]
    return obs

# Zustands- und Aktionsraum ermitteln
n_states = env.observation_space.n if hasattr(env.observation_space, 'n') else np.prod(env.observation_space.nvec)
n_actions = env.action_space.n

# Q-Tabelle initialisieren
Q = np.zeros((n_states, n_actions))
rewards_per_episode = []
success_per_episode = []

# Training starten
for ep in range(EPISODES):
    obs, _ = env.reset()
    state = obs_to_state(obs)
    total_reward = 0
    done = False
    success = 0

    while not done:
        if np.random.rand() < EPSILON:
            action = np.random.choice(n_actions)
        else:
            action = np.argmax(Q[state])

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = obs_to_state(obs)

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