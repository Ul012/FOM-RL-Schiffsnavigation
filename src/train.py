import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from navigation.environment.grid_environment import GridEnvironment
import numpy as np
import matplotlib.pyplot as plt

# Q-Learning Parameter
alpha = 0.1
gamma = 0.9
epsilon = 0.1
episodes = 500

# Umgebungskonfiguration
mode = "static"
grid_size = 5
n_states = grid_size * grid_size
n_actions = 4

# Q-Tabelle und Logging-Listen
Q = np.zeros((n_states, n_actions))
rewards_per_episode = []
success_per_episode = []

# Training
for ep in range(episodes):
    env = GridEnvironment(mode=mode)
    state, _ = env.reset()
    total_reward = 0
    done = False
    success = 0

    while not done:
        if np.random.rand() < epsilon:
            action = np.random.choice(n_actions)
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-Wert aktualisieren
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        state = next_state
        total_reward += reward

        # Erfolg erfassen (Ziel erreicht?)
        if terminated and reward == 10:
            success = 1

    rewards_per_episode.append(total_reward)
    success_per_episode.append(success)

# Lernkurve mit Moving Average
window_size = 20
plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode, alpha=0.3, label="Raw Reward", color='blue')

if len(rewards_per_episode) >= window_size:
    moving_avg = np.convolve(rewards_per_episode, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size - 1, episodes), moving_avg, label=f"Moving Average ({window_size})", color='red')

plt.xlabel("Episode")
plt.ylabel("Gesamtreward")
plt.title("Lernkurve in dynamischen Umgebungen")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Erfolgskurve (0 = gescheitert, 1 = Ziel erreicht)
plt.figure(figsize=(10, 3))
plt.plot(success_per_episode, label="Ziel erreicht", color='green', alpha=0.5)
plt.xlabel("Episode")
plt.ylabel("Erfolg (0/1)")
plt.title("Zielerreichung pro Episode")
plt.grid(True)
plt.tight_layout()
plt.show()

# Q-Tabelle speichern
np.save("q_table.npy", Q)
print("Q-Tabelle gespeichert als q_table.npy")

# Erfolgsstatistik
total_successes = sum(success_per_episode)
print(f"Erfolgreiche Zielerreichung in {total_successes}/{episodes} Episoden ({(total_successes/episodes)*100:.1f}%)")