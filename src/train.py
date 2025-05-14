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

Q = np.zeros((n_states, n_actions))
rewards_per_episode = []

for ep in range(episodes):
    env = GridEnvironment(mode=mode)
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        if np.random.rand() < epsilon:
            action = np.random.choice(n_actions)
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        state = next_state
        total_reward += reward

    rewards_per_episode.append(total_reward)

# Lernkurve anzeigen
plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Gesamtreward")
plt.title("Lernkurve mit dynamischen Umgebungen")
plt.grid(True)
plt.show()

np.save("q_table.npy", Q)
print("Q-Tabelle gespeichert als q_table.npy")
