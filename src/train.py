import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from navigation.environment.grid_environment import GridEnvironment
import numpy as np
import matplotlib.pyplot as plt

# Q-Learning Parameter
alpha = 0.1          # Lernrate
gamma = 0.9          # Diskontfaktor
epsilon = 0.1        # Epsilon-Greedy (Exploration)
episodes = 500       # Anzahl Trainingsdurchläufe

# Umgebung initialisieren
env = GridEnvironment()
n_states = env.observation_space.n
n_actions = env.action_space.n

print(f"Umgebung geladen: {n_states} Zustände, {n_actions} Aktionen.")

# Q-Tabelle initialisieren (Zustände × Aktionen)
Q = np.zeros((n_states, n_actions))
rewards_per_episode = []

# Trainingsschleife
for ep in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Epsilon-Greedy: zufällig handeln oder beste bekannte Aktion wählen
        if np.random.rand() < epsilon:
            action = np.random.choice(n_actions)  # Exploration
        else:
            action = np.argmax(Q[state])          # Exploitation

        # Aktion ausführen
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-Wert-Update
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state
        total_reward += reward

    rewards_per_episode.append(total_reward)

# Lernkurve plotten
plt.plot(rewards_per_episode)
plt.xlabel("Episode")
plt.ylabel("Gesamtreward")
plt.title("Lernkurve des Q-Learning-Agents")
plt.grid(True)
plt.show()

np.save("q_table.npy", Q)
print("Q-Tabelle gespeichert als q_table.npy")
