# evaluate_policy.py
# Ziel: Wie oft schafft der Agent es zum Ziel? → Test über z.B. 100 zufällige Karten.

import numpy as np
from config import ENV_MODE
# from navigation.environment.grid_environment import GridEnvironment
if ENV_MODE == "container":
    from navigation.environment.container_environment import ContainerShipEnv as Env
else:
    from navigation.environment.grid_environment import GridEnvironment as Env
import os
from collections import deque
import matplotlib.pyplot as plt

# Konfiguration
VISUALIZE = True

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
loop_aborts = 0

for i in range(num_test_envs):
    # env = GridEnvironment(mode=ENV_MODE)
    env = Env() if ENV_MODE == "container" else Env(mode=ENV_MODE)
    state, _ = env.reset()
    if ENV_MODE == "container":
        x, y, loaded = state
        state = x * env.grid_size * 2 + y * 2 + loaded  # Zustand zu Index umwandeln
    done = False
    episode_reward = 0
    steps = 0
    position_history = deque(maxlen=10)

    while not done and steps < max_steps_per_episode:
        action = np.argmax(Q[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        if ENV_MODE == "container":
            x, y, loaded = next_state
            next_state = x * 5 * 2 + y * 2 + loaded
        done = terminated or truncated
        episode_reward += reward
        state = next_state
        steps += 1

        pos = divmod(state, env.grid_size)
        position_history.append(pos)

        if list(position_history).count(pos) > 6:
            reward -= 15
            episode_reward += reward
            done = True
            loop_aborts += 1
            print(f"Episode {i+1}: Abbruch wegen erkennbarer Schleife.")

    total_rewards.append(episode_reward)
    if reward == 10:
        successes += 1

print(f"Modus: {ENV_MODE}")
print(f"Erfolgreiche Zielerreichung: {successes}/{num_test_envs} ({(successes/num_test_envs)*100:.1f}%)")
print(f"Durchschnittlicher Reward: {np.mean(total_rewards):.2f}")
print(f"Abbrüche wegen Schleifen: {loop_aborts}")

# Optionale Visualisierung
if VISUALIZE:
    plt.figure(figsize=(6, 4))
    plt.bar(["Erfolg", "Misserfolg"], [successes, num_test_envs - successes], color=["green", "red"])
    plt.title("Zielerreichung in Testläufen")
    plt.ylabel("Anzahl")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.hist(total_rewards, bins=20, color="blue", alpha=0.7)
    plt.title("Verteilung der Gesamtrewards")
    plt.xlabel("Reward")
    plt.ylabel("Häufigkeit")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
