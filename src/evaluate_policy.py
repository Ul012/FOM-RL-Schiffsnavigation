# evaluate_policy.py
# Ziel: Wie oft schafft der Agent es zum Ziel? → Test über z. B. 100 zufällige Karten.
import numpy as np
from navigation.environment.grid_environment import GridEnvironment

Q = np.load("q_table.npy")
num_test_envs = 100
successes = 0
total_rewards = []

for i in range(num_test_envs):
    env = GridEnvironment(mode="random")
    state, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = np.argmax(Q[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        state = next_state

    total_rewards.append(episode_reward)
    if reward == 10:
        successes += 1

print(f"Erfolgreiche Zielerreichung: {successes}/{num_test_envs} ({(successes/num_test_envs)*100:.1f}%)")
print(f"Durchschnittlicher Reward: {np.mean(total_rewards):.2f}")
