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

env = GridEnvironment()
n_states = env.observation_space.n
n_actions = env.action_space.n

# Q-Tabelle initialisieren
Q = np.zeros((n_states, n_actions))
rewards_per_episode = []

print("GridEnvironment erfolgreich geladen.")
