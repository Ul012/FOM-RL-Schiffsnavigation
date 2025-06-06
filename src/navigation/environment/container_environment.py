# container_environment.py

import sys
import os

# Projektstruktur für Imports anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Drittanbieter
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

# Lokale Module
from src.config import REWARDS

class ContainerShipEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, seed=None):
        super(ContainerShipEnv, self).__init__()
        self.grid_size = 5
        self.start_pos = (0, 0)
        self.obstacles = [(1, 3), (1, 2), (3, 1)]  # feste Hindernisse
        self.max_steps = 300  # erhöhte maximale Schrittzahl

        self.observation_space = spaces.MultiDiscrete([self.grid_size, self.grid_size, 2])
        self.action_space = spaces.Discrete(4)  # 0=oben, 1=rechts, 2=unten, 3=links

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._set_random_positions()
        self.agent_pos = self.start_pos
        self.container_loaded = False
        self.steps = 0
        self.visited_states = {}
        self.max_loop_count = 3
        self.successful_dropoffs = 0

    def _set_random_positions(self):
        positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        positions.remove(self.start_pos)
        for obstacle in self.obstacles:
            if obstacle in positions:
                positions.remove(obstacle)
        self.pickup_pos = random.choice(positions)
        positions.remove(self.pickup_pos)
        self.dropoff_pos = random.choice(positions)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._set_random_positions()
        self.agent_pos = self.start_pos
        self.container_loaded = False
        self.steps = 0
        self.visited_states = {}
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        return (self.agent_pos[0], self.agent_pos[1], int(self.container_loaded))

    def step(self, action):
        x, y = self.agent_pos
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and y < self.grid_size - 1:
            y += 1
        elif action == 2 and x < self.grid_size - 1:
            x += 1
        elif action == 3 and y > 0:
            y -= 1

        self.agent_pos = (x, y)
        self.steps += 1

        reward = REWARDS["step"]  # Schrittstrafe reduziert
        terminated = False

        obs = self._get_obs()
        state_key = (obs[0], obs[1], obs[2])
        if state_key in self.visited_states:
            self.visited_states[state_key] += 1
        else:
            self.visited_states[state_key] = 1

        # Schleifenbestrafung
        if self.visited_states[state_key] >= self.max_loop_count:
            reward = REWARDS["loop_abort"]
            terminated = True

        # Timeout-Strafe
        if self.steps >= self.max_steps:
            reward = REWARDS["timeout"]
            terminated = True

        if self.agent_pos in self.obstacles:
            reward = REWARDS["obstacle"]
            terminated = True
        elif not self.container_loaded and self.agent_pos == self.pickup_pos:
            self.container_loaded = True
            reward = REWARDS["pickup"]
        elif self.container_loaded and self.agent_pos == self.dropoff_pos:
            reward = REWARDS["dropoff"]
            terminated = True
            self.successful_dropoffs += 1

        return obs, reward, terminated, False, {}

    def pos_to_state(self, pos):
        return pos[0] * self.grid_size + pos[1]
