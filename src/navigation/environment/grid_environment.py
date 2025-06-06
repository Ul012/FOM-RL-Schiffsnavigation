# grid_environment.py

import sys
import os

# Projektstruktur für Imports anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Third-Party
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

# Lokale Module
from src.config import REWARDS

class GridEnvironment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, mode="static", seed=None):
        super(GridEnvironment, self).__init__()
        self.mode = mode
        self.grid_size = 5
        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)
        self.action_space = spaces.Discrete(4)  # 0=oben, 1=rechts, 2=unten, 3=links

        # Seed setzen, falls angegeben
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Positionen vorbereiten
        all_positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]

        # Standardwerte
        self.start_pos = (0, 0)
        self.goal_pos = (4, 4)
        self.obstacles = [(1, 1), (2, 3), (3, 1)]

        # Modus-spezifische Anpassungen
        if mode == "random_start":
            possible_starts = all_positions.copy()
            possible_starts.remove(self.goal_pos)
            for h in self.obstacles:
                if h in possible_starts:
                    possible_starts.remove(h)
            self.start_pos = random.choice(possible_starts)

        elif mode == "random_goal":
            possible_goals = all_positions.copy()
            possible_goals.remove(self.start_pos)
            for h in self.obstacles:
                if h in possible_goals:
                    possible_goals.remove(h)
            self.goal_pos = random.choice(possible_goals)

        elif mode == "random_obstacles":
            all_positions.remove(self.start_pos)
            all_positions.remove(self.goal_pos)
            self.obstacles = random.sample(all_positions, k=3)

        # State initialisieren
        self.state = self.pos_to_state(self.start_pos)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        all_positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]

        self.start_pos = (0, 0)
        self.goal_pos = (4, 4)
        self.obstacles = [(1, 1), (2, 3), (3, 1)]

        if self.mode == "random_start":
            possible_starts = [pos for pos in all_positions if pos != self.goal_pos and pos not in self.obstacles]
            self.start_pos = random.choice(possible_starts)

        elif self.mode == "random_goal":
            possible_goals = [pos for pos in all_positions if pos != self.start_pos and pos not in self.obstacles]
            self.goal_pos = random.choice(possible_goals)

        elif self.mode == "random_obstacles":
            possible_obstacles = [pos for pos in all_positions if pos != self.start_pos and pos != self.goal_pos]
            self.obstacles = random.sample(possible_obstacles, k=3)

        self.agent_position = self.start_pos
        self.state = self.pos_to_state(self.start_pos)
        return self.state, {}

    def pos_to_state(self, pos):
        return pos[0] * self.grid_size + pos[1]

    def step(self, action):
        row, col = divmod(self.state, self.grid_size)

        # Bewegung nach Aktion
        if action == 0 and row > 0:  # oben
            row -= 1
        elif action == 1 and col < self.grid_size - 1:  # rechts
            col += 1
        elif action == 2 and row < self.grid_size - 1:  # unten
            row += 1
        elif action == 3 and col > 0:  # links
            col -= 1

        next_pos = (row, col)
        next_state = self.pos_to_state(next_pos)
        self.state = next_state

        reward = REWARDS["step"]  # Bewegungskosten
        terminated = False

        if next_pos == self.goal_pos:
            reward = REWARDS["goal"]
            terminated = True
        elif next_pos in self.obstacles:
            reward = REWARDS["obstacle"]
            terminated = True

        return next_state, reward, terminated, False, {}
