import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class GridEnvironment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, mode="static"):
        super(GridEnvironment, self).__init__()
        self.grid_size = 5
        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)
        self.action_space = spaces.Discrete(4)  # 0=oben, 1=rechts, 2=unten, 3=links
        self.start_pos = (0, 0)

        if mode == "random":
            all_positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
            all_positions.remove(self.start_pos)
            self.goal_pos = random.choice(all_positions)
            all_positions.remove(self.goal_pos)
            self.hazards = random.sample(all_positions, k=3)
        else:
            self.goal_pos = (4, 4)
            self.hazards = [(1, 1), (2, 3), (3, 1)]

        self.state = self.pos_to_state(self.start_pos)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
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

        reward = -1  # standardmäßige Bewegungskosten
        terminated = False

        if next_pos == self.goal_pos:
            reward = 10
            terminated = True
        elif next_pos in self.hazards:
            reward = -10
            terminated = True

        return next_state, reward, terminated, False, {}


