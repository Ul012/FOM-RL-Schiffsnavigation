import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridEnvironment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(GridEnvironment, self).__init__()
        self.grid_size = 5
        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)
        self.action_space = spaces.Discrete(4)  # 0=oben, 1=rechts, 2=unten, 3=links
        self.start_pos = (0, 0)
        self.goal_pos = (4, 4)
        self.hazards = [(1, 1), (2, 3), (3, 1)]
        self.state = self.pos_to_state(self.start_pos)

    def pos_to_state(self, pos):
        return pos[0] * self.grid_size + pos[1]
