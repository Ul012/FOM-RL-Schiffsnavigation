# container_environment.py

# ============================================================================
# Imports
# ============================================================================

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


# ============================================================================
# ContainerShipEnv Klasse
# ============================================================================

class ContainerShipEnv(gym.Env):
    """Container-Schiff Umgebung für Q-Learning mit Pickup/Dropoff Aufgaben"""
    metadata = {"render_modes": ["human"]}

    def __init__(self, seed=None):
        super(ContainerShipEnv, self).__init__()
        self.grid_size = 5
        self.start_pos = (0, 0)
        self.obstacles = [(1, 3), (1, 2), (3, 1)]  # feste Hindernisse
        self.max_steps = 300

        self.observation_space = spaces.MultiDiscrete([self.grid_size, self.grid_size, 2])
        self.action_space = spaces.Discrete(4)  # 0=oben, 1=rechts, 2=unten, 3=links

        # Seed setzen
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Umgebung initialisieren
        self._initialize_environment()

    def _initialize_environment(self):
        """Umgebung mit zufälligen Pickup/Dropoff Positionen initialisieren"""
        self._set_random_positions()
        self.agent_pos = self.start_pos
        self.container_loaded = False
        self.steps = 0
        self.visited_states = {}
        self.max_loop_count = 3
        self.successful_dropoffs = 0

        print(f"Container-Umgebung initialisiert: Start={self.start_pos}, "
              f"Pickup={self.pickup_pos}, Dropoff={self.dropoff_pos}")

    # ============================================================================
    # Hilfsfunktionen
    # ============================================================================

    def _set_random_positions(self):
        """Zufällige Pickup und Dropoff Positionen setzen"""
        positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        positions.remove(self.start_pos)

        for obstacle in self.obstacles:
            if obstacle in positions:
                positions.remove(obstacle)

        self.pickup_pos = random.choice(positions)
        positions.remove(self.pickup_pos)
        self.dropoff_pos = random.choice(positions)

    def _get_obs(self):
        """Beobachtung erstellen"""
        return (self.agent_pos[0], self.agent_pos[1], int(self.container_loaded))

    def pos_to_state(self, pos):
        """Position zu State-Index konvertieren"""
        return pos[0] * self.grid_size + pos[1]

    def get_next_position(self, current_pos, action):
        """Neue Position nach Aktion berechnen"""
        x, y = current_pos

        if action == 0 and x > 0:  # oben
            x -= 1
        elif action == 1 and y < self.grid_size - 1:  # rechts
            y += 1
        elif action == 2 and x < self.grid_size - 1:  # unten
            x += 1
        elif action == 3 and y > 0:  # links
            y -= 1

        return (x, y)

    def calculate_reward(self, terminated_reason=None):
        """Reward basierend auf Zustand und Aktion berechnen"""
        if terminated_reason == "dropoff":
            return REWARDS["dropoff"]
        elif terminated_reason == "pickup":
            return REWARDS["pickup"]
        elif terminated_reason == "obstacle":
            return REWARDS["obstacle"]
        elif terminated_reason == "loop":
            return REWARDS["loop_abort"]
        elif terminated_reason == "timeout":
            return REWARDS["timeout"]
        else:
            return REWARDS["step"]

    def check_termination_and_rewards(self, next_pos, state_key):
        """Terminierungsbedingungen und Rewards prüfen"""
        terminated = False
        reason = None

        # Schleifenerkennung
        if self.visited_states.get(state_key, 0) >= self.max_loop_count:
            terminated = True
            reason = "loop"

        # Timeout
        elif self.steps >= self.max_steps:
            terminated = True
            reason = "timeout"

        # Hindernis getroffen
        elif next_pos in self.obstacles:
            terminated = True
            reason = "obstacle"

        # Container-spezifische Aktionen
        elif not self.container_loaded and next_pos == self.pickup_pos:
            self.container_loaded = True
            reason = "pickup"

        elif self.container_loaded and next_pos == self.dropoff_pos:
            terminated = True
            reason = "dropoff"
            self.successful_dropoffs += 1

        return terminated, reason

    def update_visited_states(self, state_key):
        """Besuchte Zustände für Schleifenerkennung aktualisieren"""
        if state_key in self.visited_states:
            self.visited_states[state_key] += 1
        else:
            self.visited_states[state_key] = 1

    # ============================================================================
    # Hauptfunktionen
    # ============================================================================

    def reset(self, seed=None, options=None):
        """Umgebung zurücksetzen"""
        super().reset(seed=seed)

        # Umgebung neu initialisieren
        self._initialize_environment()

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        """Einen Schritt in der Umgebung ausführen"""
        # Neue Position berechnen
        next_pos = self.get_next_position(self.agent_pos, action)

        # Zustand aktualisieren
        self.agent_pos = next_pos
        self.steps += 1

        # Beobachtung und State-Key erstellen
        obs = self._get_obs()
        state_key = (obs[0], obs[1], obs[2])

        # Besuchte Zustände aktualisieren
        self.update_visited_states(state_key)

        # Terminierung und Rewards prüfen
        terminated, reason = self.check_termination_and_rewards(next_pos, state_key)

        # Reward berechnen
        reward = self.calculate_reward(reason)

        return obs, reward, terminated, False, {}