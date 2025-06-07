# grid_environment.py

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
# GridEnvironment Klasse
# ============================================================================

class GridEnvironment(gym.Env):
    """Grid-basierte Umgebung für Q-Learning"""
    metadata = {"render_modes": ["human"]}

    def __init__(self, mode="static", seed=None):
        super(GridEnvironment, self).__init__()
        self.mode = mode
        self.grid_size = 5
        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)
        self.action_space = spaces.Discrete(4)  # 0=oben, 1=rechts, 2=unten, 3=links

        # Attribute für Schleifenerkennung und Timeout
        self.visited_states = {}
        self.max_steps = 50
        self.current_steps = 0
        self.loop_threshold = 6

        # Seed setzen
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Umgebung initialisieren
        self._initialize_environment()

    def _initialize_environment(self):
        """Umgebung mit Positionen initialisieren"""
        all_positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]

        # Standardwerte
        self.start_pos = (0, 0)
        self.goal_pos = (4, 4)
        self.obstacles = [(1, 1), (2, 3), (3, 1)]

        # Modus-spezifische Anpassungen
        if self.mode == "random_start":
            possible_starts = all_positions.copy()
            possible_starts.remove(self.goal_pos)
            for obstacle in self.obstacles:
                if obstacle in possible_starts:
                    possible_starts.remove(obstacle)
            self.start_pos = random.choice(possible_starts)

        elif self.mode == "random_goal":
            possible_goals = all_positions.copy()
            possible_goals.remove(self.start_pos)
            for obstacle in self.obstacles:
                if obstacle in possible_goals:
                    possible_goals.remove(obstacle)
            self.goal_pos = random.choice(possible_goals)

        elif self.mode == "random_obstacles":
            all_positions.remove(self.start_pos)
            all_positions.remove(self.goal_pos)
            self.obstacles = random.sample(all_positions, k=3)

        # Agent-Position initialisieren
        self.agent_pos = self.start_pos
        self.state = self.pos_to_state(self.start_pos)

        print(f"Umgebung initialisiert: Start={self.start_pos}, Ziel={self.goal_pos}, Hindernisse={self.obstacles}")


# ============================================================================
# Hilfsfunktionen
# ============================================================================

    def pos_to_state(self, pos):
        """Position zu State-Index konvertieren"""
        return pos[0] * self.grid_size + pos[1]

    def state_to_pos(self, state):
        """State-Index zu Position konvertieren"""
        return (state // self.grid_size, state % self.grid_size)

    def get_next_position(self, current_pos, action):
        """Neue Position nach Aktion berechnen"""
        row, col = current_pos
        new_row, new_col = row, col

        if action == 0 and row > 0:  # oben
            new_row = row - 1
        elif action == 1 and col < self.grid_size - 1:  # rechts
            new_col = col + 1
        elif action == 2 and row < self.grid_size - 1:  # unten
            new_row = row + 1
        elif action == 3 and col > 0:  # links
            new_col = col - 1

        return (new_row, new_col)

    def calculate_reward(self, next_pos, terminated_reason=None):
        """Reward basierend auf Position und Zustand berechnen"""
        reward = REWARDS["step"]  # Grundstrafe für jeden Schritt

        if terminated_reason == "goal":
            reward = REWARDS["goal"]  # Überschreibt Schrittstrafe
        elif terminated_reason == "obstacle":
            reward += REWARDS["obstacle"]  # Zusätzliche Strafe
        elif terminated_reason == "loop":
            reward += REWARDS["loop_abort"]  # Zusätzliche Strafe
        elif terminated_reason == "timeout":
            reward += REWARDS["timeout"]  # Zusätzliche Strafe

        return reward

    def check_termination(self, next_pos, next_state):
        """Terminierungsbedingungen prüfen"""
        terminated = False
        reason = None

        # Ziel erreicht (höchste Priorität)
        if next_pos == self.goal_pos:
            terminated = True
            reason = "goal"
            print(f"DEBUG: ZIEL ERREICHT! Pos={next_pos}, Ziel={self.goal_pos}")

        # Hindernis getroffen
        elif next_pos in self.obstacles:
            terminated = True
            reason = "obstacle"
            print(f"DEBUG: Hindernis getroffen! Pos={next_pos}")

        # Schleifenerkennung
        elif self.visited_states.get(next_state, 0) >= self.loop_threshold:
            terminated = True
            reason = "loop"
            print(f"DEBUG: Schleifenabbruch! Pos={next_pos} besucht {self.visited_states[next_state]}x")

        # Timeout
        elif self.current_steps >= self.max_steps:
            terminated = True
            reason = "timeout"
            print(f"DEBUG: Timeout nach {self.current_steps} Schritten!")

        return terminated, reason


# ============================================================================
# Hauptfunktionen
# ============================================================================

    def reset(self, seed=None, options=None):
        """Umgebung zurücksetzen"""
        super().reset(seed=seed)

        # Umgebung neu initialisieren
        self._initialize_environment()

        # Tracking-Attribute zurücksetzen
        self.visited_states = {}
        self.current_steps = 0

        # Warnung bei Start == Ziel
        if self.agent_pos == self.goal_pos:
            print(f"WARNING: Agent startet bereits am Ziel! Start={self.agent_pos}, Ziel={self.goal_pos}")

        return self.state, {}

    def step(self, action):
        """Einen Schritt in der Umgebung ausführen"""
        current_pos = self.state_to_pos(self.state)
        print(f"DEBUG: Vor Bewegung - Pos={current_pos}, Aktion={action}")

        # Neue Position berechnen
        next_pos = self.get_next_position(current_pos, action)
        next_state = self.pos_to_state(next_pos)

        # Zustand aktualisieren
        self.state = next_state
        self.agent_pos = next_pos
        self.current_steps += 1

        # Schleifenerkennung aktualisieren
        if next_state in self.visited_states:
            self.visited_states[next_state] += 1
        else:
            self.visited_states[next_state] = 1

        print(f"DEBUG: Nach Bewegung - Neue Pos={next_pos}")

        # Terminierung prüfen
        terminated, reason = self.check_termination(next_pos, next_state)

        # Reward berechnen
        reward = self.calculate_reward(next_pos, reason)

        print(f"DEBUG: Final - Reward={reward}, Terminiert={terminated}, Schritte={self.current_steps}")

        return next_state, reward, terminated, False, {}