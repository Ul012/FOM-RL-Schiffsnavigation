# grid_environment.py - DEBUG VERSION

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


class GridEnvironment(gym.Env):
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

        # WICHTIG: Agent-Position korrekt initialisieren
        self.agent_pos = self.start_pos
        self.state = self.pos_to_state(self.start_pos)

        # DEBUG: Position beim Start ausgeben
        print(f"DEBUG: Start={self.start_pos}, Goal={self.goal_pos}, Obstacles={self.obstacles}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        all_positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]

        # Standardwerte zurücksetzen
        self.start_pos = (0, 0)
        self.goal_pos = (4, 4)
        self.obstacles = [(1, 1), (2, 3), (3, 1)]

        if self.mode == "random_start":
            possible_starts = [pos for pos in all_positions if pos != self.goal_pos and pos not in self.obstacles]
            self.start_pos = random.choice(possible_starts)
            # DEBUG: Neuer Start
            print(f"DEBUG: Neuer random_start={self.start_pos}")

        elif self.mode == "random_goal":
            possible_goals = [pos for pos in all_positions if pos != self.start_pos and pos not in self.obstacles]
            self.goal_pos = random.choice(possible_goals)

        elif self.mode == "random_obstacles":
            possible_obstacles = [pos for pos in all_positions if pos != self.start_pos and pos != self.goal_pos]
            self.obstacles = random.sample(possible_obstacles, k=3)

        # WICHTIG: Agent-Position korrekt setzen
        self.agent_pos = self.start_pos
        self.state = self.pos_to_state(self.start_pos)

        # Tracking-Attribute für neue Episode zurücksetzen
        self.visited_states = {}
        self.current_steps = 0

        # DEBUG: Check ob Start == Goal
        if self.agent_pos == self.goal_pos:
            print(f"WARNING: Agent startet bereits am Ziel! Start={self.agent_pos}, Goal={self.goal_pos}")

        return self.state, {}

    def pos_to_state(self, pos):
        return pos[0] * self.grid_size + pos[1]

    def state_to_pos(self, state):
        """Hilfsfunktion: State zurück zu Position"""
        return (state // self.grid_size, state % self.grid_size)

    def step(self, action):
        # WICHTIG: Position aus State ableiten
        current_pos = self.state_to_pos(self.state)
        row, col = current_pos

        # DEBUG: Aktuelle Position vor Bewegung
        print(f"DEBUG: Vor Bewegung - Pos=({row},{col}), Action={action}")

        # Bewegung nach Aktion
        new_row, new_col = row, col
        if action == 0 and row > 0:  # oben
            new_row = row - 1
        elif action == 1 and col < self.grid_size - 1:  # rechts
            new_col = col + 1
        elif action == 2 and row < self.grid_size - 1:  # unten
            new_row = row + 1
        elif action == 3 and col > 0:  # links
            new_col = col - 1

        next_pos = (new_row, new_col)
        next_state = self.pos_to_state(next_pos)
        self.state = next_state
        self.agent_pos = next_pos  # Agent-Position aktualisieren

        # Schrittzähler erhöhen
        self.current_steps += 1

        # IMMER erst Schrittstrafe
        reward = REWARDS["step"]  # -1
        terminated = False

        # DEBUG: Nach Bewegung
        print(f"DEBUG: Nach Bewegung - Neue Pos={next_pos}, Reward bisher={reward}")

        # Schleifenerkennung - Zustand-Besuchszähler aktualisieren
        if next_state in self.visited_states:
            self.visited_states[next_state] += 1
        else:
            self.visited_states[next_state] = 1

        # Schleifenabbruch-Check
        if self.visited_states[next_state] >= self.loop_threshold:
            reward += REWARDS["loop_abort"]  # Zusätzliche Strafe
            terminated = True
            print(f"DEBUG: Schleifenabbruch! Pos={next_pos} besucht {self.visited_states[next_state]}x")

        # Timeout-Check
        elif self.current_steps >= self.max_steps:
            reward += REWARDS["timeout"]  # Zusätzliche Strafe
            terminated = True
            print(f"DEBUG: Timeout nach {self.current_steps} Schritten!")

        # Ziel-Check (höchste Priorität)
        elif next_pos == self.goal_pos:
            reward = REWARDS["goal"]  # Überschreibt Schrittstrafe
            terminated = True
            print(f"DEBUG: ZIEL ERREICHT! Pos={next_pos}, Goal={self.goal_pos}, Final Reward={reward}")

        # Hindernis-Check
        elif next_pos in self.obstacles:
            reward += REWARDS["obstacle"]  # Zusätzliche Strafe
            terminated = True
            print(f"DEBUG: Hindernis getroffen! Pos={next_pos}, Obstacles={self.obstacles}")

        print(f"DEBUG: Final - Reward={reward}, Terminated={terminated}, Steps={self.current_steps}")
        return next_state, reward, terminated, False, {}