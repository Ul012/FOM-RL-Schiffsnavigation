import pygame
import numpy as np
import sys
import os
import time

# Projektstruktur anpassen, damit GridEnvironment importierbar ist
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from navigation.environment.grid_environment import GridEnvironment

# Modusumschalter: "static" oder "random"
ENV_MODE = "random"

# Q-Tabelle laden (muss zu Umgebung passen!)
Q = np.load("q_table.npy")

# Umgebung vorbereiten
env = GridEnvironment(mode=ENV_MODE)

# Pygame-Setup
CELL_SIZE = 80
GRID_SIZE = env.grid_size
WIDTH = HEIGHT = CELL_SIZE * GRID_SIZE

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("🚢 Schiffsnavigation – Policylauf")
font = pygame.font.SysFont("Segoe UI Emoji", 40)

actions_map = {0: '↑', 1: '→', 2: '↓', 3: '←'}
color_map = {
    'goal': (255, 255, 255),   # kariertes Ziel als weißes Feld
    'hazard': (139, 0, 0),     # dunkles Rot für gefährliche Felsen
    'agent': (0, 0, 128),      # Marineblau
    'start': (255, 140, 0)     # orange
}

def draw_grid(agent_pos):
    screen.fill((173, 216, 230))  # heller Blauton (Wasserhintergrund)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (200, 200, 200), rect, 1)

            pos = (i, j)
            if pos == agent_pos:
                txt = font.render("🚢", True, color_map['agent'])
            elif pos == env.start_pos:
                txt = font.render("S", True, color_map['start'])
            elif pos == env.goal_pos:
                txt = font.render("🏁", True, color_map['goal'])  # Ziel: Zielflagge
            elif pos in env.hazards:
                txt = font.render("🪨", True, color_map['hazard'])  # Hindernis: Felsen
            else:
                state = env.pos_to_state(pos)
                best_action = np.argmax(Q[state])
                txt = font.render(actions_map[best_action], True, (0, 0, 0))

            screen.blit(txt, (j * CELL_SIZE + 25, i * CELL_SIZE + 20))
    pygame.display.flip()

# Agent vom Start zum Ziel bewegen
def run_agent():
    pos = env.start_pos
    draw_grid(pos)
    time.sleep(0.5)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if pos == env.goal_pos or pos in env.hazards:
            time.sleep(1.5)
            running = False
            continue

        state = env.pos_to_state(pos)
        action = np.argmax(Q[state])

        # Bewegung berechnen
        row, col = pos
        if action == 0 and row > 0:
            row -= 1
        elif action == 1 and col < GRID_SIZE - 1:
            col += 1
        elif action == 2 and row < GRID_SIZE - 1:
            row += 1
        elif action == 3 and col > 0:
            col -= 1

        pos = (row, col)
        draw_grid(pos)
        time.sleep(0.5)

    pygame.quit()

# Start
run_agent()