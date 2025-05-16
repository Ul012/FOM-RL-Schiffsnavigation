import pygame
import numpy as np
import sys
import os
import time

# Pfad anpassen für den Import aus navigation.environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from navigation.environment.grid_environment import GridEnvironment

# Modusauswahl: "static" oder "random"
ENV_MODE = "static"

# Q-Tabelle laden
Q = np.load("q_table.npy")

# Umgebung initialisieren
env = GridEnvironment(mode=ENV_MODE)

# Visualisierungsparameter
CELL_SIZE = 80
GRID_SIZE = env.grid_size
WIDTH = HEIGHT = CELL_SIZE * GRID_SIZE

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AGENTEN-PFAD NACH GELERNTER POLICY")
font = pygame.font.SysFont("Segoe UI Emoji", 40)

actions_map = {0: '↑', 1: '→', 2: '↓', 3: '←'}
color_map = {
    'goal': (0, 200, 0),        # grün
    'hazard': (200, 0, 0),      # rot
    'agent': (30, 144, 255),    # blau
    'start': (255, 140, 0)      # orange
}

def draw_grid(agent_pos):
    screen.fill((224, 247, 255))  # #E0F7FF maritimer Hintergrund
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (200, 200, 200), rect, 1)

            pos = (i, j)
            if pos == agent_pos:
                txt = font.render("🚢", True, color_map['agent'])
                screen.blit(txt, (j * CELL_SIZE + 25, i * CELL_SIZE + 20))
            elif pos == env.start_pos:
                txt = font.render("🧭", True, color_map['start'])
                screen.blit(txt, (j * CELL_SIZE + 25, i * CELL_SIZE + 20))
            elif pos == env.goal_pos:
                txt = font.render("🏁", True, color_map['goal'])
                screen.blit(txt, (j * CELL_SIZE + 25, i * CELL_SIZE + 20))
            elif pos in env.hazards:
                txt = font.render("🪨", True, color_map['hazard'])
                screen.blit(txt, (j * CELL_SIZE + 25, i * CELL_SIZE + 20))
            else:
                state = env.pos_to_state(pos)
                best_action = np.argmax(Q[state])
                txt = font.render(actions_map[best_action], True, (0, 0, 0))
                screen.blit(txt, (j * CELL_SIZE + 25, i * CELL_SIZE + 20))

    pygame.display.flip()

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
        time.sleep(0.4)

    pygame.quit()

run_agent()
