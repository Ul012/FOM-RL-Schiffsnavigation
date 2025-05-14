import pygame
import numpy as np
import sys
import os

# Pfad anpassen für den Import aus navigation.environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from navigation.environment.grid_environment import GridEnvironment

# Q-Tabelle laden
Q = np.load("q_table.npy")

# Umgebung initialisieren
env = GridEnvironment()

# Visualisierungsparameter
CELL_SIZE = 80
GRID_SIZE = env.grid_size
WIDTH = HEIGHT = CELL_SIZE * GRID_SIZE

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("GELERNTE POLICY")
font = pygame.font.SysFont("Segoe UI Symbol", 40)

actions_map = {0: '↑', 1: '→', 2: '↓', 3: '←'}

def draw_policy():
    screen.fill((255, 255, 255))  # Hintergrund weiß

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (200, 200, 200), rect, 1)  # Zellenrahmen

            pos = (i, j)
            if pos == env.goal_pos:
                txt = font.render("✪", True, (0, 128, 0))
            elif pos in env.hazards:
                txt = font.render("✖", True, (200, 0, 0))
            else:
                state = env.pos_to_state(pos)
                best_action = np.argmax(Q[state])
                symbol = actions_map[best_action]
                txt = font.render(symbol, True, (0, 0, 0))

            screen.blit(txt, (j * CELL_SIZE + 30, i * CELL_SIZE + 20))

    pygame.display.flip()

# Hauptloop
running = True
draw_policy()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
