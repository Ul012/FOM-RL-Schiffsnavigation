# visualize_policy.py

import pygame
import numpy as np
import sys
import os
import time
import imageio
from pathlib import Path
from PIL import Image

# === Konfiguration ===
from config import ENV_MODE
# ENV_MODE = "static"           # "static", "random_start", "random_goal", "random_obstacles"
EXPORT_FRAMES = False         # Optionaler Export von Bildern und GIF
EXPORT_PATH = "export"        # Speicherort für Screenshots und Animation

# Projektstruktur für Import anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from navigation.environment.grid_environment import GridEnvironment

# Q-Tabelle laden
Q = np.load("q_table.npy")

# Umgebung vorbereiten
env = GridEnvironment(mode=ENV_MODE)

# Pygame-Setup
CELL_SIZE = 80
GRID_SIZE = env.grid_size
WIDTH = HEIGHT = CELL_SIZE * GRID_SIZE

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AGENTEN-PFAD NACH GELERNTER POLICY")
font = pygame.font.SysFont("Segoe UI Emoji", 40)

actions_map = {0: '↑', 1: '→', 2: '↓', 3: '←'}
color_map = {
    'goal': (0, 200, 0),
    'hazard': (200, 0, 0),
    'agent': (30, 144, 255),
    'start': (255, 140, 0)
}

frames = []

def draw_grid(agent_pos, save_frame=False):
    screen.fill((224, 247, 255))  # #E0F7FF
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (200, 200, 200), rect, 1)

            pos = (i, j)
            if pos == agent_pos:
                txt = font.render("🚢", True, color_map['agent'])
            elif pos == env.start_pos:
                txt = font.render("🧭", True, color_map['start'])
            elif pos == env.goal_pos:
                txt = font.render("🏁", True, color_map['goal'])
            elif pos in env.hazards:
                txt = font.render("🪨", True, color_map['hazard'])
            else:
                state = env.pos_to_state(pos)
                best_action = np.argmax(Q[state])
                txt = font.render(actions_map[best_action], True, (0, 0, 0))

            screen.blit(txt, (j * CELL_SIZE + 25, i * CELL_SIZE + 20))
    pygame.display.flip()

    if save_frame:
        Path(EXPORT_PATH).mkdir(exist_ok=True)
        frame_path = f"{EXPORT_PATH}/frame_{len(frames):03d}.png"
        pygame.image.save(screen, frame_path)
        frames.append(frame_path)

def run_agent():
    pos = env.start_pos
    draw_grid(pos, save_frame=EXPORT_FRAMES)
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
        draw_grid(pos, save_frame=EXPORT_FRAMES)
        time.sleep(0.4)

    pygame.quit()

    # Export nach Beendigung
    if EXPORT_FRAMES and frames:
        print("Erstelle GIF und PDF...")
        images = [imageio.imread(f) for f in frames]
        imageio.mimsave(f"{EXPORT_PATH}/agent_run.gif", images, duration=0.5)
        print(f"GIF gespeichert unter {EXPORT_PATH}/agent_run.gif")

        # PDF aus letztem Frame
        Image.open(frames[-1]).save(f"{EXPORT_PATH}/final_frame.pdf")
        print(f"PDF-Screenshot gespeichert unter {EXPORT_PATH}/final_frame.pdf")

# Start
run_agent()
