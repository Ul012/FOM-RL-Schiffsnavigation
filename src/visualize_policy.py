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
EXPORT_FRAMES = False
EXPORT_PATH = "export"

# Projektstruktur für Import anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
if ENV_MODE == "container":
    from navigation.environment.container_environment import ContainerShipEnv as Env
else:
    from navigation.environment.grid_environment import GridEnvironment as Env

Q = np.load("q_table.npy")
print("Q-Tabelle geladen: q_table.npy")

env = Env()
GRID_SIZE = env.grid_size
CELL_SIZE = 80
WIDTH = HEIGHT = CELL_SIZE * GRID_SIZE

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(f"AGENTEN-PFAD NACH GELERNTER POLICY – Modus: {ENV_MODE}")
font = pygame.font.SysFont("Segoe UI Emoji", 40)

actions_map = {0: '↑', 1: '→', 2: '↓', 3: '←'}
color_map = {
    'goal': (0, 200, 0),
    'hazard': (200, 0, 0),
    'agent': (30, 144, 255),
    'start': (255, 140, 0)
}

frames = []

def obs_to_state(obs):
    if ENV_MODE == "container":
        return obs[0] * GRID_SIZE + obs[1]
    return obs

def draw_grid(agent_pos, save_frame=False):
    screen.fill((224, 247, 255))
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, (200, 200, 200), rect, 1)
            pos = (i, j)
            if pos == agent_pos:
                txt = font.render("🚢", True, color_map['agent'])
            elif pos == env.start_pos:
                txt = font.render("🧭", True, color_map['start'])
            elif hasattr(env, "goal_pos") and pos == env.goal_pos:
                txt = font.render("🏁", True, color_map['goal'])
            elif hasattr(env, "dropoff_pos") and pos == env.dropoff_pos:
                txt = font.render("📦", True, color_map['goal'])
            elif pos in getattr(env, "hazards", []):
                txt = font.render("🪨", True, color_map['hazard'])
            else:
                state = obs_to_state((i, j, 0) if ENV_MODE == "container" else env.pos_to_state((i, j)))
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
    obs, _ = env.reset()
    state = obs_to_state(obs)
    pos = (obs[0], obs[1]) if ENV_MODE == "container" else divmod(obs, GRID_SIZE)

    draw_grid(pos, save_frame=EXPORT_FRAMES)
    time.sleep(0.5)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        action = np.argmax(Q[state])
        obs, _, terminated, truncated, _ = env.step(action)
        state = obs_to_state(obs)
        pos = (obs[0], obs[1]) if ENV_MODE == "container" else divmod(obs, GRID_SIZE)
        draw_grid(pos, save_frame=EXPORT_FRAMES)
        time.sleep(0.4)
        if terminated or truncated:
            time.sleep(1.5)
            running = False

    pygame.quit()

    if EXPORT_FRAMES and frames:
        print("Erstelle GIF und PDF...")
        images = [imageio.imread(f) for f in frames]
        imageio.mimsave(f"{EXPORT_PATH}/agent_run.gif", images, duration=0.5)
        print(f"GIF gespeichert unter {EXPORT_PATH}/agent_run.gif")
        Image.open(frames[-1]).save(f"{EXPORT_PATH}/final_frame.pdf")
        print(f"PDF-Screenshot gespeichert unter {EXPORT_PATH}/final_frame.pdf")

run_agent()
