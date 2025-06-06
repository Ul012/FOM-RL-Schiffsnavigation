# visualize_policy.py

# ============================================================================
# Imports
# ============================================================================

import sys
import os

# Projektstruktur für Import anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Drittanbieter
import pygame
import numpy as np
import time
import imageio
from pathlib import Path
from PIL import Image

# Lokale Module
from config import (ENV_MODE, EPISODES, MAX_STEPS, LOOP_THRESHOLD, ALPHA, GAMMA, EPSILON,
                    ACTIONS, REWARDS, GRID_SIZE, NUM_TEST_ENVS, Q_TABLE_PATH,
                    EXPORT_FRAMES, EXPORT_PATH, CELL_SIZE, FRAME_DELAY)
from navigation.environment.grid_environment import GridEnvironment
from navigation.environment.container_environment import ContainerShipEnv


# ============================================================================
# Hilfsfunktionen
# ============================================================================

# Initialisierung der Umgebung
def initialize_environment():
    env = ContainerShipEnv() if ENV_MODE == "container" else GridEnvironment(mode=ENV_MODE)
    grid_size = env.grid_size
    print(f"Umgebung initialisiert: {ENV_MODE}-Modus, Grid-Größe: {grid_size}x{grid_size}")
    return env, grid_size


# Q-Tabelle laden
def load_q_table(filepath=Q_TABLE_PATH):
    try:
        Q = np.load(filepath)
        print(f"Q-Tabelle geladen: {filepath}, Shape: {Q.shape}")
        return Q
    except FileNotFoundError:
        print(f"FEHLER: Q-Tabelle nicht gefunden: {filepath}")
        print("Bitte führen Sie zuerst das Training mit train.py aus.")
        sys.exit(1)


# Zustandscodierung je nach Umgebung
def obs_to_state(obs, env_mode=ENV_MODE, grid_size=None):
    if env_mode == "container":
        return obs[0] * grid_size + obs[1] + (grid_size * grid_size) * obs[2]
    return obs


# Pygame initialisieren
def initialize_pygame(grid_size):
    pygame.init()
    width = height = CELL_SIZE * grid_size
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"AGENTEN-PFAD NACH GELERNTER POLICY – Modus: {ENV_MODE}")
    font = pygame.font.SysFont("Segoe UI Emoji", 40)
    print(f"Pygame initialisiert: {width}x{height} Pixel")
    return screen, font


# Export-Ordner erstellen
def setup_export_path():
    if EXPORT_FRAMES:
        Path(EXPORT_PATH).mkdir(exist_ok=True)
        print(f"Export-Pfad erstellt: {EXPORT_PATH}")


# Beste Aktion für einen Zustand ermitteln
def get_best_action(Q, state):
    return np.argmax(Q[state])


# Position aus Observation extrahieren
def get_position_from_obs(obs, env_mode):
    if env_mode == "container":
        return (obs[0], obs[1])
    else:
        return divmod(obs, GRID_SIZE)


# ============================================================================
# Visualisierungsfunktionen
# ============================================================================

# Grid mit Agentenpfad zeichnen
def draw_grid(screen, font, env, agent_pos, Q, grid_size, frames, save_frame=False):
    # Farbkonfiguration
    colors = {
        'background': (224, 247, 255),
        'grid_line': (200, 200, 200),
        'text': (0, 0, 0)
    }

    # Aktions-Mapping
    actions_map = {0: '↑', 1: '→', 2: '↓', 3: '←'}

    # Hintergrund füllen
    screen.fill(colors['background'])

    # Grid zeichnen
    for i in range(grid_size):
        for j in range(grid_size):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, colors['grid_line'], rect, 1)
            pos = (i, j)

            # Symbol für aktuelle Position bestimmen
            if pos == agent_pos:
                symbol = "🚢"
            elif hasattr(env, 'start_pos') and pos == env.start_pos:
                symbol = "🧭"
            elif hasattr(env, "pickup_pos") and pos == env.pickup_pos:
                symbol = "📤"
            elif hasattr(env, "dropoff_pos") and pos == env.dropoff_pos:
                symbol = "📦"
            elif hasattr(env, "goal_pos") and pos == env.goal_pos:
                symbol = "🏁"
            elif pos in getattr(env, "obstacles", []):
                symbol = "🪨"
            else:
                # Policy-Pfeil anzeigen
                state = obs_to_state((i, j, 0) if ENV_MODE == "container" else env.pos_to_state((i, j)), ENV_MODE,
                                     grid_size)
                if state < Q.shape[0]:
                    best_action = get_best_action(Q, state)
                    symbol = actions_map[best_action]
                else:
                    symbol = "?"

            # Symbol rendern und zeichnen
            text_surface = font.render(symbol, True, colors['text'])
            text_rect = text_surface.get_rect(center=(j * CELL_SIZE + CELL_SIZE // 2,
                                                      i * CELL_SIZE + CELL_SIZE // 2))
            screen.blit(text_surface, text_rect)

    pygame.display.flip()

    # Frame exportieren, falls gewünscht
    if save_frame and EXPORT_FRAMES:
        frame_path = f"{EXPORT_PATH}/frame_{len(frames):03d}.png"
        pygame.image.save(screen, frame_path)
        frames.append(frame_path)


# GIF und PDF aus Frames erstellen
def create_export_files(frames):
    if not frames:
        return

    print("Erstelle GIF und PDF...")
    try:
        # GIF erstellen
        images = [imageio.imread(f) for f in frames]
        gif_path = f"{EXPORT_PATH}/agent_run.gif"
        imageio.mimsave(gif_path, images, duration=0.5)
        print(f"GIF gespeichert unter {gif_path}")

        # PDF vom letzten Frame
        pdf_path = f"{EXPORT_PATH}/final_frame.pdf"
        Image.open(frames[-1]).save(pdf_path)
        print(f"PDF-Screenshot gespeichert unter {pdf_path}")

    except Exception as e:
        print(f"Fehler beim Erstellen der Export-Dateien: {e}")


# Debug-Informationen für Agentenschritte
def print_step_info(step, obs, action, reward, terminated, truncated):
    action_names = {0: "Oben (↑)", 1: "Rechts (→)", 2: "Unten (↓)", 3: "Links (←)"}
    print(f"Schritt {step}: Pos={obs}, Aktion={action_names.get(action, '?')}, "
          f"Reward={reward}, Beendet={terminated or truncated}")


# ============================================================================
# HAUPTFUNKTION
# ============================================================================

# Agent mit gelernter Policy ausführen
def run_agent():
    print(f"Starte Agenten-Visualisierung für {ENV_MODE}-Modus...")

    # Initialisierung
    env, grid_size = initialize_environment()
    Q = load_q_table()
    setup_export_path()
    screen, font = initialize_pygame(grid_size)

    # Tracking-Listen
    frames = []
    step_count = 0
    total_reward = 0

    print(f"Agent startet mit Policy-Ausführung...")

    # Episode starten
    obs, _ = env.reset()
    state = obs_to_state(obs, ENV_MODE, grid_size)
    agent_pos = get_position_from_obs(obs, ENV_MODE)

    # Ersten Frame zeichnen
    draw_grid(screen, font, env, agent_pos, Q, grid_size, frames, save_frame=EXPORT_FRAMES)
    time.sleep(0.5)

    print(f"Startposition: {agent_pos}")

    # Hauptschleife
    running = True
    while running:
        # Pygame Events verarbeiten
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        if not running:
            break

        # Beste Aktion auswählen
        action = get_best_action(Q, state)

        # Schritt ausführen
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Zustand aktualisieren
        state = obs_to_state(obs, ENV_MODE, grid_size)
        agent_pos = get_position_from_obs(obs, ENV_MODE)

        # Tracking
        step_count += 1
        total_reward += reward

        # Debug-Info
        print_step_info(step_count, agent_pos, action, reward, terminated, truncated)

        # Grid neu zeichnen
        draw_grid(screen, font, env, agent_pos, Q, grid_size, frames, save_frame=EXPORT_FRAMES)

        # Verzögerung
        time.sleep(FRAME_DELAY)

        # Episode beenden
        if done:
            success = check_success(reward, ENV_MODE)
            print(f"\nEpisode beendet nach {step_count} Schritten!")
            print(f"Endposition: {agent_pos}")
            print(f"Gesamt-Reward: {total_reward}")
            print(f"Erfolgreich: {'Ja' if success else 'Nein'}")
            time.sleep(1.5)
            running = False

        # Sicherheitsabbruch
        if step_count >= MAX_STEPS:
            print(f"Maximale Schrittanzahl ({MAX_STEPS}) erreicht - Episode abgebrochen")
            running = False

    # Pygame beenden
    pygame.quit()

    # Export-Dateien erstellen
    if EXPORT_FRAMES and frames:
        create_export_files(frames)

    # Zusammenfassung
    print_execution_summary(step_count, total_reward, ENV_MODE)


# Erfolgserkennung je nach Umgebung
def check_success(reward, env_mode):
    if env_mode == "container":
        return reward == REWARDS["dropoff"]
    else:  # Grid-Environment
        return reward == REWARDS["goal"]


# Ausführungszusammenfassung
def print_execution_summary(steps, total_reward, env_mode):
    print(f"\n" + "=" * 60)
    print(f"AUSFÜHRUNGSZUSAMMENFASSUNG ({env_mode}-Modus)")
    print("=" * 60)
    print(f"Gesamte Schritte: {steps}")
    print(f"Gesamt-Reward: {total_reward}")
    print(f"Durchschnittlicher Reward pro Schritt: {total_reward / steps:.2f}" if steps > 0 else "N/A")

    if EXPORT_FRAMES:
        print(f"Frames exportiert nach: {EXPORT_PATH}")

    print("Visualisierung abgeschlossen!")


# ============================================================================
# AUSFÜHRUNG
# ============================================================================

if __name__ == "__main__":
    run_agent()