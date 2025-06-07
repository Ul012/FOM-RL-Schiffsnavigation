# evaluate_policy.py

# ============================================================================
# Imports
# ============================================================================

import sys
import os
from pathlib import Path

# Projektstruktur für Import anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Drittanbieter
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Lokale Module
from config import (ENV_MODE, EPISODES, MAX_STEPS, LOOP_THRESHOLD, REWARDS,
                    Q_TABLE_PATH, EXPORT_PDF, EXPORT_PATH)
from navigation.environment.grid_environment import GridEnvironment
from navigation.environment.container_environment import ContainerShipEnv


# ============================================================================
# Hilfsfunktionen
# ============================================================================

def initialize_environment():
    """Umgebung initialisieren"""
    env = ContainerShipEnv() if ENV_MODE == "container" else GridEnvironment(mode=ENV_MODE)
    grid_size = env.grid_size
    print(f"Umgebung initialisiert: {ENV_MODE}-Modus, Grid-Größe: {grid_size}x{grid_size}")
    return env, grid_size


# Q-Tabelle laden

def load_q_table(env_mode=ENV_MODE):
    filepath = f"q_table_{env_mode}.npy"
    try:
        Q = np.load(filepath)
        print(f"Q-Tabelle geladen: {filepath}")
        return Q
    except FileNotFoundError:
        print(f"FEHLER: Q-Tabelle nicht gefunden: {filepath}")
        return None


# Zustandscodierung je nach Umgebung

def obs_to_state(obs, env_mode=ENV_MODE, grid_size=None):
    if env_mode == "container":
        return obs[0] * grid_size + obs[1] + (grid_size * grid_size) * obs[2]
    return obs


# Klassifizierung des Ergebnisses einer Episode

def classify_episode_result(reward, cause, episode_reward, env_mode):
    success = False

    if env_mode == "container":
        if reward == REWARDS["dropoff"]:
            cause = "Ziel erreicht"
            success = True
        elif reward == REWARDS["loop_abort"]:
            cause = "Schleifenabbruch"
        elif reward == REWARDS["obstacle"]:
            cause = "Hindernis-Kollision"
        elif reward == REWARDS["timeout"]:
            cause = "Timeout"
    else:  # Grid-Environment
        if reward == REWARDS["goal"]:
            cause = "Ziel erreicht"
            success = True
        elif reward == REWARDS["loop_abort"] or reward == (REWARDS["loop_abort"] + REWARDS["step"]):
            cause = "Schleifenabbruch"
        elif reward == REWARDS["timeout"] or reward == (REWARDS["timeout"] + REWARDS["step"]):
            cause = "Timeout"
        elif reward == REWARDS["obstacle"] or (reward < 0 and reward != REWARDS["step"]):
            cause = "Hindernis-Kollision"

    return cause, success


# Export-Ordner erstellen

def setup_export():
    if EXPORT_PDF:
        Path(EXPORT_PATH).mkdir(exist_ok=True)


# ============================================================================
# Visualisierungsfunktionen
# ============================================================================

# Balkendiagramm für Erfolg vs. Misserfolg

def create_success_plot(results_solved, env_mode):
    plt.figure(figsize=(8, 5))

    # Zuordnung der Farben
    colors = []
    labels = list(results_solved.keys())
    for label in labels:
        if "solved" in label.lower() or "success" in label.lower():
            colors.append("green")
        else:
            colors.append("red")

    bars = plt.bar(results_solved.keys(), results_solved.values(), color=colors)
    plt.title(f"Lösungsrate ({env_mode}-Modus)")
    plt.xlabel("Ergebnis")
    plt.ylabel("Anzahl Episoden")

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.5,
                 f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()

    # PDF Export
    if EXPORT_PDF:
        plt.savefig(f"{EXPORT_PATH}/evaluate_policy_success_rate.pdf", format='pdf', bbox_inches='tight')
        print(f"Success Rate Plot gespeichert: {EXPORT_PATH}/evaluate_policy_success_rate.pdf")

    plt.show()


# Reward-Histogramm

def create_reward_histogram(rewards_all, env_mode):
    avg_reward = np.mean(rewards_all)

    plt.figure(figsize=(10, 6))
    plt.hist(rewards_all, bins=20, edgecolor='black', alpha=0.7)
    plt.title(f"Verteilung der Episoden-Rewards ({env_mode}-Modus)")
    plt.xlabel("Kumulativer Episode-Reward")
    plt.ylabel("Häufigkeit")
    plt.axvline(avg_reward, color='red', linestyle='--',
                label=f'Durchschnitt: {avg_reward:.2f}')
    plt.legend()
    plt.tight_layout()

    # PDF Export
    if EXPORT_PDF:
        plt.savefig(f"{EXPORT_PATH}/evaluate_policy_reward_histogram.pdf", format='pdf', bbox_inches='tight')
        print(f"Reward Histogram gespeichert: {EXPORT_PATH}/evaluate_policy_reward_histogram.pdf")

    plt.show()


# ============================================================================
# Hauptfunktion
# ============================================================================

# Evaluiert die trainierte Policy über mehrere Episoden

def evaluate_policy():
    # Initialisierung
    env, grid_size = initialize_environment()
    Q = load_q_table(ENV_MODE)
    setup_export()

    if Q is None:
        return

    results_cause = defaultdict(int)
    results_solved = defaultdict(int)
    rewards_all = []

    print(f"Starte Evaluation mit {EPISODES} Episoden...")

    for episode in range(EPISODES):
        obs, _ = env.reset()
        state = obs_to_state(obs, ENV_MODE, grid_size)
        episode_reward = 0
        visited_states = {}
        steps = 0
        cause = "Timeout"
        goal_reached = False
        loop_detected = False

        # Episode durchführen
        for step in range(MAX_STEPS):
            action = np.argmax(Q[state])
            obs, reward, terminated, _, _ = env.step(action)
            next_state = obs_to_state(obs, ENV_MODE, grid_size)
            episode_reward += reward
            steps += 1

            # Schleifenerkennung für Grid-Environment
            if ENV_MODE != "container":
                visited_states[next_state] = visited_states.get(next_state, 0) + 1
                if visited_states[next_state] >= LOOP_THRESHOLD:
                    cause = "Schleifenabbruch"
                    loop_detected = True
                    episode_reward += REWARDS["loop_abort"]
                    break

            # Erfolg erkennen
            if ENV_MODE == "container":
                if reward == REWARDS["dropoff"]:
                    goal_reached = True
                    cause = "Ziel erreicht"
                elif reward == REWARDS["loop_abort"]:
                    loop_detected = True
                    cause = "Schleifenabbruch"
            else:
                if reward == REWARDS["goal"]:
                    goal_reached = True
                    cause = "Ziel erreicht"

            state = next_state

            if terminated:
                # Klassifizierung basierend auf letztem Reward
                if not goal_reached and not loop_detected:
                    cause, _ = classify_episode_result(reward, cause, episode_reward, ENV_MODE)
                break

        # Timeout-Check
        if not terminated and steps >= MAX_STEPS:
            cause = "Timeout"
            episode_reward += REWARDS["timeout"]

        # Alternative Klassifizierung basierend auf Gesamt-Reward
        if cause == "Timeout" and episode_reward > 0:
            if ENV_MODE != "container" and episode_reward >= REWARDS["goal"]:
                cause = "Ziel erreicht"

        # Erfolg bestimmen
        success = (cause == "Ziel erreicht")

        # Statistiken aktualisieren
        results_cause[cause] += 1
        results_solved["solved episode" if success else "failed episode"] += 1
        rewards_all.append(episode_reward)

    # Ergebnisse ausgeben
    print_results(results_cause, results_solved, rewards_all)

    # Visualisierungen erstellen
    create_success_plot(results_solved, ENV_MODE)
    create_reward_histogram(rewards_all, ENV_MODE)


# Evaluationsergebnisse ausgeben

def print_results(results_cause, results_solved, rewards_all):
    avg_reward = np.mean(rewards_all)

    print(f"\n" + "=" * 60)
    print(f"EVALUATIONSERGEBNISSE ({EPISODES} Episoden, Modus: {ENV_MODE})")
    print("=" * 60)

    print(f"\nDetaillierte Ursachen-Verteilung:")
    total_episodes = sum(results_cause.values())
    for cause, count in sorted(results_cause.items()):
        percentage = (count / total_episodes) * 100
        print(f"  {cause}: {count} ({percentage:.1f}%)")

    print(f"\nLösungsrate:")
    solved_count = results_solved["solved episode"]
    failed_count = results_solved["failed episode"]
    print(f"  Erfolgreich: {solved_count}/{EPISODES} ({(solved_count / EPISODES) * 100:.1f}%)")
    print(f"  Fehlgeschlagen: {failed_count}/{EPISODES} ({(failed_count / EPISODES) * 100:.1f}%)")

    print(f"\nReward-Statistiken:")
    print(f"  Durchschnitt: {avg_reward:.2f}")
    print(f"  Minimum: {min(rewards_all):.2f}")
    print(f"  Maximum: {max(rewards_all):.2f}")
    print(f"  Median: {np.median(rewards_all):.2f}")

    if EXPORT_PDF:
        print(f"\nPDF-Exports gespeichert in: {EXPORT_PATH}")


# ============================================================================
# Ausführung
# ============================================================================

if __name__ == "__main__":
    evaluate_policy()