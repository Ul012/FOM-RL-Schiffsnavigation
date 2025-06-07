# train.py

# ============================================================================
# Imports
# ============================================================================

import sys
import os
from pathlib import Path

# Projektstruktur für Import anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Drittanbieter
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Lokale Module
from config import (ENV_MODE, EPISODES, MAX_STEPS, ALPHA, GAMMA, EPSILON,
                    ACTIONS, REWARDS, GRID_SIZE, Q_TABLE_PATH,
                    EXPORT_PDF, EXPORT_PATH)
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


def obs_to_state(obs, env_mode=ENV_MODE, grid_size=None):
    """Zustandscodierung je nach Umgebung"""
    if env_mode == "container":
        return obs[0] * grid_size + obs[1] + (grid_size * grid_size) * obs[2]
    return obs


def initialize_q_table(env):
    """Q-Tabelle initialisieren"""
    n_states = env.observation_space.n if hasattr(env.observation_space, 'n') else np.prod(env.observation_space.nvec)
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    print(f"Q-Tabelle initialisiert: {n_states} Zustände, {n_actions} Aktionen")
    return Q, n_states, n_actions


def select_action(Q, state, epsilon, n_actions):
    """Epsilon-greedy Aktionsauswahl"""
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)
    else:
        return np.argmax(Q[state])


def update_q_value(Q, state, action, reward, next_state, alpha=ALPHA, gamma=GAMMA):
    """Q-Wert Update (Q-Learning)"""
    Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])


def check_success(reward, env_mode):
    """Erfolgserkennung je nach Umgebung"""
    if env_mode == "container":
        return reward == REWARDS["dropoff"]
    else:  # Grid-Environment
        return reward == REWARDS["goal"]


def save_q_table(Q, env_mode=ENV_MODE):
    """Q-Tabelle speichern"""
    filepath = f"q_table_{env_mode}.npy"
    np.save(filepath, Q)
    print(f"Q-Tabelle gespeichert: {filepath}")


def setup_export():
    """Export-Ordner erstellen"""
    if EXPORT_PDF:
        Path(EXPORT_PATH).mkdir(exist_ok=True)


# ============================================================================
# Visualisierungsfunktionen
# ============================================================================

def create_learning_curve(rewards_per_episode, env_mode, window_size=20):
    """Lernkurve mit Moving Average"""
    plt.figure(figsize=(12, 6))

    # Raw Rewards
    plt.plot(rewards_per_episode, alpha=0.3, label="Raw Reward", color='blue')

    # Moving Average
    if len(rewards_per_episode) >= window_size:
        moving_avg = np.convolve(rewards_per_episode, np.ones(window_size) / window_size, mode='valid')
        plt.plot(range(window_size - 1, len(rewards_per_episode)), moving_avg,
                 label=f"Moving Average ({window_size})", color='red', linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Gesamtreward")
    plt.title(f"Lernkurve ({env_mode}-Modus)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # PDF Export
    if EXPORT_PDF:
        plt.savefig(f"{EXPORT_PATH}/train_learning_curve.pdf", format='pdf', bbox_inches='tight')
        print(f"Learning Curve gespeichert: {EXPORT_PATH}/train_learning_curve.pdf")

    plt.show()


def create_success_curve(success_per_episode, env_mode):
    """Erfolgskurve"""
    plt.figure(figsize=(12, 4))
    plt.plot(success_per_episode, label="Ziel erreicht", color='green', alpha=0.7, linewidth=1)

    # Moving Average für Erfolgsrate
    window_size = min(max(10, EPISODES // 20), len(success_per_episode) // 10)
    if len(success_per_episode) >= window_size:
        success_moving_avg = np.convolve(success_per_episode, np.ones(window_size) / window_size, mode='valid')
        plt.plot(range(window_size - 1, len(success_per_episode)), success_moving_avg,
                 label=f"Erfolgsrate MA ({window_size})", color='darkgreen', linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Erfolg (0/1)")
    plt.title(f"Zielerreichung pro Episode ({env_mode}-Modus)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # PDF Export
    if EXPORT_PDF:
        plt.savefig(f"{EXPORT_PATH}/train_success_curve.pdf", format='pdf', bbox_inches='tight')
        print(f"Success Curve gespeichert: {EXPORT_PATH}/train_success_curve.pdf")

    plt.show()


def create_training_statistics(rewards_per_episode, success_per_episode, env_mode):
    """Trainingsstatistiken"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Reward-Histogramm
    ax1.hist(rewards_per_episode, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.set_title("Verteilung der Episode-Rewards")
    ax1.set_xlabel("Reward")
    ax1.set_ylabel("Häufigkeit")
    ax1.axvline(np.mean(rewards_per_episode), color='red', linestyle='--',
                label=f'Durchschnitt: {np.mean(rewards_per_episode):.2f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Erfolgsrate über Zeit (kumulativ)
    cumulative_success = np.cumsum(success_per_episode) / np.arange(1, len(success_per_episode) + 1)
    ax2.plot(cumulative_success, color='green', linewidth=2)
    ax2.set_title("Kumulative Erfolgsrate")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Erfolgsrate")
    ax2.grid(True, alpha=0.3)

    # Reward-Entwicklung (letzte X Episoden)
    display_episodes = min(1000, EPISODES // 2)
    ax3.plot(rewards_per_episode[-display_episodes:], alpha=0.6, color='blue')
    ax3.set_title(f"Reward-Entwicklung (letzte {display_episodes} Episoden)")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Reward")
    ax3.grid(True, alpha=0.3)

    # Erfolg vs. Misserfolg (Balkendiagramm)
    total_success = sum(success_per_episode)
    total_failure = len(success_per_episode) - total_success
    ax4.bar(['Erfolg', 'Misserfolg'], [total_success, total_failure],
            color=['green', 'red'], alpha=0.7)
    ax4.set_title("Erfolg vs. Misserfolg")
    ax4.set_ylabel("Anzahl Episoden")

    # Prozentwerte auf Balken
    for i, v in enumerate([total_success, total_failure]):
        percentage = (v / len(success_per_episode)) * 100
        ax4.text(i, v + len(success_per_episode) * 0.01, f'{v}\n({percentage:.1f}%)',
                 ha='center', va='bottom')

    plt.suptitle(f"Trainingsstatistiken ({env_mode}-Modus)", fontsize=16)
    plt.tight_layout()

    # PDF Export
    if EXPORT_PDF:
        plt.savefig(f"{EXPORT_PATH}/train_statistics.pdf", format='pdf', bbox_inches='tight')
        print(f"Training Statistics gespeichert: {EXPORT_PATH}/train_statistics.pdf")

    plt.show()


# ============================================================================
# Hauptfunktion
# ============================================================================

def train_agent():
    """Training des Q-Learning Agenten über mehrere Episoden"""
    # Initialisierung
    env, grid_size = initialize_environment()
    Q, n_states, n_actions = initialize_q_table(env)
    setup_export()

    # Tracking-Listen
    rewards_per_episode = []
    success_per_episode = []
    steps_per_episode = []

    print(f"Starte Training mit {EPISODES} Episoden...")
    print(f"Hyperparameter: α={ALPHA}, γ={GAMMA}, ε={EPSILON}")

    # Training Loop
    for episode in range(EPISODES):
        obs, _ = env.reset()
        state = obs_to_state(obs, ENV_MODE, grid_size)
        total_reward = 0
        steps = 0
        success = False

        # Episode durchführen
        for step in range(MAX_STEPS):
            # Aktion auswählen
            action = select_action(Q, state, EPSILON, n_actions)

            # Schritt ausführen
            obs, reward, terminated, truncated, _ = env.step(action)
            next_state = obs_to_state(obs, ENV_MODE, grid_size)
            done = terminated or truncated

            # Q-Wert aktualisieren
            update_q_value(Q, state, action, reward, next_state)

            # Tracking
            state = next_state
            total_reward += reward
            steps += 1

            # Erfolgserkennung
            if check_success(reward, ENV_MODE):
                success = True

            if done:
                break

        # Episode-Statistiken speichern
        rewards_per_episode.append(total_reward)
        success_per_episode.append(1 if success else 0)
        steps_per_episode.append(steps)

        # Fortschritt ausgeben
        if (episode + 1) % max(1, EPISODES // 10) == 0:
            recent_episodes = min(100, episode + 1)
            recent_success_rate = np.mean(success_per_episode[-recent_episodes:]) * 100
            print(f"Episode {episode + 1}/{EPISODES}: "
                  f"Reward={total_reward:.2f}, Steps={steps}, "
                  f"Erfolgsrate (letzte {recent_episodes}): {recent_success_rate:.1f}%")

    # Training abgeschlossen
    print_training_results(rewards_per_episode, success_per_episode, steps_per_episode)

    # Q-Tabelle speichern
    save_q_table(Q, ENV_MODE)

    # Visualisierungen erstellen
    create_learning_curve(rewards_per_episode, ENV_MODE)
    create_success_curve(success_per_episode, ENV_MODE)
    create_training_statistics(rewards_per_episode, success_per_episode, ENV_MODE)

    return Q, rewards_per_episode, success_per_episode


def print_training_results(rewards_per_episode, success_per_episode, steps_per_episode):
    """Trainingsergebnisse ausgeben"""
    total_successes = sum(success_per_episode)
    avg_reward = np.mean(rewards_per_episode)
    avg_steps = np.mean(steps_per_episode)

    print(f"\n" + "=" * 60)
    print(f"TRAININGSERGEBNISSE ({EPISODES} Episoden, Modus: {ENV_MODE})")
    print("=" * 60)

    print(f"\nErfolgsstatistik:")
    print(f"  Erfolgreiche Episoden: {total_successes}/{EPISODES} ({(total_successes / EPISODES) * 100:.1f}%)")

    # Erfolgsrate in verschiedenen Phasen
    phase_size = min(500, EPISODES // 4)
    if len(success_per_episode) >= phase_size * 2:
        early_success = np.mean(success_per_episode[:phase_size]) * 100
        late_success = np.mean(success_per_episode[-phase_size:]) * 100
        print(f"  Frühe Phase (erste {phase_size}): {early_success:.1f}%")
        print(f"  Späte Phase (letzte {phase_size}): {late_success:.1f}%")
        print(f"  Verbesserung: {late_success - early_success:+.1f} Prozentpunkte")

    print(f"\nReward-Statistiken:")
    print(f"  Durchschnitt: {avg_reward:.2f}")
    print(f"  Minimum: {min(rewards_per_episode):.2f}")
    print(f"  Maximum: {max(rewards_per_episode):.2f}")
    print(f"  Standardabweichung: {np.std(rewards_per_episode):.2f}")

    print(f"\nSchritt-Statistiken:")
    print(f"  Durchschnittliche Schritte: {avg_steps:.1f}")
    print(f"  Minimum: {min(steps_per_episode)}")
    print(f"  Maximum: {max(steps_per_episode)}")

    print(f"\nHyperparameter:")
    print(f"  Lernrate (α): {ALPHA}")
    print(f"  Discount Factor (γ): {GAMMA}")
    print(f"  Epsilon (ε): {EPSILON}")

    if EXPORT_PDF:
        print(f"\nPDF-Exports gespeichert in: {EXPORT_PATH}")


# ============================================================================
# Ausführung
# ============================================================================

if __name__ == "__main__":
    train_agent()