# train.py

# ============================================================================
# Imports
# ============================================================================

import sys
import os

# Projektstruktur für Import anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# Drittanbieter
import numpy as np

# Lokale Module
from config import ENV_MODE, EPISODES, MAX_STEPS, EPSILON, ALPHA, GAMMA, SEED

# Utils
from utils.common import set_all_seeds, obs_to_state, check_success, setup_export
from utils.environment import initialize_environment
from utils.qlearning import initialize_q_table, select_action, update_q_value, save_q_table
from utils.visualization import create_learning_curve, create_success_curve, create_training_statistics
from utils.reporting import print_training_results


# ============================================================================
# Hauptfunktion
# ============================================================================

# Training des Q-Learning Agenten über mehrere Episoden
def train_agent():
    # Seed für Reproduzierbarkeit setzen
    set_all_seeds()

    # Initialisierung
    env, grid_size = initialize_environment(ENV_MODE)
    Q, n_states, n_actions = initialize_q_table(env)
    setup_export()

    # Tracking-Listen
    rewards_per_episode = []
    success_per_episode = []
    steps_per_episode = []

    print(f"Starte Training mit {EPISODES} Episoden...")
    print(f"Hyperparameter: α={ALPHA}, γ={GAMMA}, ε={EPSILON}, Seed={SEED}")

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


# ============================================================================
# Ausführung
# ============================================================================

if __name__ == "__main__":
    train_agent()