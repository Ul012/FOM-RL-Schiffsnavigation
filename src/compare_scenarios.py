# compare_scenarios.py

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
import pandas as pd
from collections import defaultdict
import random

# Lokale Module
from config import (EVAL_MAX_STEPS, LOOP_THRESHOLD, REWARDS, EXPORT_PDF, EXPORT_PATH,
                    EVAL_EPISODES, CONVERGENCE_THRESHOLD, SEED)
from navigation.environment.grid_environment import GridEnvironment
from navigation.environment.container_environment import ContainerShipEnv

# ============================================================================
# Szenarien-Definition
# ============================================================================

SCENARIOS = {
    "static": {
        "env_mode": "static",
        "q_table_path": "q_table_static.npy",
        "environment": "grid"
    },
    "random_start": {
        "env_mode": "random_start",
        "q_table_path": "q_table_random_start.npy",
        "environment": "grid"
    },
    "random_goal": {
        "env_mode": "random_goal",
        "q_table_path": "q_table_random_goal.npy",
        "environment": "grid"
    },
    "random_obstacles": {
        "env_mode": "random_obstacles",
        "q_table_path": "q_table_random_obstacles.npy",
        "environment": "grid"
    },
    "container": {
        "env_mode": "container",
        "q_table_path": "q_table_container.npy",
        "environment": "container"
    }
}


# ============================================================================
# Hilfsfunktionen
# ============================================================================

# Seed-Konfiguration für Reproduzierbarkeit
def set_all_seeds(seed=None):
    if seed is None:
        seed = SEED

    random.seed(seed)
    np.random.seed(seed)
    print(f"Seeds gesetzt auf: {seed}")
    return seed


# Initialisierung der Umgebung für ein Szenario
def initialize_environment(scenario_config):
    if scenario_config["environment"] == "container":
        env = ContainerShipEnv()
    else:
        env = GridEnvironment(mode=scenario_config["env_mode"])
    return env, env.grid_size


# Laden der Q-Tabelle
def load_q_table(filepath):
    try:
        return np.load(filepath)
    except FileNotFoundError:
        print(f"WARNUNG: Q-Tabelle nicht gefunden: {filepath}")
        return None


# Zustandscodierung je nach Umgebungstyp
def obs_to_state(obs, env_mode, grid_size):
    if env_mode == "container":
        return obs[0] * grid_size + obs[1] + (grid_size * grid_size) * obs[2]
    return obs


# Erfolgsprüfung je nach Umgebungstyp
def check_success(reward, env_mode):
    if env_mode == "container":
        return reward == REWARDS["dropoff"]
    else:
        return reward == REWARDS["goal"]


# Erstellung des Export-Ordners
def setup_export():
    if EXPORT_PDF:
        Path(EXPORT_PATH).mkdir(exist_ok=True)


# ============================================================================
# Evaluation
# ============================================================================

# Evaluation eines einzelnen Szenarios
def evaluate_single_scenario(scenario_name, scenario_config):
    print(f"Evaluiere Szenario: {scenario_name}")

    env, grid_size = initialize_environment(scenario_config)
    Q = load_q_table(scenario_config["q_table_path"])

    if Q is None:
        return None

    results = {
        "success_count": 0,
        "timeout_count": 0,
        "loop_abort_count": 0,
        "obstacle_count": 0,
        "episode_rewards": [],
        "steps_to_goal": [],
        "success_per_episode": []
    }

    for episode in range(EVAL_EPISODES):
        obs, _ = env.reset()
        state = obs_to_state(obs, scenario_config["env_mode"], grid_size)
        episode_reward = 0
        steps = 0
        visited_states = {}

        terminated_by_environment = False
        while steps < EVAL_MAX_STEPS:
            action = np.argmax(Q[state])
            obs, reward, terminated, _, _ = env.step(action)
            next_state = obs_to_state(obs, scenario_config["env_mode"], grid_size)
            episode_reward += reward
            steps += 1

            if check_success(reward, scenario_config["env_mode"]):
                results["success_count"] += 1
                results["steps_to_goal"].append(steps)
                break

            visited_states[next_state] = visited_states.get(next_state, 0) + 1
            if visited_states[next_state] >= LOOP_THRESHOLD:
                results["loop_abort_count"] += 1
                break

            if terminated:
                results["obstacle_count"] += 1
                terminated_by_environment = True
                break

            state = next_state
        else:
            if not terminated_by_environment:
                results["timeout_count"] += 1

        results["episode_rewards"].append(episode_reward)
        results["success_per_episode"].append(
            1 if results["success_count"] > len(results["success_per_episode"]) else 0)

    total = results["success_count"] + results["timeout_count"] + results["loop_abort_count"] + results[
        "obstacle_count"]
    print(f"  Erfolg: {results['success_count']}, Timeout: {results['timeout_count']}, "
          f"Schleifen: {results['loop_abort_count']}, Hindernisse: {results['obstacle_count']}, Total: {total}")

    return results


# Berechnung der Leistungsmetriken
def calculate_metrics(scenario_results):
    if scenario_results is None:
        return None

    total = EVAL_EPISODES
    return {
        "success_rate": scenario_results["success_count"] / total,
        "timeout_rate": scenario_results["timeout_count"] / total,
        "loop_abort_rate": scenario_results["loop_abort_count"] / total,
        "obstacle_rate": scenario_results["obstacle_count"] / total,
        "avg_reward": np.mean(scenario_results["episode_rewards"]),
        "reward_std": np.std(scenario_results["episode_rewards"]),
        "avg_steps_to_goal": np.mean(scenario_results["steps_to_goal"]) if scenario_results["steps_to_goal"] else None
    }


# ============================================================================
# Visualisierung
# ============================================================================

# Erstellung der Vergleichstabelle
def create_comparison_table(all_metrics):
    data = []
    for scenario_name, metrics in all_metrics.items():
        if metrics is None:
            continue
        data.append({
            "Szenario": scenario_name,
            "Erfolg (%)": f"{metrics['success_rate'] * 100:.1f}",
            "Timeout (%)": f"{metrics['timeout_rate'] * 100:.1f}",
            "Schleifen (%)": f"{metrics['loop_abort_rate'] * 100:.1f}",
            "Hindernisse (%)": f"{metrics['obstacle_rate'] * 100:.1f}",
            "Ø Reward": f"{metrics['avg_reward']:.2f}",
            "Ø Schritte": f"{metrics['avg_steps_to_goal']:.1f}" if metrics['avg_steps_to_goal'] else "N/A"
        })

    df = pd.DataFrame(data)
    print("\n" + "=" * 80)
    print("SZENARIEN-VERGLEICH")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    return df


# Visualisierung des Erfolgsraten-Vergleichs
def create_success_rate_comparison(all_metrics):
    scenarios = [name for name, metrics in all_metrics.items() if metrics is not None]
    success_rates = [metrics["success_rate"] * 100 for name, metrics in all_metrics.items() if metrics is not None]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(scenarios, success_rates, color='lightgray', edgecolor='black')

    plt.title("Erfolgsraten-Vergleich", fontweight='bold', pad=20)
    plt.xlabel("Szenario")
    plt.ylabel("Erfolgsrate (%)")
    plt.ylim(0, 110)

    for bar, rate in zip(bars, success_rates):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if EXPORT_PDF:
        plt.savefig(f"{EXPORT_PATH}/success_rates.pdf", format='pdf', bbox_inches='tight')
    plt.show()


# Erstellung des gestapelten Balkendiagramms für Terminierungsarten
def create_stacked_failure_chart(all_metrics):
    scenarios = []
    success_rates = []
    timeout_rates = []
    loop_rates = []
    obstacle_rates = []

    for scenario_name, metrics in all_metrics.items():
        if metrics is None:
            continue
        scenarios.append(scenario_name)
        success_rates.append(metrics["success_rate"] * 100)
        timeout_rates.append(metrics["timeout_rate"] * 100)
        loop_rates.append(metrics["loop_abort_rate"] * 100)
        obstacle_rates.append(metrics["obstacle_rate"] * 100)

    fig, ax = plt.subplots(figsize=(12, 8))

    bottom_timeout = success_rates
    bottom_loop = [success_rates[i] + timeout_rates[i] for i in range(len(scenarios))]
    bottom_obstacle = [success_rates[i] + timeout_rates[i] + loop_rates[i] for i in range(len(scenarios))]

    ax.bar(scenarios, success_rates, label='Erfolg', color='green', alpha=0.8)
    ax.bar(scenarios, timeout_rates, bottom=bottom_timeout, label='Timeout', color='red', alpha=0.7)
    ax.bar(scenarios, loop_rates, bottom=bottom_loop, label='Schleifenabbruch', color='orange', alpha=0.7)
    ax.bar(scenarios, obstacle_rates, bottom=bottom_obstacle, label='Hindernis-Kollision', color='brown', alpha=0.7)

    ax.set_xlabel('Szenario')
    ax.set_ylabel('Anteil (%)')
    ax.set_title('Terminierungsarten pro Szenario', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.xticks(rotation=45)
    plt.tight_layout()

    if EXPORT_PDF:
        plt.savefig(f"{EXPORT_PATH}/failure_modes.pdf", format='pdf', bbox_inches='tight')
    plt.show()


# ============================================================================
# Hauptfunktion
# ============================================================================

# Vergleich aller Szenarien
def compare_all_scenarios():
    # Seed für Reproduzierbarkeit setzen
    set_all_seeds()

    print("Starte Szenarien-Vergleich...")
    setup_export()

    all_results = {}
    all_metrics = {}

    for scenario_name, scenario_config in SCENARIOS.items():
        results = evaluate_single_scenario(scenario_name, scenario_config)
        all_results[scenario_name] = results
        all_metrics[scenario_name] = calculate_metrics(results)

    create_comparison_table(all_metrics)
    create_success_rate_comparison(all_metrics)
    create_stacked_failure_chart(all_metrics)

    print(
        f"\n✅ Vergleich abgeschlossen. Parameter aus config.py: EVAL_MAX_STEPS={EVAL_MAX_STEPS}, LOOP_THRESHOLD={LOOP_THRESHOLD}")

    return all_results, all_metrics


# ============================================================================
# Ausführung
# ============================================================================

if __name__ == "__main__":
    compare_all_scenarios()