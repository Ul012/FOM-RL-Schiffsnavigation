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

# Lokale Module
from config import REWARDS, EXPORT_PDF, EXPORT_PATH, MAX_STEPS, LOOP_THRESHOLD
from navigation.environment.grid_environment import GridEnvironment
from navigation.environment.container_environment import ContainerShipEnv

# ============================================================================
# Konfiguration
# ============================================================================

# Szenarien-Definition
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

# Evaluation-Parameter
EVAL_EPISODES = 500
CONVERGENCE_THRESHOLD = 0.8  # 80% Erfolgsrate für Konvergenz


# ============================================================================
# Hilfsfunktionen
# ============================================================================

def initialize_environment(scenario_config):
    """Umgebung für Szenario initialisieren"""
    if scenario_config["environment"] == "container":
        env = ContainerShipEnv()
    else:
        env = GridEnvironment(mode=scenario_config["env_mode"])

    grid_size = env.grid_size
    return env, grid_size


def load_q_table(filepath):
    """Q-Tabelle laden"""
    try:
        Q = np.load(filepath)
        return Q
    except FileNotFoundError:
        print(f"WARNUNG: Q-Tabelle nicht gefunden: {filepath}")
        return None


def obs_to_state(obs, env_mode, grid_size):
    """Zustandscodierung je nach Umgebung"""
    if env_mode == "container":
        return obs[0] * grid_size + obs[1] + (grid_size * grid_size) * obs[2]
    return obs


def check_success(reward, env_mode):
    """Erfolg prüfen je nach Umgebung"""
    if env_mode == "container":
        return reward == REWARDS["dropoff"]
    else:
        return reward == REWARDS["goal"]


def setup_export():
    """Export-Ordner erstellen"""
    if EXPORT_PDF:
        Path(EXPORT_PATH).mkdir(exist_ok=True)


# ============================================================================
# Evaluationsfunktionen
# ============================================================================

def evaluate_single_scenario(scenario_name, scenario_config):
    """Einzelnes Szenario evaluieren"""
    print(f"Evaluiere Szenario: {scenario_name}")

    # Umgebung und Q-Tabelle laden
    env, grid_size = initialize_environment(scenario_config)
    Q = load_q_table(scenario_config["q_table_path"])

    if Q is None:
        return None

    # Metriken sammeln
    results = {
        "success_count": 0,
        "total_episodes": EVAL_EPISODES,
        "episode_rewards": [],
        "steps_to_goal": [],
        "success_per_episode": [],
        "causes": defaultdict(int),
        "timeout_count": 0,
        "loop_abort_count": 0,
        "obstacle_count": 0
    }

    # Episoden durchführen
    for episode in range(EVAL_EPISODES):
        obs, _ = env.reset()
        state = obs_to_state(obs, scenario_config["env_mode"], grid_size)
        episode_reward = 0
        steps = 0
        success = False
        cause = "timeout"
        visited_states = {}  # Für Grid-Environment Schleifenerkennung

        # Episode ausführen
        for step in range(MAX_STEPS):
            action = np.argmax(Q[state])
            obs, reward, terminated, _, _ = env.step(action)
            next_state = obs_to_state(obs, scenario_config["env_mode"], grid_size)
            episode_reward += reward
            steps += 1

            # Schleifenerkennung für Grid-Environment (da nicht in Environment implementiert)
            if scenario_config["environment"] == "grid":
                if next_state in visited_states:
                    visited_states[next_state] += 1
                else:
                    visited_states[next_state] = 1

                if visited_states[next_state] >= LOOP_THRESHOLD:
                    cause = "loop_abort"
                    terminated = True
                    break

            # Erfolg prüfen
            if check_success(reward, scenario_config["env_mode"]):
                success = True
                cause = "success"
                break

            # Andere Terminierungsgründe
            if terminated:
                if reward == REWARDS["obstacle"] or (
                        reward < REWARDS["step"] and reward != REWARDS["goal"] and reward != REWARDS["dropoff"]):
                    cause = "obstacle"
                elif reward == REWARDS["loop_abort"]:
                    cause = "loop_abort"
                break

            state = next_state

        # Timeout-Check
        if not terminated and steps >= MAX_STEPS:
            cause = "timeout"

        # Ergebnisse sammeln
        results["episode_rewards"].append(episode_reward)
        results["success_per_episode"].append(1 if success else 0)
        results["causes"][cause] += 1

        if success:
            results["success_count"] += 1
            results["steps_to_goal"].append(steps)

        if cause == "timeout":
            results["timeout_count"] += 1
        elif cause == "loop_abort":
            results["loop_abort_count"] += 1
        elif cause == "obstacle":
            results["obstacle_count"] += 1

    return results


def calculate_convergence_episode(success_per_episode, window_size=50):
    """Episode berechnen, ab der Konvergenz erreicht wird"""
    if len(success_per_episode) < window_size:
        return None

    for i in range(window_size, len(success_per_episode)):
        window_success_rate = np.mean(success_per_episode[i - window_size:i])
        if window_success_rate >= CONVERGENCE_THRESHOLD:
            return i

    return None


def calculate_metrics(scenario_results):
    """Metriken aus Evaluationsergebnissen berechnen"""
    if scenario_results is None:
        return None

    metrics = {}

    # Basis-Metriken
    metrics["success_rate"] = scenario_results["success_count"] / scenario_results["total_episodes"]
    metrics["avg_reward"] = np.mean(scenario_results["episode_rewards"])
    metrics["reward_std"] = np.std(scenario_results["episode_rewards"])

    # Effizienz-Metriken
    if scenario_results["steps_to_goal"]:
        metrics["avg_steps_to_goal"] = np.mean(scenario_results["steps_to_goal"])
        metrics["steps_std"] = np.std(scenario_results["steps_to_goal"])
    else:
        metrics["avg_steps_to_goal"] = None
        metrics["steps_std"] = None

    # Failure-Raten
    total = scenario_results["total_episodes"]
    metrics["timeout_rate"] = scenario_results["timeout_count"] / total
    metrics["loop_abort_rate"] = scenario_results["loop_abort_count"] / total
    metrics["obstacle_rate"] = scenario_results["obstacle_count"] / total

    # Konvergenz (falls Training-Daten verfügbar wären)
    # Hier als Platzhalter - könnte aus Training-Logs gelesen werden
    metrics["convergence_episode"] = None

    return metrics


# ============================================================================
# Visualisierungsfunktionen
# ============================================================================

def create_comparison_table(all_metrics):
    """Vergleichstabelle erstellen"""
    df_data = []

    for scenario_name, metrics in all_metrics.items():
        if metrics is None:
            continue

        row = {
            "Szenario": scenario_name,
            "Erfolgsrate (%)": f"{metrics['success_rate'] * 100:.1f}",
            "Ø Schritte zum Ziel": f"{metrics['avg_steps_to_goal']:.1f}" if metrics['avg_steps_to_goal'] else "N/A",
            "Ø Reward": f"{metrics['avg_reward']:.2f}",
            "Reward Std": f"{metrics['reward_std']:.2f}",
            "Timeout (%)": f"{metrics['timeout_rate'] * 100:.1f}",
            "Schleifenabbruch (%)": f"{metrics['loop_abort_rate'] * 100:.1f}",
            "Hindernis-Kollision (%)": f"{metrics['obstacle_rate'] * 100:.1f}"
        }
        df_data.append(row)

    df = pd.DataFrame(df_data)
    print("\n" + "=" * 120)
    print("SZENARIEN-VERGLEICH")
    print("=" * 120)
    print(df.to_string(index=False))

    # Zusätzliche Detailanalyse
    print("\n" + "=" * 120)
    print("FAILURE-MODI VERTEILUNG")
    print("=" * 120)
    print("Szenario".ljust(20) + "Erfolg".ljust(10) + "Timeout".ljust(12) + "Schleifen".ljust(12) + "Hindernisse".ljust(
        12) + "Gesamt")
    print("-" * 120)

    for scenario_name, metrics in all_metrics.items():
        if metrics is None:
            continue

        success = metrics['success_rate'] * 100
        timeout = metrics['timeout_rate'] * 100
        loop = metrics['loop_abort_rate'] * 100
        obstacle = metrics['obstacle_rate'] * 100
        total = success + timeout + loop + obstacle

        print(f"{scenario_name:<20}{success:>6.1f}%{timeout:>10.1f}%{loop:>10.1f}%{obstacle:>10.1f}%{total:>10.1f}%")

    print("=" * 120)

    return df


def create_success_rate_comparison(all_metrics):
    """Erfolgsraten-Vergleich als Balkendiagramm"""
    scenarios = []
    success_rates = []

    for scenario_name, metrics in all_metrics.items():
        if metrics is None:
            continue
        scenarios.append(scenario_name)
        success_rates.append(metrics["success_rate"] * 100)

    plt.figure(figsize=(12, 7))
    bars = plt.bar(scenarios, success_rates, color='lightgray', edgecolor='black', alpha=0.8)

    plt.title("Erfolgsraten-Vergleich der Szenarien", fontsize=14, fontweight='bold', pad=30)
    plt.xlabel("Szenario")
    plt.ylabel("Erfolgsrate (%)")
    plt.ylim(0, 110)  # Mehr Platz für Beschriftungen

    # Werte auf Balken anzeigen
    for bar, rate in zip(bars, success_rates):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if EXPORT_PDF:
        plt.savefig(f"{EXPORT_PATH}/compare_success_rates.pdf", format='pdf', bbox_inches='tight')
        print(f"Success Rate Comparison gespeichert: {EXPORT_PATH}/compare_success_rates.pdf")

    plt.show()


def create_radar_chart(all_metrics):
    """Radar Chart für mehrdimensionalen Vergleich"""
    # Metriken für Radar Chart normalisieren (0-1)
    radar_labels = ["Erfolgsrate", "Reward-Qualität", "Effizienz", "Stabilität"]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
    angles += angles[:1]  # Schließe den Kreis

    colors = ['blue', 'green', 'orange', 'red', 'purple']

    for i, (scenario_name, metrics) in enumerate(all_metrics.items()):
        if metrics is None:
            continue

        # Normalisierte Werte berechnen
        values = []

        # 1. Erfolgsrate (bereits 0-1)
        values.append(metrics["success_rate"])

        # 2. Reward-Qualität: Normalisiert auf positive Skala
        # Höhere (weniger negative) Rewards = besser
        max_possible_reward = REWARDS["goal"] if "container" not in scenario_name else REWARDS["dropoff"]
        min_expected = REWARDS["step"] * MAX_STEPS  # Worst case: nur Schrittstrafen
        reward_norm = max(0, min(1, (metrics["avg_reward"] - min_expected) / (max_possible_reward - min_expected)))
        values.append(reward_norm)

        # 3. Effizienz: Weniger Schritte zum Ziel = besser
        # Nur für erfolgreiche Episoden, normalisiert auf 1-50 Schritte
        if metrics["avg_steps_to_goal"] and metrics["avg_steps_to_goal"] > 0:
            efficiency = max(0, min(1, (MAX_STEPS - metrics["avg_steps_to_goal"]) / MAX_STEPS))
        else:
            efficiency = 0  # Keine erfolgreichen Episoden
        values.append(efficiency)

        # 4. Stabilität: Weniger Standardabweichung = besser
        # Normalisiert basierend auf typischen Reward-Schwankungen
        max_expected_std = abs(REWARDS["goal"] - REWARDS["step"] * MAX_STEPS)
        stability = max(0, min(1, 1 - (metrics["reward_std"] / max_expected_std)))
        values.append(stability)

        values += values[:1]  # Schließe den Kreis

        ax.plot(angles, values, 'o-', linewidth=2, label=scenario_name, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels)
    ax.set_ylim(0, 1)
    ax.set_title("Mehrdimensionaler Szenarien-Vergleich", size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)

    # Erklärungen hinzufügen
    plt.figtext(0.02, 0.02,
                "Effizienz: Weniger Schritte zum Ziel = höhere Effizienz\n" +
                "Stabilität: Geringere Reward-Schwankungen = höhere Stabilität",
                fontsize=8, ha='left')

    plt.tight_layout()

    if EXPORT_PDF:
        plt.savefig(f"{EXPORT_PATH}/compare_radar_chart.pdf", format='pdf', bbox_inches='tight')
        print(f"Radar Chart gespeichert: {EXPORT_PATH}/compare_radar_chart.pdf")

    plt.show()


def create_failure_modes_comparison(all_metrics):
    """Vergleich der Failure-Modi als Stacked Bar Chart"""
    scenarios = []
    timeout_rates = []
    loop_rates = []
    obstacle_rates = []
    success_rates = []

    for scenario_name, metrics in all_metrics.items():
        if metrics is None:
            continue
        scenarios.append(scenario_name)
        timeout_rates.append(metrics["timeout_rate"] * 100)
        loop_rates.append(metrics["loop_abort_rate"] * 100)
        obstacle_rates.append(metrics["obstacle_rate"] * 100)
        success_rates.append(metrics["success_rate"] * 100)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Stacked Bar Chart
    bottom_success = np.zeros(len(scenarios))
    bottom_timeout = success_rates
    bottom_loop = np.array(success_rates) + np.array(timeout_rates)
    bottom_obstacle = np.array(success_rates) + np.array(timeout_rates) + np.array(loop_rates)

    bars1 = ax.bar(scenarios, success_rates, label='Erfolg', color='green', alpha=0.8)
    bars2 = ax.bar(scenarios, timeout_rates, bottom=bottom_timeout, label='Timeout', color='red', alpha=0.7)
    bars3 = ax.bar(scenarios, loop_rates, bottom=bottom_loop, label='Schleifenabbruch', color='orange', alpha=0.7)
    bars4 = ax.bar(scenarios, obstacle_rates, bottom=bottom_obstacle, label='Hindernis-Kollision', color='brown',
                   alpha=0.7)

    ax.set_xlabel('Szenario')
    ax.set_ylabel('Anteil (%)')
    ax.set_title('Verteilung der Episode-Ergebnisse (Stacked)', fontweight='bold', fontsize=14)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Prozentangaben auf Segmenten
    for i, scenario in enumerate(scenarios):
        # Erfolg
        if success_rates[i] > 5:  # Nur bei größeren Segmenten
            ax.text(i, success_rates[i] / 2, f'{success_rates[i]:.0f}%',
                    ha='center', va='center', fontweight='bold', color='white')

        # Timeout
        if timeout_rates[i] > 5:
            ax.text(i, bottom_timeout[i] + timeout_rates[i] / 2, f'{timeout_rates[i]:.0f}%',
                    ha='center', va='center', fontweight='bold', color='white')

        # Loop
        if loop_rates[i] > 5:
            ax.text(i, bottom_loop[i] + loop_rates[i] / 2, f'{loop_rates[i]:.0f}%',
                    ha='center', va='center', fontweight='bold', color='white')

        # Obstacle
        if obstacle_rates[i] > 5:
            ax.text(i, bottom_obstacle[i] + obstacle_rates[i] / 2, f'{obstacle_rates[i]:.0f}%',
                    ha='center', va='center', fontweight='bold', color='white')

    plt.xticks(rotation=45)
    plt.tight_layout()

    if EXPORT_PDF:
        plt.savefig(f"{EXPORT_PATH}/compare_failure_modes.pdf", format='pdf', bbox_inches='tight')
        print(f"Failure Modes Comparison gespeichert: {EXPORT_PATH}/compare_failure_modes.pdf")

    plt.show()


# ============================================================================
# Hauptfunktion
# ============================================================================

def compare_all_scenarios():
    """Alle Szenarien vergleichen"""
    print("Starte Szenarien-Vergleich...")
    setup_export()

    all_results = {}
    all_metrics = {}

    # Alle Szenarien evaluieren
    for scenario_name, scenario_config in SCENARIOS.items():
        results = evaluate_single_scenario(scenario_name, scenario_config)
        all_results[scenario_name] = results
        all_metrics[scenario_name] = calculate_metrics(results)

    # Vergleichstabelle erstellen
    df = create_comparison_table(all_metrics)

    # Visualisierungen erstellen
    create_success_rate_comparison(all_metrics)
    create_radar_chart(all_metrics)
    create_failure_modes_comparison(all_metrics)

    print(f"\n{'=' * 60}")
    print("ZUSAMMENFASSUNG")
    print(f"{'=' * 60}")

    # Beste/Schlechteste Szenarien identifizieren
    valid_metrics = {k: v for k, v in all_metrics.items() if v is not None}

    if valid_metrics:
        best_success = max(valid_metrics.items(), key=lambda x: x[1]["success_rate"])
        best_efficiency = min([item for item in valid_metrics.items() if item[1]["avg_steps_to_goal"]],
                              key=lambda x: x[1]["avg_steps_to_goal"])
        most_stable = min(valid_metrics.items(), key=lambda x: x[1]["reward_std"])

        print(f"Beste Erfolgsrate: {best_success[0]} ({best_success[1]['success_rate'] * 100:.1f}%)")
        print(f"Effizientestes Szenario: {best_efficiency[0]} ({best_efficiency[1]['avg_steps_to_goal']:.1f} Schritte)")
        print(f"Stabilstes Szenario: {most_stable[0]} (Std: {most_stable[1]['reward_std']:.2f})")

    if EXPORT_PDF:
        print(f"\nAlle Visualisierungen gespeichert in: {EXPORT_PATH}")

    return all_results, all_metrics


# ============================================================================
# Ausführung
# ============================================================================

if __name__ == "__main__":
    compare_all_scenarios()