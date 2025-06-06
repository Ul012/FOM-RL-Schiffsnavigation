# evaluate_policy.py

import sys
import os

# Projektstruktur für Import anpassen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Drittanbieter
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Lokale Module
from config import ENV_MODE, EPISODES, MAX_STEPS, LOOP_THRESHOLD, REWARDS
from navigation.environment.grid_environment import GridEnvironment
from navigation.environment.container_environment import ContainerShipEnv

# Umgebung initialisieren
env = ContainerShipEnv() if ENV_MODE == "container" else GridEnvironment(mode=ENV_MODE)
grid_size = env.grid_size

# Q-Tabelle laden
Q = np.load("q_table.npy")
print("Q-Tabelle geladen: q_table.npy")

# Zustandscodierung je nach Umgebung
def obs_to_state(obs):
    if ENV_MODE == "container":
        return obs[0] * env.grid_size + obs[1] + (env.grid_size * env.grid_size) * obs[2]
    return obs

results_cause = defaultdict(int)
results_solved = defaultdict(int)
rewards_all = []

for ep in range(EPISODES):
    obs, _ = env.reset()
    state = obs_to_state(obs)
    episode_reward = 0
    visited_states = {}
    steps = 0
    success = False
    cause = "Timeout"
    goal_reached = False
    loop_detected = False

    # DEBUG: Erste Episode detailliert loggen
    if ep == 0:
        print(f"=== EPISODE 0 DEBUG ===")
        print(f"Start State: {state}, Start Obs: {obs}")
        if hasattr(env, 'agent_pos'):
            print(f"Agent Position: {env.agent_pos}")
        if hasattr(env, 'goal_pos'):
            print(f"Goal Position: {env.goal_pos}")
        elif ENV_MODE == "container":
            print(f"Pickup Position: {env.pickup_pos}")
            print(f"Dropoff Position: {env.dropoff_pos}")
            print(f"Container Loaded: {env.container_loaded}")
            print(f"Obstacles: {env.obstacles}")

    for step in range(MAX_STEPS):
        action = np.argmax(Q[state])
        obs, reward, terminated, _, _ = env.step(action)
        next_state = obs_to_state(obs)
        episode_reward += reward
        steps += 1

        # Schleifenerkennung: Nur wenn die Umgebung selbst keine hat
        if ENV_MODE != "container":  # Grid-Environment hat keine eigene Schleifenerkennung
            visited_states[next_state] = visited_states.get(next_state, 0) + 1
            if visited_states[next_state] >= LOOP_THRESHOLD:
                print(f"[EP {ep}] Schleifenabbruch bei state {next_state} (besucht {visited_states[next_state]}x)")
                cause = "Schleifenabbruch"
                loop_detected = True
                episode_reward += REWARDS["loop_abort"]
                break

        # Erfolg erkennen basierend auf dem aktuellen Reward
        if ENV_MODE == "container":
            if reward == REWARDS["dropoff"]:
                goal_reached = True
                cause = "Ziel erreicht"
                success = True
            elif reward == REWARDS["loop_abort"]:
                loop_detected = True
                cause = "Schleifenabbruch"
        else:  # Grid-Environment
            if reward == REWARDS["goal"]:
                goal_reached = True
                cause = "Ziel erreicht"
                success = True

        # OPTIONAL: Test-Modus - künstlich einige Episoden als fehlgeschlagen markieren
        # Um die Logik zu testen, können Sie diese Zeilen einkommentieren:
        # if ep % 10 == 0:  # Jede 10. Episode als "Timeout" markieren
        #     cause = "Timeout"
        #     goal_reached = False
        #     success = False
        #     break

        state = next_state

        if terminated:
            print(f"[EP {ep}] terminated=True, final_reward={reward}, total_reward={episode_reward}, steps={steps}")

            # Falls noch nicht klassifiziert, basierend auf letztem Reward klassifizieren
            if not goal_reached and not loop_detected:
                if ENV_MODE == "container":
                    if reward == REWARDS["obstacle"]:
                        cause = "Hindernis-Kollision"
                    elif reward == REWARDS["timeout"]:
                        cause = "Timeout"
                else:  # Grid-Environment
                    if reward == REWARDS["obstacle"] or (reward < 0 and reward != REWARDS["step"]):
                        # Negative Rewards außer normalem Schritt-Reward
                        if reward == REWARDS["loop_abort"] or reward == (REWARDS["loop_abort"] + REWARDS["step"]):
                            cause = "Schleifenabbruch"
                        elif reward == REWARDS["timeout"] or reward == (REWARDS["timeout"] + REWARDS["step"]):
                            cause = "Timeout"
                        else:
                            cause = "Hindernis-Kollision"
            break

    # Timeout-Check nur wenn Episode nicht bereits beendet
    if not terminated and steps >= MAX_STEPS:
        print(f"[EP {ep}] Timeout erreicht nach {steps} Schritten, total_reward={episode_reward}")
        cause = "Timeout"
        episode_reward += REWARDS["timeout"]

    # Alternative Klassifizierung basierend auf Gesamt-Reward (falls Environment-spezifische Erkennung fehlschlägt)
    if cause == "Timeout" and episode_reward > 0:
        # Wenn positiver Gesamt-Reward, aber als Timeout klassifiziert -> wahrscheinlich Erfolg
        if ENV_MODE != "container" and episode_reward >= REWARDS["goal"]:
            cause = "Ziel erreicht"
            success = True

    # Erfolg final bestimmen: Nur wenn Ziel erreicht wurde
    success = (cause == "Ziel erreicht")

    results_cause[cause] += 1

    # Korrekte Zuordnung zu solved/failed
    if success:
        results_solved["solved episode"] += 1
    else:
        results_solved["failed episode"] += 1

    rewards_all.append(episode_reward)

    # Debug für erste paar Episoden
    if ep < 5:
        print(f"[EP {ep}] Endergebnis: {cause}, Total Reward: {episode_reward}, Steps: {steps}, Success: {success}")

# Zusätzliche Verifikation: Zeige Statistiken über alle Ursachen
print(f"\nDetaillierte Ursachen-Verteilung:")
total_episodes = sum(results_cause.values())
for cause, count in sorted(results_cause.items()):
    percentage = (count / total_episodes) * 100
    print(f"  {cause}: {count} ({percentage:.1f}%)")

# Sanity Check: Anzahl sollte EPISODES entsprechen
print(f"\nSanity Check: {total_episodes} von {EPISODES} Episoden erfasst")

avg_reward = np.mean(rewards_all)

print(f"\nAuswertung ({EPISODES} Episoden, Modus: {ENV_MODE}):")
for k, v in results_cause.items():
    print(f"{v}/{EPISODES} Episoden ({(v / EPISODES) * 100:.1f}%): {k}")

print(f"\nLösungsrate:")
solved_count = results_solved["solved episode"]
failed_count = results_solved["failed episode"]
print(f"{solved_count}/{EPISODES} Episoden erfolgreich ({(solved_count / EPISODES) * 100:.1f}%)")
print(f"{failed_count}/{EPISODES} Episoden fehlgeschlagen ({(failed_count / EPISODES) * 100:.1f}%)")

print(f"\nDurchschnittlicher Reward: {avg_reward:.2f}")

# Zusätzliche Statistiken
print(f"\nReward-Statistiken:")
print(f"Min Reward: {min(rewards_all):.2f}")
print(f"Max Reward: {max(rewards_all):.2f}")
print(f"Median Reward: {np.median(rewards_all):.2f}")

# Diagramm: Solved vs Failed
plt.figure(figsize=(8, 5))

# Farben explizit zuordnen basierend auf Label
colors = []
labels = list(results_solved.keys())
for label in labels:
    if "solved" in label.lower() or "success" in label.lower():
        colors.append("green")
    else:  # failed/failure
        colors.append("red")

bars = plt.bar(results_solved.keys(), results_solved.values(), color=colors)
plt.title(f"Lösungsrate ({ENV_MODE}-Modus)")
plt.xlabel("Ergebnis")
plt.ylabel("Anzahl Episoden")
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.5, f'{int(height)}', ha='center', va='bottom')
plt.tight_layout()
plt.show()

# Reward-Histogramm
plt.figure(figsize=(10, 6))
plt.hist(rewards_all, bins=20, edgecolor='black', alpha=0.7)
plt.title(f"Verteilung der Episoden-Rewards ({ENV_MODE}-Modus)")
plt.xlabel("Kumulativer Episode-Reward")
plt.ylabel("Häufigkeit")
plt.axvline(avg_reward, color='red', linestyle='--', label=f'Durchschnitt: {avg_reward:.2f}')
plt.legend()
plt.tight_layout()
plt.show()