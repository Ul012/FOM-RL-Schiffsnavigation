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

results = defaultdict(int)
rewards_all = []

for ep in range(EPISODES):      # Eine Schleife über die Anzahl der zu testenden Episoden
    obs, _ = env.reset()        # Setzt die Umgebung zurück. Der zweite Rückgabewert ( _ ) wird ignoriert.
    state = obs_to_state(obs)   # Wandelt die Beobachtung in einen diskreten Zustand (state) für die Q-Tabelle um
    episode_reward = 0          # Zählt die Gesamtbelohnung innerhalb der Episode
    visited_states = {}         # Ein Dictionary, das zählt, wie oft ein Zustand besucht wurde — zur Schleifenerkennung
    steps = 0                   # Schrittzähler innerhalb der aktuellen Episode
    success = False             # Flag, ob das Ziel erfolgreich erreicht wurde
    cause = "Timeout"

    for _ in range(MAX_STEPS):                              # Innere Schleife: Schritte innerhalb einer Episode. Führt die Aktion bis zum Abbruch aus (maximal MAX_STEPS Schritte pro Episode)
        action = np.argmax(Q[state])                        # Wähle die beste (maximal bewertete) Aktion aus der Q-Tabelle für den aktuellen Zustand.
        obs, reward, terminated, _, _ = env.step(action)    # Führe die Aktion in der Umgebung aus. Rückgabe: neue Beobachtung, Belohnung, Terminationsflag
        state = obs_to_state(obs)
        episode_reward += reward                            # Die Belohnung für den aktuellen Schritt wird zur Gesamtbelohnung addiert
        steps += 1                                          # Schrittzähler hochzählen

        visited_states[state] = visited_states.get(state, 0) + 1    # Schleifenerkennung (gegen endlose Wiederholungen)
        if visited_states[state] >= LOOP_THRESHOLD:                 # Wenn der Zustand zu oft besucht wurde, liegt vermutlich eine Schleife vor.
            print(f"[EP {ep}] Schleifenabbruch bei state {state}")  # Debugging
            cause = "Schleifenabbruch"                              # Ursache aktualisieren, Strafe für Schleife anwenden, Episode beenden
            episode_reward += REWARDS["loop_abort"]
            break

        if terminated:
            print(f"[EP {ep}] terminated=True, reward={reward}")  # Debugging
            if reward == REWARDS["goal"] or reward == REWARDS["dropoff"]:
                cause = "Ziel erreicht"
                success = True
            elif reward == REWARDS["obstacle"]:
                cause = "Hindernis-Kollision"
            break

    if not success and steps >= MAX_STEPS:
        print(f"[EP {ep}] Timeout erreicht nach {steps} Schritten")  # Debugging
        cause = "Timeout"

    results[cause] += 1
    rewards_all.append(episode_reward)

avg_reward = np.mean(rewards_all)

print(f"\nAuswertung ({EPISODES} Episoden, Modus: {ENV_MODE}):")
for k, v in results.items():
    print(f"{v}/{EPISODES} Episoden ({(v / EPISODES) * 100:.1f}%): {k}")
print(f"\nDurchschnittlicher Reward: {avg_reward:.2f}")

# Balkendiagramm mit Text
plt.figure(figsize=(8, 5))
bars = plt.bar(results.keys(), results.values(), color='steelblue')
plt.title(f"Ausgang der Episoden ({ENV_MODE}-Modus)")
plt.xlabel(f"Episoden-Endtyp (insg. {EPISODES} Episoden)")
plt.ylabel("Anzahl")
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.5, f'{int(height)}', ha='center', va='bottom')
plt.tight_layout()
plt.show()

# Reward-Histogramm
plt.figure(figsize=(8, 5))
plt.hist(rewards_all, bins=20, edgecolor='black')
plt.title(f"Verteilung der Episoden-Rewards ({ENV_MODE}-Modus)")
plt.xlabel("Reward")
plt.ylabel("Häufigkeit")
plt.tight_layout()
plt.show()
