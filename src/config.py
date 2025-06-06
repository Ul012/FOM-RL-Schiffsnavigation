# config.py

# Umgebungskonfiguration
ENV_MODE = "container"  # Optionen: static, random_start, random_goal, random_obstacles, container
GRID_SIZE = 5              # Für grid-basierte Umgebung
ACTIONS = 4                # 0 = oben, 1 = rechts, 2 = unten, 3 = links

# Rewardsystem
REWARDS = {
    "step": -1,
    "goal": 10,
    "obstacle": -10,
    "loop_abort": -10,
    "timeout": -10,
    "pickup": 8,
    "dropoff": 20
}
# Q-Learning Parameter
ALPHA = 0.1       # Lernrate
GAMMA = 0.9       # Diskontierungsfaktor
EPSILON = 0.1     # Explorationsrate
EPISODES = 500    # Trainings-Episoden
ACTIONS = 4       # Anzahl möglicher Aktionen (0 = hoch, 1 = rechts, 2 = runter, 3 = links)

# Evaluation
MAX_STEPS = 50              # Max. Schritte pro Episode in evaluate_policy
LOOP_THRESHOLD = 6          # Schleifenwiederholungen für Abbruch
NUM_TEST_ENVS = 100         # Anzahl Testumgebungen bei evaluate_policy
