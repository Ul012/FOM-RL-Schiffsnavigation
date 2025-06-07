# config.py

# ============================================================================
# Umgebungskonfiguration
# ============================================================================

ENV_MODE = "random_goal"  # Optionen: static, random_start, random_goal, random_obstacles, container
GRID_SIZE = 5
ACTIONS = 4  # 0 = oben, 1 = rechts, 2 = unten, 3 = links

# ============================================================================
# Rewardsystem
# ============================================================================

REWARDS = {
    "step": -1,
    "goal": 10,
    "obstacle": -10,
    "loop_abort": -10,
    "timeout": -10,
    "pickup": 8,
    "dropoff": 20
}

# ============================================================================
# Q-Learning Parameter
# ============================================================================

ALPHA = 0.1  # Lernrate
GAMMA = 0.9  # Diskontierungsfaktor
EPSILON = 0.1  # Explorationsrate
EPISODES = 500  # Trainings-Episoden

# ============================================================================
# Training Parameter
# ============================================================================

MAX_STEPS = 100  # Max. Schritte pro Episode (initial: 50)
LOOP_THRESHOLD = 15  # Schleifenwiederholungen für Abbruch (initial: 6)

# ============================================================================
# Evaluation Parameter
# ============================================================================

NUM_TEST_ENVS = 100  # Anzahl Testumgebungen
EVAL_EPISODES = 100  # Anzahl Episoden für Evaluation
EVAL_MAX_STEPS = 100  # Max. Schritte pro Episode in Evaluation

# ============================================================================
# Dateipfade
# ============================================================================

# Q-Tabelle Pfad basierend auf Modus
def get_q_table_path(env_mode):
    return f"q_table_{env_mode}.npy"

Q_TABLE_PATH = get_q_table_path(ENV_MODE)

RESULTS_PATH = "results/"
PLOTS_PATH = "plots/"

# ============================================================================
# Visualisierung Parameter
# ============================================================================

SHOW_VALUES = True  # Q-Werte in Heatmaps anzeigen
COLORMAP_STYLE = "viridis"  # Colormap für Heatmaps
VISUALIZATION_DELAY = 0.5  # Verzögerung zwischen Schritten (Sekunden)

CELL_SIZE = 80  # Größe einer Grid-Zelle in Pixeln
FRAME_DELAY = 0.4  # Verzögerung zwischen Frames (Sekunden)
ARROW_SCALE = 0.3  # Größe der Policy-Pfeile
SHOW_GRID_LINES = True  # Grid-Linien in Visualisierungen

FIGURE_SIZE = (10, 6)  # Plot-Größe
DPI_SETTING = 100  # Auflösung für gespeicherte Plots

# ============================================================================
# Export Parameter
# ============================================================================

EXPORT_PDF = True  # PDF-Export für Visualisierungen
EXPORT_PATH = "exports/"  # Pfad für exportierte Dateien

# ============================================================================
# Debug Parameter
# ============================================================================

DEBUG_MODE = False  # Debug-Ausgaben aktivieren
VERBOSE_TRAINING = True  # Detaillierte Training-Ausgaben