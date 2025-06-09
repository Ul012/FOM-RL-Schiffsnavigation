# Funktionsweise des Systems

## Q-Learning Algorithmus

Das System implementiert klassisches Q-Learning mit der Update-Regel:

```
Q(s,a) ← Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
```

Wobei:
- `α` (ALPHA) = Lernrate (aus config.py)
- `γ` (GAMMA) = Diskontierungsfaktor (aus config.py)
- `r` = Sofortiger Reward
- `s'` = Folgezustand

## Entscheidungslogik

Der Agent wählt in jedem Zustand eine Aktion basierend auf der Q-Tabelle. Dabei wird eine **Epsilon-Greedy-Strategie** verwendet:

- **Exploration** (ε): Mit Wahrscheinlichkeit `EPSILON` wird eine zufällige Aktion gewählt
- **Exploitation** (1-ε): Sonst die beste bekannte Aktion aus der Q-Tabelle

```python
# Aus utils/qlearning.py
def select_action(Q, state, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)  # Exploration
    else:
        return np.argmax(Q[state])          # Exploitation
```

## Zustandsrepräsentation

### Grid-Umgebungen (Static, Random Modes)
- **Zustandsraum**: 25 Zustände (5x5 Gitter)
- **Codierung**: Einzelner Integer-Wert (0-24)
- **Formel**: `state = row * GRID_SIZE + col`

### Container-Umgebung
- **Zustandsraum**: 50 Zustände (25 Positionen × 2 Container-States)
- **Codierung**: Position + Container-Status
- **Formel**: `state = x * grid_size + y + (grid_size * grid_size) * container_loaded`

```python
# Aus utils/common.py
def obs_to_state(obs, env_mode, grid_size=None):
    if env_mode == "container":
        return obs[0] * grid_size + obs[1] + (grid_size * grid_size) * obs[2]
    return obs
```

## Belohnungsstruktur

**Alle Reward-Werte sind in `config.py` unter `REWARDS` konfigurierbar:**

| Ereignis | Grid-Umgebungen | Container-Umgebung | Beschreibung |
|----------|----------------|-------------------|--------------|
| **Ziel erreicht** | REWARDS["goal"] | - | Normale Zielerreichung |
| **Container aufgenommen** | - | REWARDS["pickup"] | Pickup erfolgreich |
| **Container abgeliefert** | - | REWARDS["dropoff"] | Dropoff erfolgreich (Hauptziel) |
| **Normaler Schritt** | REWARDS["step"] | REWARDS["step"] | Bewegungskosten (typisch negativ) |
| **Hinderniskollision** | REWARDS["obstacle"] | REWARDS["obstacle"] | Kollision mit Hindernis |
| **Schleifenabbruch** | REWARDS["loop_abort"] | REWARDS["loop_abort"] | Zyklische Bewegungen |
| **Timeout** | REWARDS["timeout"] | REWARDS["timeout"] | Maximale Schritte erreicht |

### Typische Reward-Struktur
- **Positive Belohnungen**: Zielerreichung, Pickup/Dropoff
- **Negative Belohnungen**: Bewegungskosten, Hindernisse, Abbrüche
- **Skalierung**: Ziel-Rewards deutlich höher als Bewegungskosten

## Terminierungsbedingungen

Das System erkennt verschiedene Episode-Enden in dieser **Prioritätsreihenfolge**:

### 1. Erfolgreiche Terminierung
- **Grid-Umgebungen**: Ziel erreicht (REWARDS["goal"])
- **Container-Umgebung**: Container erfolgreich abgeliefert (REWARDS["dropoff"])

### 2. Schleifenerkennung
- **Bedingung**: Zustand wird öfter als `LOOP_THRESHOLD` besucht
- **Konfiguration**: Über `config.py` anpassbar
- **Implementierung**: Tracking in `visited_states` Dictionary

### 3. Hinderniskollision
- **Bedingung**: Agent versucht in Hindernis-Feld zu bewegen
- **Verhalten**: Sofortige Terminierung mit negativem Reward

### 4. Timeout
- **Bedingung**: Maximale Schrittanzahl (`MAX_STEPS`) erreicht
- **Konfiguration**: Szenario-abhängig über `config.py`

```python
# Aus utils/evaluation.py
def check_loop_detection(visited_states, next_state, env_mode):
    if env_mode != "container":
        visited_states[next_state] = visited_states.get(next_state, 0) + 1
        if visited_states[next_state] >= LOOP_THRESHOLD:
            return True
    return False
```

## Szenario-Modi

**Über `config.py` werden verschiedene Umgebungsszenarien gesteuert:**

| Modus | ENV_MODE | Beschreibung | Komplexität |
|-------|----------|--------------|-------------|
| **Statisch** | `"static"` | Feste Positionen für alle Elemente | Niedrig |
| **Zufälliger Start** | `"random_start"` | Variable Startposition, festes Ziel | Mittel |
| **Zufälliges Ziel** | `"random_goal"` | Fester Start, variables Ziel | Mittel |
| **Zufällige Hindernisse** | `"random_obstacles"` | Variable Hindernispositionen | Hoch |
| **Container** | `"container"` | Pickup/Dropoff mit erweitertem Zustandsraum | Sehr hoch |

### Umgebungs-Initialisierung
```python
# Aus utils/environment.py
def initialize_environment(env_mode):
    env = ContainerShipEnv() if env_mode == "container" else GridEnvironment(mode=env_mode)
    return env, env.grid_size
```

## Q-Tabellen-Verwaltung

### Automatische Dateibenennung
Jedes Szenario erhält eine eigene Q-Tabelle basierend auf `ENV_MODE`:
- `q_table_static.npy`
- `q_table_random_start.npy`
- `q_table_random_goal.npy`
- `q_table_random_obstacles.npy`
- `q_table_container.npy`

### Automatische Verwaltung
```python
# Aus utils/qlearning.py
def save_q_table(Q, env_mode):
    filepath = f"q_table_{env_mode}.npy"
    np.save(filepath, Q)

def load_q_table(env_mode):
    filepath = f"q_table_{env_mode}.npy"
    return np.load(filepath)
```

## Modulare Systemarchitektur

Das System ist in wiederverwendbare Module aufgeteilt:

### Utils-Module
```python
utils/
├── common.py           # Basis-Funktionen (Seeds, State-Conversion)
├── qlearning.py        # Q-Learning Algorithmus
├── environment.py      # Umgebungs-Management
├── evaluation.py       # Bewertungslogik
├── position.py         # Position/State Konvertierungen
├── visualization.py    # Plotting-Funktionen
└── reporting.py        # Ausgabe-Funktionen
```

### Environment-Module
```python
envs/
├── grid_environment.py      # Standard Grid-Umgebung
└── container_environment.py # Container Pickup/Dropoff
```

## Datenfluss

### Training-Pipeline
1. **Initialisierung**: Umgebung und Q-Tabelle erstellen
2. **Episode-Loop**: Für jede Trainings-Episode
   - Environment reset
   - Schritt-Loop bis Terminierung
   - Q-Wert Updates nach jedem Schritt
3. **Tracking**: Rewards, Erfolge, Schritte sammeln
4. **Speicherung**: Q-Tabelle und Statistiken sichern
5. **Visualisierung**: Lernkurven und Metriken darstellen

### Evaluation-Pipeline
1. **Laden**: Gespeicherte Q-Tabelle importieren
2. **Greedy-Policy**: Nur Exploitation (ε = 0)
3. **Statistik-Sammlung**: Erfolgsraten, Terminierungsarten
4. **Analyse**: Quantitative Bewertung der Policy-Qualität

## Reproduzierbarkeit

### Seed-Management
```python
# Aus utils/common.py
def set_all_seeds(seed=None):
    random.seed(seed)
    np.random.seed(seed)
    # Umgebungs-spezifische Seeds werden automatisch gesetzt
```

### Deterministische Ergebnisse
Bei gleichem Seed und gleichen Hyperparametern sind alle Ergebnisse vollständig reproduzierbar:
- Umgebungs-Zufälligkeit (Start/Ziel/Hindernisse)
- Agent-Entscheidungen (Epsilon-Greedy)
- Q-Tabellen-Initialisierung

## Performance-Charakteristika

### Konvergenz-Eigenschaften
**Abhängig von Parametern in `config.py`:**
- **Static**: Schnellste Konvergenz
- **Random Modes**: Mittlere Konvergenz-Geschwindigkeit
- **Container**: Langsamste Konvergenz (komplexester Zustandsraum)

### Speicher-Effizienz
- **Grid Q-Tabelle**: 25 × 4 Werte
- **Container Q-Tabelle**: 50 × 4 Werte
- **Minimal Memory Footprint**: Unter 1KB pro Q-Tabelle

### Skalierbarkeit
Das modulare Design ermöglicht:
- **Einfache Erweiterung** um neue Szenarien
- **Parameter-Anpassung** ohne Code-Änderungen
- **Wiederverwendung** in anderen RL-Projekten

**Hinweis**: Alle konkreten Parameter-Werte und Rewards sind in `config.py` konfiguriert und können dort eingesehen und angepasst werden.