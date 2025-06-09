# Installation und Ausführung

## Systemvoraussetzungen

- Python 3.8 oder höher
- Mindestens 2 GB verfügbarer Speicherplatz
- Empfohlen: Anaconda oder Miniconda für das Paketmanagement

## Installation

### 1. Repository klonen
```bash
git clone [repository-url]
cd ship-navigation-rl
```

### 2. Virtuelle Umgebung erstellen
```bash
python -m venv rl-venv
```

### 3. Virtuelle Umgebung aktivieren
```bash
# Windows
rl-venv\Scripts\activate

# macOS/Linux
source rl-venv/bin/activate
```

### 4. Abhängigkeiten installieren
```bash
pip install -r requirements.txt
```

### 5. Installation verifizieren
```bash
cd src
python -c "import gymnasium, numpy, matplotlib, pygame; print('Installation erfolgreich')"
```

## Projektstruktur

Nach der Installation haben Sie folgende Struktur:

```
ship-navigation-rl/
├── src/
│   ├── train.py                    # Einzelszenario-Training
│   ├── train_all_scenarios.py      # Multi-Szenario-Training
│   ├── compare_scenarios.py        # Szenarien-Vergleich
│   ├── evaluate_policy.py          # Policy-Evaluation
│   ├── visualize_policy.py         # Visuelle Darstellung
│   ├── inspect_q_tables.py         # Q-Tabellen-Analyse
│   ├── config.py                   # Zentrale Konfiguration
│   ├── envs/                       # Umgebungs-Implementierungen
│   │   ├── __init__.py
│   │   ├── grid_environment.py     # Grid-Umgebung
│   │   └── container_environment.py # Container-Umgebung
│   └── utils/                      # Wiederverwendbare Module
│       ├── __init__.py
│       ├── common.py              # Basis-Hilfsfunktionen
│       ├── environment.py         # Umgebungs-Initialisierung
│       ├── qlearning.py           # Q-Learning Algorithmus
│       ├── evaluation.py          # Bewertungslogik
│       ├── position.py            # Position/State Konvertierungen
│       ├── visualization.py       # Plotting-Funktionen
│       └── reporting.py           # Ausgabe-Funktionen
├── exports/                        # Generierte Visualisierungen
├── docs/                          # MkDocs Dokumentation
├── requirements.txt
└── README.md
```

## Grundlegende Verwendung

### Einzelszenario-Training
```bash
cd src
python train.py
```

Das Training verwendet die Konfiguration aus `config.py`. Nach Abschluss wird eine Q-Tabelle gespeichert und Lerndiagramme angezeigt.

### Multi-Szenario-Training
```bash
python train_all_scenarios.py
```

Trainiert alle verfügbaren Szenarien automatisch. Bietet Optionen für:
- Automatisierte oder interaktive Visualisierung
- Sequenzielles oder paralleles Training
- Szenario-Auswahl

### Szenarien-Vergleich
```bash
python compare_scenarios.py
```

Führt eine statistische Analyse aller trainierten Szenarien durch und erstellt Vergleichsdiagramme.

### Policy-Evaluation
```bash
python evaluate_policy.py
```

Evaluiert eine trainierte Policy ohne weitere Lernschritte.

### Visuelle Darstellung
```bash
python visualize_policy.py
```

Zeigt die gelernte Policy in einer animierten Pygame-Darstellung mit Emojis:
- 🚢 Agent/Schiff
- 🧭 Start (bei Grid-Umgebungen)
- 📦 Pickup (bei Container-Umgebung)
- 🏁 Ziel/Dropoff
- 🪨 Hindernisse
- ↑→↓← Policy-Pfeile

### Q-Tabellen-Inspektion
```bash
python inspect_q_tables.py
```

Analysiert und vergleicht Q-Tabellen verschiedener Szenarien mit interaktiven Optionen.

## Konfiguration

### Zentrale Parameter in config.py
```python
ENV_MODE                     # Szenario-Auswahl ("static", "container", etc.)
EPISODES                     # Anzahl Trainings-Episoden
MAX_STEPS                    # Maximale Schritte pro Episode
ALPHA                        # Lernrate (typisch: 0.05-0.2)
GAMMA                        # Diskontierungsfaktor (typisch: 0.9-0.99)
EPSILON                      # Explorationsrate (typisch: 0.05-0.3)
```

### Evaluations-Parameter
```python
EVAL_EPISODES                # Episoden für Szenario-Vergleich
EVAL_MAX_STEPS               # Maximale Schritte bei Evaluation
LOOP_THRESHOLD               # Schwellwert für Schleifenerkennung
```

### Export-Einstellungen
```python
EXPORT_PDF                   # PDF-Export aktivieren (True/False)
EXPORT_PATH                  # Zielordner für Exports
```

### Visualisierungs-Parameter
```python
CELL_SIZE                    # Größe einer Grid-Zelle in Pixeln
FRAME_DELAY                  # Verzögerung zwischen Frames (Sekunden)
SHOW_VISUALIZATIONS          # Interaktive Diagramme anzeigen (True/False)
```

## Verfügbare Szenarien

Das System unterstützt folgende Umgebungsszenarien:

| Szenario | ENV_MODE | Beschreibung |
|----------|----------|--------------|
| Statisch | `"static"` | Feste Positionen (🧭🏁🪨) |
| Zufälliger Start | `"random_start"` | Variable Startposition (🚢🏁🪨) |
| Zufälliges Ziel | `"random_goal"` | Variable Zielposition (🧭🏁🪨) |
| Zufällige Hindernisse | `"random_obstacles"` | Variable Hindernisse (🧭🏁🪨) |
| Container | `"container"` | Pickup/Dropoff-Aufgabe (🚢📦🏁) |

## Fehlerbehebung

### Häufige Probleme

**ModuleNotFoundError bei Imports:**
```bash
# Utils-Module nicht gefunden
cd src
python -c "from utils import set_all_seeds; print('Utils OK')"

# Environment-Module nicht gefunden  
python -c "from envs import GridEnvironment; print('Envs OK')"
```

**Pygame-Fenster nicht sichtbar:**
```bash
# Pygame-Installation überprüfen
pip install --upgrade pygame
python -c "import pygame; pygame.init(); print('Pygame OK')"
```

**Matplotlib-Darstellungsfehler:**
```bash
pip install --upgrade matplotlib
```

**Q-Tabelle nicht gefunden:**
```bash
# Erst Training durchführen
python train.py
# Dann Visualisierung
python visualize_policy.py
```

### Cache-Probleme beheben
```bash
# Python-Cache löschen
find . -name "*.pyc" -delete
find . -name "__pycache__" -delete
```

### Abhängigkeiten aktualisieren
```bash
pip install --upgrade -r requirements.txt
```

### Neue Abhängigkeiten erfassen
```bash
pip freeze > requirements.txt
```

## Import-System

Das Projekt nutzt eine modulare Import-Struktur:

```python
# Utils-Module importieren
from utils import set_all_seeds, load_q_table
from utils.visualization import create_learning_curve

# Environment-Module importieren  
from envs import GridEnvironment, ContainerShipEnv
```

Bei Import-Problemen prüfen Sie:
1. Korrekte Ordnerstruktur (`envs/` und `utils/` vorhanden)
2. `__init__.py` Dateien in allen Package-Ordnern
3. Aktueller Pfad (`cd src` vor Ausführung)

## Dokumentation

### Lokale Dokumentation starten
```bash
mkdocs serve
```

Die Dokumentation ist dann unter http://127.0.0.1:8000 verfügbar.

### Dokumentation erstellen
```bash
mkdocs build
```

Erstellt statische HTML-Dateien im `site/`-Verzeichnis.