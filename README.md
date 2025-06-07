# Q-Learning Navigation für Schiffssteuerung

Dieses Projekt implementiert Q-Learning zur autonomen Navigation eines Schiffes durch eine 5x5-Gitterwelt. Das System umfasst verschiedene Umgebungsmodi, automatisiertes Multi-Szenario-Training und umfassende Evaluationstools.

## Projektübersicht

- **Q-Learning-Implementierung** mit automatisiertem Training
- **Gymnasium-kompatible Umgebungen** für verschiedene Navigationsszenarien
- **Multi-Szenario-Training** mit `train_all_scenarios.py`
- **Szenarien-Vergleich** mit statistischer Analyse
- **Robuste Terminierungserkennung** (Erfolg, Timeout, Schleifen, Hindernisse)
- **Professionelle Visualisierung** und PDF-Export
- **Umfassende Dokumentation** mit MkDocs

## Installation

### Virtuelle Umgebung erstellen
```bash
python -m venv rl-venv
rl-venv\Scripts\activate          # Windows
source rl-venv/bin/activate       # macOS/Linux
```

### Abhängigkeiten installieren
```bash
pip install -r requirements.txt
```

## Verwendung

### Einzelnes Szenario trainieren
```bash
cd src
python train.py
```

### Alle Szenarien automatisch trainieren
```bash
python train_all_scenarios.py
```

### Szenarien vergleichen
```bash
python compare_scenarios.py
```

### Policy evaluieren
```bash
python evaluate_policy.py
```

### Visuell darstellen
```bash
python visualize_policy.py
```

## Verfügbare Szenarien

| Szenario | Beschreibung | Komplexität |
|----------|--------------|-------------|
| `static` | Feste Positionen für Start, Ziel und Hindernisse | Niedrig |
| `random_start` | Zufällige Startposition bei festem Ziel | Mittel |
| `random_goal` | Feste Startposition mit zufälligem Ziel | Mittel |
| `random_obstacles` | Variable Hindernispositionen pro Episode | Hoch |
| `container` | Pickup/Dropoff-Aufgabe mit erweiterten Zuständen | Sehr hoch |

## Konfiguration

Alle Parameter werden zentral in `config.py` verwaltet:

```python
ENV_MODE = "static"           # Szenario-Auswahl
EPISODES = 2000              # Trainings-Episoden
MAX_STEPS = 50               # Maximale Schritte pro Episode
ALPHA = 0.1                  # Lernrate
GAMMA = 0.9                  # Diskontierungsfaktor
EPSILON = 0.1                # Explorationsrate
```

## Projektstruktur

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
│   └── navigation/
│       └── environment/
│           ├── grid_environment.py      # Grid-Umgebung
│           └── container_environment.py # Container-Umgebung
├── exports/                        # Generierte Visualisierungen
├── docs/                          # MkDocs Dokumentation
├── requirements.txt
└── README.md
```

## Wissenschaftliche Evaluierung

Das Projekt bietet umfassende Analysemöglichkeiten:

- **Erfolgsraten-Vergleich** zwischen Szenarien
- **Terminierungsarten-Analyse** (Erfolg, Timeout, Schleifen, Hindernisse)
- **Lernkurven-Visualisierung** mit statistischen Metriken
- **Parameter-Logging** für Reproduzierbarkeit
- **Professionelle Diagramme** mit PDF-Export

## Dokumentation

Lokale Dokumentation starten:
```bash
mkdocs serve
```

Verfügbar unter: http://127.0.0.1:8000

## Technische Details

- **Framework**: OpenAI Gymnasium
- **Algorithmus**: Q-Learning mit Epsilon-Greedy-Exploration
- **Visualisierung**: Matplotlib mit PDF-Export
- **Dokumentation**: MkDocs mit Material Theme