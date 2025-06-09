# 🚢 Q-Learning für Schiffsnavigation

Dieses Projekt implementiert Q-Learning zur autonomen Navigation eines Schiffes durch eine 5x5-Gitterwelt. Das System umfasst verschiedene Umgebungsmodi, automatisiertes Multi-Szenario-Training und umfassende Evaluationstools.

## 📋 Projektübersicht

- **Q-Learning-Implementierung** mit automatisiertem Training
- **Gymnasium-kompatible Umgebungen** für verschiedene Navigationsszenarien
- **Multi-Szenario-Training** mit `train_all_scenarios.py`
- **Szenarien-Vergleich** mit statistischer Analyse
- **Robuste Terminierungserkennung** (Erfolg, Timeout, Schleifen, Hindernisse)
- **Professionelle Visualisierung** und PDF-Export
- **Modulare Code-Architektur** mit wiederverwendbaren Utils
- **Umfassende Dokumentation** mit MkDocs

## ⚙️ Installation

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

## 🚀 Verwendung

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

### Q-Tabellen inspizieren
```bash
python inspect_q_tables.py
```

## 🗺️ Verfügbare Szenarien

| Szenario | Beschreibung | Komplexität | Emojis |
|----------|--------------|-------------|---------|
| `static` | Feste Positionen für Start, Ziel und Hindernisse | Niedrig | 🧭🏁🪨 |
| `random_start` | Zufällige Startposition bei festem Ziel | Mittel | 🚢🏁🪨 |
| `random_goal` | Feste Startposition mit zufälligem Ziel | Mittel | 🧭🏁🪨 |
| `random_obstacles` | Variable Hindernispositionen pro Episode | Hoch | 🧭🏁🪨 |
| `container` | Pickup/Dropoff-Aufgabe mit erweiterten Zuständen | Sehr hoch | 🚢📦🏁 |

## ⚙️ Konfiguration

Alle Parameter werden zentral in `config.py` verwaltet:

```python
ENV_MODE    # Szenario-Auswahl
EPISODES    # Trainings-Episoden
MAX_STEPS   # Maximale Schritte pro Episode
ALPHA       # Lernrate
GAMMA       # Diskontierungsfaktor
EPSILON     # Explorationsrate
```

## 📁 Projektstruktur

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

## 🎮 Visualisierung

Das System verwendet intuitive Emojis für die visuelle Darstellung:

### Grid-Umgebungen
- 🚢 **Agent/Schiff** - Aktuelle Position
- 🧭 **Start** - Startposition (bei festen Starts)
- 🏁 **Ziel** - Zielposition
- 🪨 **Hindernis** - Nicht passierbare Felder
- ↑→↓← **Policy-Pfeile** - Optimale Aktionen der gelernten Policy

### Container-Umgebung
- 🚢 **Schiff** - Agent-Position
- 📦 **Pickup** - Container-Abholposition
- 🏁 **Dropoff** - Container-Abgabeposition (Ziel)
- 🪨 **Hindernis** - Nicht passierbare Felder

## 📊 Wissenschaftliche Evaluierung

Analysemöglichkeiten:

- **Erfolgsraten-Vergleich** zwischen Szenarien
- **Terminierungsarten-Analyse** (Erfolg, Timeout, Schleifen, Hindernisse)
- **Lernkurven-Visualisierung** mit statistischen Metriken
- **Parameter-Logging** für Reproduzierbarkeit
- **Diagramme** mit PDF-Export

## 🏗️ Code-Architektur

Das Projekt folgt modernen Software-Engineering-Prinzipien:

- **DRY-Prinzip**: Keine Code-Duplikation durch Utils-Module
- **Modulare Struktur**: Klare Trennung von Verantwortlichkeiten
- **Clean Code**: Lesbare und wartbare Implementierung
- **Wiederverwendbarkeit**: Utils können in anderen RL-Projekten genutzt werden

## 📚 Dokumentation

Lokale Dokumentation starten:
```bash
mkdocs serve
```

Verfügbar unter: http://127.0.0.1:8000

## 🔧 Technische Details

- **Framework**: OpenAI Gymnasium
- **Algorithmus**: Q-Learning mit Epsilon-Greedy-Exploration
- **Visualisierung**: Matplotlib mit PDF-Export, Pygame für interaktive Darstellung
- **Dokumentation**: MkDocs mit Material Theme
- **Code-Struktur**: Modulare Python-Pakete mit Utils-Bibliothek