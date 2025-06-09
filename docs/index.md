# 🚢 Q-Learning für Schiffsnavigation

Dieses Projekt implementiert einen Q-Learning-Algorithmus zur autonomen Navigation eines Agenten durch verschiedene Gitterumgebungen. Das System ermöglicht das Training und die Evaluation von Navigationsstrategien unter verschiedenen Umgebungsbedingungen mit einer modularen Code-Architektur.

## 🎯 Projektziele

- Entwicklung eines robusten Q-Learning-Agenten für Navigationsprobleme
- Implementierung verschiedener Umgebungsszenarien mit unterschiedlichen Komplexitätsgraden
- Bereitstellung von Evaluations- und Vergleichstools für wissenschaftliche Analyse
- Modulare Code-Architektur nach Clean Code Prinzipien
- Dokumentation der Implementierung und Ergebnisse

## 🏗️ Systemarchitektur

Das Projekt besteht aus mehreren Komponenten:

- **Training**: Automatisiertes Lernen für einzelne oder multiple Szenarien
- **Evaluation**: Quantitative Analyse der gelernten Policies
- **Vergleich**: Statistische Auswertung verschiedener Szenarien
- **Visualisierung**: Grafische Darstellung der Agentenverhalten und Lernfortschritte
- **Utils-Module**: Wiederverwendbare Komponenten für Q-Learning Projekte

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

## 🗺️ Verfügbare Umgebungsszenarien

| Szenario | Beschreibung | Anwendungsbereich | Emojis |
|----------|--------------|-------------------|---------|
| **Static** | Konstante Positionen für alle Elemente | Grundlegendes Q-Learning | 🧭🏁🪨 |
| **Random Start** | Variable Startpositionen | Robustheitstesting | 🚢🏁🪨 |
| **Random Goal** | Variable Zielpositionen | Adaptive Navigation | 🧭🏁🪨 |
| **Random Obstacles** | Variable Hindernisverteilungen | Dynamische Umgebungen | 🧭🏁🪨 |
| **Container** | Pickup/Dropoff-Aufgaben | Komplexe Aufgabenstellungen | 🚢📦🏁 |

## ⚙️ Technische Spezifikationen

- **Umgebung**: 5x5 Gitterwelt (OpenAI Gymnasium-kompatibel)
- **Algorithmus**: Q-Learning mit Epsilon-Greedy-Exploration
- **Zustandsraum**: Diskret (25 Zustände für Grid, erweitert für Container)
- **Aktionsraum**: 4 Bewegungsrichtionen (Oben, Rechts, Unten, Links)
- **Terminierungsbedingungen**: Zielerreichung, Timeout, Schleifenerkennung, Hinderniskollision
- **Visualisierung**: Matplotlib mit PDF-Export, Pygame für interaktive Darstellung
- **Code-Struktur**: Modulare Python-Pakete mit Utils-Bibliothek

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

Das System bietet umfassende Analysemöglichkeiten:

- **Erfolgsraten-Vergleich** zwischen verschiedenen Szenarien
- **Statistische Auswertung** von Terminierungsarten
- **Lernkurven-Analyse** mit Moving-Average-Glättung
- **Parameter-Sensitivitätsanalyse** durch zentrale Konfiguration
- **Reproduzierbare Experimente** mit Seed-Management
- **Diagramme** mit automatischem PDF-Export

## 🏗️ Code-Architektur

Das Projekt folgt modernen Software-Engineering-Prinzipien:

- **DRY-Prinzip**: Keine Code-Duplikation durch Utils-Module
- **Modulare Struktur**: Klare Trennung von Verantwortlichkeiten
- **Clean Code**: Lesbare und wartbare Implementierung
- **Wiederverwendbarkeit**: Utils können in anderen RL-Projekten genutzt werden
- **Skalierbarkeit**: Einfache Erweiterung um neue Szenarien und Algorithmen

## ⚙️ Installation und Setup

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

### Erste Schritte
```bash
cd src
python train.py                   # Training starten
python visualize_policy.py        # Ergebnisse visualisieren
```

---

**📚 Weiterführende Informationen:**

- [Funktionsweise des Systems](funktionsweise.md)
- [Training und Konfiguration](training.md)
- [Visualisierung und Export](visualisierung.md)