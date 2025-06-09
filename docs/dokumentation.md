# Dokumentation und Entwicklung

## Lokale Dokumentation

```bash
# Entwicklungsmodus starten
mkdocs serve
# Verfügbar unter: http://127.0.0.1:8000

# Statische HTML-Dateien generieren  
mkdocs build

# Dokumentation mit Auto-Reload
mkdocs serve --dev-addr=127.0.0.1:8001
```

## Dokumentationsstruktur

```
docs/
├── index.md              # Projektübersicht und Ziele
├── setup.md              # Installation und Ausführung  
├── funktionsweise.md     # Systemfunktionalität und Algorithmus
├── training.md           # Training, Parameter und Konfiguration
├── visualisierung.md     # Analyse-Tools und Export-Funktionen
└── dokumentation.md      # Diese Entwicklungsdokumentation
```

### Inhaltliche Abdeckung

| Datei | Fokus | Zielgruppe |
|-------|-------|------------|
| **index.md** | Projektübersicht, Ziele, Architektur | Alle Nutzer |
| **setup.md** | Installation, Grundlagen, Troubleshooting | Neue Nutzer |
| **funktionsweise.md** | Q-Learning, Algorithmus, Module | Entwickler, Forscher |
| **training.md** | Parameter, Szenarien, Best Practices | Praktische Anwendung |
| **visualisierung.md** | Plots, Exports, Interpretation | Analyse und Ergebnisse |
| **dokumentation.md** | MkDocs, Entwicklung, Wartung | Entwickler |

## Projektarchitektur-Dokumentation

### Code-Struktur
```
ship-navigation-rl/
├── src/                           # Hauptcode
│   ├── *.py                       # Ausführbare Scripts
│   ├── envs/                      # Umgebungs-Implementierungen
│   │   ├── __init__.py           # Package-Initialisierung
│   │   ├── grid_environment.py    # Standard Grid-Umgebung
│   │   └── container_environment.py # Container Pickup/Dropoff
│   └── utils/                     # Wiederverwendbare Module
│       ├── __init__.py           # Utils Package mit Exports
│       ├── common.py             # Basis-Hilfsfunktionen
│       ├── environment.py        # Umgebungs-Management
│       ├── qlearning.py          # Q-Learning Algorithmus
│       ├── evaluation.py         # Bewertungslogik
│       ├── position.py           # Position/State Konvertierungen
│       ├── visualization.py      # Plotting-Funktionen
│       └── reporting.py          # Ausgabe-Funktionen
├── docs/                          # MkDocs Dokumentation
├── exports/                       # Generierte Visualisierungen
├── mkdocs.yml                     # Dokumentations-Konfiguration
├── requirements.txt               # Python-Abhängigkeiten
└── README.md                      # Projekt-Hauptdokumentation
```

### Modulare Architektur

Das System folgt modernen Software-Engineering-Prinzipien:

- **DRY-Prinzip**: Keine Code-Duplikation durch Utils-Module
- **Single Responsibility**: Jedes Modul hat eine klare Aufgabe
- **Clean Imports**: Strukturierte Package-Hierarchie
- **Wiederverwendbarkeit**: Utils können in anderen RL-Projekten genutzt werden

## GitHub Pages Deployment

```bash
# Automatisches Deployment zu GitHub Pages
mkdocs gh-deploy

# Mit spezifischem Branch
mkdocs gh-deploy --remote-branch gh-pages

# Mit benutzerdefinierter Commit-Message
mkdocs gh-deploy -m "Update documentation v2.0"
```

### Deployment-Konfiguration
```yaml
# mkdocs.yml
site_url: https://username.github.io/ship-navigation-rl/
repo_url: https://github.com/username/ship-navigation-rl
repo_name: ship-navigation-rl
```

## Toolchain-Details

### Abhängigkeiten
```bash
# Basis-Installation
pip install mkdocs>=1.5.0
pip install mkdocs-material>=9.0.0

# Erweiterte Features (optional)
pip install mkdocs-mermaid2-plugin    # Diagramme
pip install mkdocs-pdf-export-plugin  # PDF-Export
pip install mkdocs-git-revision-date-plugin  # Git-Integration
```

### Erweiterte mkdocs.yml Konfiguration
```yaml
site_name: Q-Learning Navigation
site_description: Q-Learning für autonome Schiffsnavigation
site_author: [Ihr Name]

theme:
  name: material
  palette:
    - scheme: default
      primary: blue
      accent: cyan
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: blue
      accent: cyan
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.top
    - toc.integrate
    - search.highlight
    - content.code.copy

nav:
  - 🏠 Startseite: index.md
  - ⚙️ Setup: setup.md
  - 🧠 Funktionsweise: funktionsweise.md
  - 🎯 Training: training.md
  - 📊 Visualisierung: visualisierung.md
  - 📚 Entwicklung: dokumentation.md

markdown_extensions:
  - admonition
  - codehilite
  - toc:
      permalink: true
  - pymdownx.emoji:
      emoji_index: !!python/name:materialx.emoji.twemoji
      emoji_generator: !!python/name:materialx.emoji.to_svg

plugins:
  - search
  - git-revision-date

extra:
  social:
    - icon: fontawesome/brands/github
      link: https://github.com/username/ship-navigation-rl
```

## Entwicklungsworkflow

### Dokumentation aktualisieren
1. **Lokale Bearbeitung**: Markdown-Dateien in `docs/` editieren
2. **Preview**: `mkdocs serve` für Live-Vorschau
3. **Testing**: Alle Links und Code-Beispiele prüfen
4. **Deployment**: `mkdocs gh-deploy` für Veröffentlichung

### Code-Dokumentation Standards
```python
# Funktions-Kommentare im Utils-Stil
# Kurze Beschreibung über der Funktion
def function_name(param1, param2):
    # Implementation details
    pass
```

### Versionskontrolle für Dokumentation
```bash
# Dokumentations-Updates committen
git add docs/
git commit -m "docs: Update training documentation with new utils structure"

# Mit semantischen Commit-Messages
git commit -m "docs(setup): Add troubleshooting for pygame issues"
git commit -m "docs(training): Document container-specific parameters"
```

## Code-Qualität und Standards

### Import-Konventionen
```python
# Standard-Imports
import sys
import os
import numpy as np

# Lokale Imports (gruppiert)
from config import ENV_MODE, EPISODES
from envs import GridEnvironment, ContainerShipEnv
from utils import set_all_seeds, load_q_table
```

### Dokumentations-Integration
- **Code-Beispiele**: Alle Code-Blocks sind getestet und funktional
- **Parameter-Referenz**: Zentrale config.py wird in allen Docs referenziert
- **Pfad-Konsistenz**: Alle Pfadangaben entsprechen der neuen Struktur
- **Cross-References**: Links zwischen verwandten Dokumentations-Abschnitten

## Wartung und Updates

### Regelmäßige Aufgaben
- **Code-Beispiele aktualisieren**: Bei Änderungen an der API
- **Screenshots erneuern**: Bei UI/Visualisierungs-Änderungen
- **Performance-Metriken**: Bei Optimierungen oder neuen Features
- **Dependency-Updates**: requirements.txt und mkdocs.yml synchron halten

### Dokumentations-Metriken
- **Vollständigkeit**: Alle Features dokumentiert
- **Aktualität**: Code-Beispiele funktionieren mit aktueller Version
- **Zugänglichkeit**: Verschiedene Nutzergruppen berücksichtigt
- **Konsistenz**: Einheitliche Formatierung und Terminologie

## Integration mit der Projektentwicklung

### Continuous Documentation
```bash
# Bei jedem Feature-Update auch Dokumentation prüfen
git add src/ docs/
git commit -m "feat: Add q-table inspection tool

- New inspect_q_tables.py script
- Interactive analysis options
- Updated documentation in visualisierung.md"
```

### Release-Vorbereitung
1. **Dokumentation vollständig aktualisieren**
2. **Code-Beispiele testen**
3. **Screenshots/Diagramme erneuern**
4. **mkdocs build** ohne Errors
5. **GitHub Pages deployment**

Die Dokumentation ist integraler Bestandteil der Softwarequalität und ermöglicht sowohl neuen Nutzern den Einstieg als auch erfahrenen Entwicklern die effiziente Nutzung des Systems.