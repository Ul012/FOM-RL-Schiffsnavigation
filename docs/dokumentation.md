# Dokumentation und Entwicklung

## Lokale Dokumentation

```bash
# Entwicklungsmodus starten
mkdocs serve
# Verfügbar unter: http://127.0.0.1:8000

# Statische HTML-Dateien generieren  
mkdocs build
```

## Dokumentationsstruktur

```
docs/
├── index.md              # Projektübersicht
├── setup.md              # Installation und Ausführung  
├── funktionsweise.md     # Systemfunktionalität
├── training.md           # Training und Parameter
├── visualisierung.md     # Analyse-Tools
└── dokumentation.md      # Diese Entwicklungsdokumentation
```

## GitHub Pages

```bash
# Dokumentation veröffentlichen
mkdocs gh-deploy
```

## Toolchain-Details

### Abhängigkeiten
```
mkdocs>=1.5.0
mkdocs-material>=9.0.0
```

### Konfigurationsdatei
Die zentrale Konfiguration erfolgt über `mkdocs.yml` im Projektroot:

```yaml
site_name: Q-Learning Navigation
theme:
  name: material
  features:
    - navigation.tabs
    - navigation.sections
    - toc.integrate

nav:
  - Startseite: index.md
  - Setup: setup.md
  - Funktionsweise: funktionsweise.md
  - Training: training.md
  - Visualisierung: visualisierung.md
  - Entwicklung: dokumentation.md
```