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
python -c "import gymnasium, numpy, matplotlib; print('Installation erfolgreich')"
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

Zeigt die gelernte Policy in einer animierten Darstellung.

## Konfiguration

### Zentrale Parameter in config.py
```python
ENV_MODE = "static"           # Szenario-Auswahl
EPISODES = 2000              # Anzahl Trainings-Episoden
MAX_STEPS = 50               # Maximale Schritte pro Episode
ALPHA = 0.1                  # Lernrate
GAMMA = 0.9                  # Diskontierungsfaktor
EPSILON = 0.1                # Explorationsrate
```

### Evaluations-Parameter
```python
EVAL_EPISODES = 500          # Episoden für Szenario-Vergleich
EVAL_MAX_STEPS = 50          # Maximale Schritte bei Evaluation
LOOP_THRESHOLD = 25          # Schwellwert für Schleifenerkennung
```

### Export-Einstellungen
```python
EXPORT_PDF = True            # PDF-Export aktivieren
EXPORT_PATH = "exports/"     # Zielordner für Exports
```

## Fehlerbehebung

### Häufige Probleme

**ModuleNotFoundError bei Gymnasium:**
```bash
pip install gymnasium[classic_control]
```

**Matplotlib-Darstellungsfehler:**
```bash
pip install --upgrade matplotlib
```

**Probleme mit Python 3.12:**
```bash
pip install setuptools wheel
```

### Abhängigkeiten aktualisieren
```bash
pip install --upgrade -r requirements.txt
```

### Neue Abhängigkeiten erfassen
```bash
pip freeze > requirements.txt
```

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