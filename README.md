# Simulation von Schiffsnavigation mit Reinforcement Learning

Dieses Projekt zeigt, wie ein Q-Learning-Agent ein Schiff durch eine Gitterwelt navigiert. Die Umgebung basiert auf OpenAI Gymnasium und wird visuell mit Pygame dargestellt. Die Dokumentation erfolgt mit MkDocs.

---

## Inhalte

- Eigene Grid-Umgebung (OpenAI Gym-kompatibel)
- Q-Learning-Implementierung mit Trainingsschleife
- Visualisierung der Policy und des Agentenlaufs
- Dynamische oder statische Hindernisverteilung
- Optionaler GIF- und PDF-Export zur Ergebnisdokumentation
- Dokumentation mit MkDocs

---

## Setup

### 1. Virtuelle Umgebung erstellen und aktivieren

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate         # macOS/Linux
```

### 2. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

Falls Probleme mit `distutils` auftreten (Python 3.12), installiere zusätzlich manuell:

```bash
pip install setuptools wheel
```

### 3. Training starten

```bash
cd src
python train.py
```

### 4. Visualisierung (Policy-Demo mit Pygame)

```bash
python visualize_policy.py
```

Optional kannst du in `visualize_policy.py` den Parameter `EXPORT_FRAMES = True` setzen, um eine GIF- und PDF-Ausgabe zu erhalten.

### 5. Dokumentation lokal anzeigen

```bash
mkdocs serve
```

Die Seite ist danach unter [http://127.0.0.1:8000](http://127.0.0.1:8000) erreichbar.
