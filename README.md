# Simulation von Schiffsnavigation mit Reinforcement Learning

Dieses Projekt zeigt, wie ein Q-Learning-Agent ein Schiff durch eine Gitterwelt navigiert. Die Umgebung basiert auf OpenAI Gymnasium und wird visuell mit Pygame dargestellt. Die Dokumentation erfolgt mit MkDocs.

---

## Inhalte

- Eigene Grid-Umgebung (OpenAI Gym-kompatibel)
- Q-Learning-Implementierung mit Trainingsschleife
- Visualisierung der Policy und des Agentenlaufs
- Dynamische oder statische Hindernisverteilung
- Dokumentation mit MkDocs

---

## Setup

```bash
# Virtuelle Umgebung erstellen und aktivieren
python -m venv venv
venv\Scripts\activate          # oder: source venv/bin/activate

# Abhängigkeiten installieren
pip install -r requirements.txt

# Training starten
cd src
python train.py

# Visualisierung starten
python run_policy.py

# Dokumentation lokal anzeigen
mkdocs serve
```