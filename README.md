# 🚢 Simulation von Schiffsnavigation mit Reinforcement Learning

Dieses Projekt zeigt, wie ein Q-Learning-Agent ein Schiff durch eine 5x5-Gitterwelt navigiert. Die Umgebung basiert auf OpenAI Gymnasium und wird visuell mit Pygame dargestellt. Die Dokumentation erfolgt mit MkDocs.

---

## 📌 Inhalte

- Eigene Grid-Umgebung (OpenAI Gym-kompatibel)
- Q-Learning-Implementierung mit Trainingsschleife
- Verschiedene Umgebungsmodi:
  - Fester Start, Ziel und Hindernisse (`static`)
  - Zufällige Start-/Zielpositionen (`random_goal`)
  - Zufällige Hindernisse (`random_obstacles`)
  - Pickup/Dropoff-Szenario mit Containertransport (`pickup_dropoff`)
- Schleifen- und Timeout-Erkennung zur Stabilisierung
- Visualisierung der Policy und Agentenläufe mit Pygame
- Optionaler GIF- und PDF-Export zur Ergebnisdokumentation
- **Evaluation**: Zielerreichung, Reward, Schleifenabbrüche mit `evaluate_policy.py`
- Dokumentation mit MkDocs

---

## ⚙️ Setup

### 1. Virtuelle Umgebung erstellen und aktivieren
```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux
```

### 2. Abhängigkeiten installieren
```bash
pip install -r requirements.txt
```

Falls Probleme mit `distutils` auftreten (Python 3.12):
```bash
pip install setuptools wheel
```

---

## 🚀 Ausführung

### 3. Training starten
```bash
cd src
python train.py
```

### 4. Policy visualisieren
```bash
python visualize_policy.py
```
- Darstellung mit 🧭 Start, 🏁 Ziel, 🪨 Hindernissen
- Export von GIF/PDF durch `EXPORT_FRAMES = True`

### 5. Evaluation durchführen
```bash
python evaluate_policy.py
```
- Zielerreichungen, durchschnittliche Rewards, Schleifenabbrüche
- Robuste Erfolgserkennung mit `reward >= 10`
- Automatisches Laden der passenden Q-Tabelle je nach Modus

---

## 📚 Dokumentation (lokal)

```bash
mkdocs serve
```
Danach erreichbar unter [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🔀 Modus-Übersicht

| Modus         | Beschreibung                                                                 |
|---------------|------------------------------------------------------------------------------|
| `static`      | Fester Start-, Ziel- und Hindernisbereich. Gut zum Einstieg, stabile Lernkurve. |
| `random_start`| Zufälliger Startpunkt bei festem Ziel. Testet Robustheit beim Startverhalten. |
| `random_goal` | Fester Startpunkt, aber zufälliges Ziel. Erfordert flexible Zielnavigation.   |
| `random_obstacles` | Hindernisse variieren pro Episode. Erhöht Unsicherheit, erschwert Lernen. |
| `container`   | Zwei-Ziel-Aufgabe: Container muss zuerst abgeholt, dann zum Ziel gebracht werden. |

---

## 📁 Projektstruktur

```text
FOM-rl-shipnav-qlearning/
├── src/
│   ├── train.py                    ← Q-Learning Training
│   ├── evaluate_policy.py          ← Auswertung (Erfolg, Reward, Schleifen)
│   ├── visualize_policy.py         ← Visualisierung mit Pygame
│   ├── q_table.py                  ← Initialisierung und Speicherung von Q-Tabellen
│   ├── config.py                   ← Konfiguration der Modi
│   └── navigation/
│       ├── __init__.py
│       ├── environment/
│       │   ├── __init__.py
│       │   ├── grid_environment.py     ← Basisumgebung
│       │   └── container_environment.py← Pickup/Dropoff-Modus
├── q_table.npy                     ← Beispielhafte gespeicherte Q-Tabelle
├── requirements.txt
├── README.md
├── mkdocs.yml
├── .gitignore
└── docs/
    ├── index.md                    ← Startseite der MkDocs-Dokumentation
    └── setup.md                    ← Installationsanleitung
```