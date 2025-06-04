# Setup & Ausführung

## Lokale Ausführung

```bash
# Virtuelle Umgebung aktivieren
rl-venv\Scripts\activate

# Installieren der Abhängigkeiten
pip install -r requirements.txt

# Trainingsskript starten
cd src
python train.py
```

## Evaluation

```bash
python evaluate_policy.py
```

- Gibt Zielerreichungen, Rewards und Schleifenabbrüche aus
- Funktioniert mit verschiedenen ENV_MODE-Einstellungen

## Visualisierung

```bash
python visualize_policy.py
```

- Zeigt Agentenlauf in Echtzeit
- Emojis für Start (🧭), Ziel (🏁), Hindernis (🪨), Pickup (📤), Dropoff (📦)
- Optionaler GIF- und PDF-Export durch `EXPORT_FRAMES = True`

## Dokumentation lokal aufrufen

```bash
mkdocs serve
```

Dann im Browser öffnen: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## Abhängigkeiten aktualisieren

Wenn du bereits Pakete installiert hast und die `requirements.txt` auf den aktuellen Stand bringen möchtest:

```bash
pip freeze > requirements.txt
```

Wenn du alle Pakete aus der `requirements.txt` auf die neuesten kompatiblen Versionen aktualisieren möchtest:

```bash
pip install --upgrade -r requirements.txt
```