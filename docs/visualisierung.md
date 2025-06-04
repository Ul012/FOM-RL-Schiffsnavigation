# Visualisierung der Policy

Die Datei `visualize_policy.py` zeigt die Policy des Agenten als animierten Lauf in der Umgebung.

## Darstellungsmerkmale

- 🧭 Startposition
- 🚢 Aktuelle Agentenposition (wird bei jedem Schritt aktualisiert)
- 🏁 Zielposition
- 🪨 Hindernisse
- 📤 Pickup-Punkt (Container abholen)
- 📦 Dropoff-Punkt (Container abliefern)

Diese Emojis helfen bei der intuitiven Interpretation des Agentenverhaltens.

## Export

Wenn `EXPORT_FRAMES = True` gesetzt ist, erzeugt das Skript:

- eine animierte GIF-Datei des Agentenlaufs
- einen Screenshot der Zielerreichung als PDF