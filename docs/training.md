# Training und Konfiguration

## Einzelszenario-Training

```bash
cd src
python train.py
```

Das Training verwendet die Parameter aus `config.py` und erstellt:
- Trainierte Q-Tabelle (`q_table_[szenario].npy`)  
- Lernkurven-Diagramme
- Erfolgsstatistiken

## Multi-Szenario-Training

```bash
python train_all_scenarios.py
```

Trainiert automatisch alle Szenarien mit konfigurierbaren Optionen:
- **Visualisierungsmodus**: Interaktiv oder automatisiert
- **Szenario-Auswahl**: Einzeln oder alle
- **Training-Modus**: Sequenziell oder parallel

## Zentrale Parameter

```python
# config.py - Wichtigste Einstellungen
ENV_MODE                     # Szenario-Auswahl
EPISODES                     # Trainings-Episoden
MAX_STEPS                    # Schritte pro Episode
ALPHA                        # Lernrate
GAMMA                        # Diskontfaktor
EPSILON                      # Explorationsrate
```

## Parameter-Kategorien

### Für schnelles Testing
- Weniger Episoden
- Kürzere Episoden
- Höhere Explorationsrate

### Für robuste Ergebnisse
- Mehr Episoden
- Längere Episoden
- Ausgewogene Exploration

### Für Timeout-Analyse
- Sehr kurze Episoden (`MAX_STEPS`)
- Niedrige Schleifenschwelle (`LOOP_THRESHOLD`)

## Szenario-spezifische Besonderheiten

- **Static**: Schnelle Konvergenz, hohe Erfolgsrate
- **Random-Modi**: Langsamere Konvergenz, moderate Erfolgsrate
- **Container**: Komplexer Zustandsraum, längere Trainingszeit

## Trainingsüberwachung

Das System zeigt automatisch:
- Episode-Fortschritt mit Erfolgsrate
- Lernkurven (Raw + Moving Average)
- Terminierungsarten-Verteilung
- Finale Statistiken