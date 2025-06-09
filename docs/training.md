# Training und Konfiguration

## Einzelszenario-Training

```bash
cd src
python train.py
```

Das Training verwendet die Parameter aus `config.py` und erstellt:
- Trainierte Q-Tabelle (`q_table_[szenario].npy`)  
- Lernkurven-Diagramme mit Moving Average
- Erfolgsstatistiken und Trainings-Metriken
- PDF-Exports (falls `EXPORT_PDF = True`)

### Ausgabe-Beispiel
```
Starte Training mit [EPISODES] Episoden...
Hyperparameter: α=[ALPHA], γ=[GAMMA], ε=[EPSILON], Seed=[SEED]
Episode 50/[EPISODES]: Reward=X.XX, Steps=XX, Erfolgsrate (letzte 50): XX.X%
...
Q-Tabelle gespeichert: q_table_[env_mode].npy
```

## Multi-Szenario-Training

```bash
python train_all_scenarios.py
```

Trainiert automatisch alle Szenarien mit konfigurierbaren Optionen:
- **Visualisierungsmodus**: Interaktiv oder automatisiert
- **Szenario-Auswahl**: Einzeln oder alle
- **Training-Modus**: Sequenziell oder parallel

### Verfügbare Szenarien
| Szenario | ENV_MODE | Emojis | Besonderheit |
|----------|----------|---------|--------------|
| Statisch | `"static"` | 🧭🏁🪨 | Schnelle Konvergenz |
| Zufälliger Start | `"random_start"` | 🚢🏁🪨 | Variable Startposition |
| Zufälliges Ziel | `"random_goal"` | 🧭🏁🪨 | Adaptive Navigation |
| Zufällige Hindernisse | `"random_obstacles"` | 🧭🏁🪨 | Dynamische Umgebung |
| Container | `"container"` | 🚢📦🏁 | Pickup/Dropoff-Aufgabe |

### Interaktive Optionen
Das Script bietet Benutzerinteraktion für:
- Szenario-Auswahl (einzeln oder alle)
- Visualisierungsmodus (interaktiv/automatisiert)
- Training-Modus (sequenziell/parallel)

## Zentrale Parameter

**Alle Werte werden in `config.py` konfiguriert:**

```python
ENV_MODE                     # Szenario-Auswahl
EPISODES                     # Trainings-Episoden
MAX_STEPS                    # Schritte pro Episode
ALPHA                        # Lernrate
GAMMA                        # Diskontfaktor
EPSILON                      # Explorationsrate
SEED                         # Reproduzierbarkeit
```

### Reward-System
**Definiert in `config.py` unter `REWARDS`:**
- **step**: Bewegungskosten (typisch negativ)
- **goal**: Ziel erreicht (positiv)
- **obstacle**: Hindernis-Kollision (negativ)
- **loop_abort**: Schleifenabbruch (negativ)
- **timeout**: Episode-Timeout (negativ)
- **pickup**: Container aufgenommen (positiv)
- **dropoff**: Container abgeliefert (höchste positive Belohnung)

### Schleifenerkennung
```python
LOOP_THRESHOLD               # Wiederholungen bis Abbruch
```

### Export-Einstellungen
```python
EXPORT_PDF                   # PDF-Export aktivieren
EXPORT_PATH                  # Zielordner für Exports
```

## Parameter-Kategorien

### Für schnelles Testing
- **Weniger Episoden**: Schnelle Iterationen
- **Kürzere Episoden**: Reduzierte `MAX_STEPS`
- **Höhere Explorationsrate**: Erhöhtes `EPSILON`

### Für robuste Ergebnisse
- **Mehr Episoden**: Stabile Konvergenz
- **Längere Episoden**: Ausreichend Zeit für komplexe Pfade
- **Ausgewogene Exploration**: Moderates `EPSILON`

### Für Timeout-Analyse
- **Sehr kurze Episoden**: Niedrige `MAX_STEPS`
- **Niedrige Schleifenschwelle**: Reduzierte `LOOP_THRESHOLD`

### Für Container-Szenario
- **Erweiterte Episoden**: Höhere Episode- und Schritt-Anzahl
- **Angepasste Exploration**: Container-Zustandsraum berücksichtigen

## Typische Parameter-Bereiche

### Lernrate (ALPHA)
- **Schnelles Lernen**: 0.2-0.5 (kann instabil werden)
- **Stabiles Lernen**: 0.05-0.2 (empfohlener Bereich)
- **Konservativ**: 0.01-0.05 (sehr langsam aber stabil)

### Diskontfaktor (GAMMA)
- **Kurzsichtig**: 0.8-0.9 (sofortige Belohnungen bevorzugt)
- **Ausgewogen**: 0.9-0.95 (typischer Bereich)
- **Weitsichtig**: 0.95-0.99 (langfristige Planung)

### Exploration (EPSILON)
- **Hohe Exploration**: 0.2-0.5 (für unbekannte Umgebungen)
- **Moderate Exploration**: 0.05-0.2 (ausgewogenes Lernen)
- **Geringe Exploration**: 0.01-0.05 (near-greedy)

## Szenario-spezifische Eigenschaften

### Static Environment
- **Konvergenz**: Schnell (relative wenige Episoden)
- **Erfolgsrate**: Sehr hoch (>90% bei guten Parametern)
- **Besonderheit**: Deterministische, ideale Lernumgebung

### Random Start/Goal/Obstacles
- **Konvergenz**: Mittel (mehr Episoden erforderlich)
- **Erfolgsrate**: Moderat (abhängig von Randomisierung)
- **Besonderheit**: Robustheitstesting der Policy

### Container Environment
- **Konvergenz**: Langsam (deutlich mehr Episoden)
- **Erfolgsrate**: Herausfordernd (abhängig von Pickup/Dropoff-Komplexität)
- **Besonderheit**: Erweiterte Zustandsräume mit mehreren Teilzielen

## Trainingsüberwachung

### Automatische Ausgaben
Das System zeigt in Echtzeit:
- Episode-Fortschritt mit aktueller Erfolgsrate
- Lernkurven (Raw Rewards + Moving Average)
- Terminierungsarten-Verteilung
- Finale Trainingsstatistiken

### Generierte Visualisierungen
1. **Lernkurve**: Rohe Belohnungen + Moving Average
2. **Erfolgskurve**: Zielerreichung pro Episode
3. **Trainingsstatistiken**: 4-Panel Übersichtsgrafik
   - Reward-Histogramm
   - Kumulative Erfolgsrate
   - Reward-Entwicklung (aktuelle Episoden)
   - Erfolg vs. Misserfolg

### Export-Dateien
**Automatisch generiert (falls `EXPORT_PDF = True`):**
- `train_learning_curve.pdf`
- `train_success_curve.pdf`
- `train_statistics.pdf`

## Modulare Utils-Architektur

Das Training nutzt modulare Utils-Komponenten für maximale Wiederverwendbarkeit:

- **utils/common.py**: Basis-Funktionen
- **utils/qlearning.py**: Q-Learning Algorithmus  
- **utils/visualization.py**: Plotting-Funktionen
- **utils/environment.py**: Umgebungs-Management

### Vorteile der modularen Struktur
- **DRY-Prinzip**: Keine Code-Duplikation
- **Wartbarkeit**: Änderungen nur an einer Stelle
- **Wiederverwendbarkeit**: Utils in anderen RL-Projekten nutzbar
- **Testbarkeit**: Einzelne Module isoliert testbar

## Reproduzierbarkeit

### Seed-Management
**Zentrale Seed-Kontrolle** über `config.py`:
- Deterministisches Verhalten bei gleichem Seed
- Reproduzierbare Experimente
- Vergleichbare Ergebnisse zwischen Läufen

### Parameter-Logging
Alle relevanten Hyperparameter werden automatisch protokolliert für spätere Reproduktion.

## Troubleshooting

### Häufige Probleme
- **Niedrige Erfolgsrate**: Parameter anpassen oder mehr Episoden
- **Langsame Konvergenz**: Lernrate oder Exploration adjustieren
- **Memory-Probleme**: Episode-Anzahl reduzieren

### Performance-Optimierung
- **Paralleles Training**: Multi-Szenario mit parallel-Option
- **Reduzierte Visualisierung**: `EXPORT_PDF = False`
- **Batch-Processing**: Automatisierter Modus ohne interaktive Plots

**Hinweis**: Alle konkreten Parameter-Werte finden Sie in der aktuellen `config.py` - diese Dokumentation beschreibt die Konzepte und Bereiche.