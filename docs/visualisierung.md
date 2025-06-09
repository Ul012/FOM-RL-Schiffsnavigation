# Visualisierung und Analyse

## Policy-Darstellung

```bash
python visualize_policy.py
```

Zeigt die gelernte Policy als animierte Pygame-Darstellung mit intuitiven Emojis:

### Grid-Umgebungen (Static, Random Start/Goal/Obstacles)
- 🚢 **Agent/Schiff** - Aktuelle Position
- 🧭 **Start** - Startposition (bei festen Starts)
- 🏁 **Ziel** - Zielposition
- 🪨 **Hindernis** - Nicht passierbare Felder
- ↑→↓← **Policy-Pfeile** - Optimale Aktionen der gelernten Policy

### Container-Umgebung
- 🚢 **Schiff** - Agent-Position
- 📦 **Pickup** - Container-Abholposition
- 🏁 **Dropoff** - Container-Abgabeposition (Ziel)
- 🪨 **Hindernis** - Nicht passierbare Felder

### Interaktive Features
- **Echtzeit-Animation**: Schritt-für-Schritt Bewegung
- **Schritt-Logging**: Ausgabe von Position und Reward
- **Automatischer Screenshot**: Finale Position wird gespeichert
- **Fenster-Steuerung**: Manuelles Schließen oder automatisches Ende

## Policy-Evaluation

```bash
python evaluate_policy.py
```

Führt quantitative Evaluation ohne weitere Lernschritte durch und erstellt:

### Visualisierungen
1. **Erfolgsraten-Balkendiagramm** - Solved vs. Failed Episodes
2. **Reward-Histogramm** - Verteilung der Episode-Rewards mit Durchschnittslinie

### Ausgabe-Metriken
- Detaillierte Ursachen-Verteilung (Ziel erreicht, Timeout, Schleifen, Hindernisse)
- Lösungsrate in Prozent
- Reward-Statistiken (Durchschnitt, Min/Max, Median)

## Szenarien-Vergleich

```bash
python compare_scenarios.py
```

Erstellt automatisch drei umfassende Visualisierungen:

### 1. Erfolgsraten-Vergleich
Balkendiagramm der Erfolgsraten aller Szenarien mit:
- Prozentangaben auf den Balken
- Sortierung nach Performance
- Professionelle Formatierung

### 2. Terminierungsarten-Analyse  
Gestapeltes Balkendiagramm zeigt die Verteilung von:
- **Erfolg** (grün) - Ziel erreicht
- **Timeout** (rot) - Maximale Schritte überschritten
- **Schleifenabbruch** (orange) - Wiederholte Zustandszyklen
- **Hindernis-Kollision** (braun) - Kollision mit Hindernissen

### 3. Statistische Vergleichstabelle
Tabellarische Übersicht mit detaillierten Metriken pro Szenario:
- Erfolgsraten in Prozent
- Verteilung der verschiedenen Terminierungsarten
- Durchschnittliche Rewards und Schrittanzahlen
- Statistische Kennwerte (Standardabweichungen)

## Q-Tabellen-Inspektion

```bash
python inspect_q_tables.py
```

Interaktive Analyse-Tool mit erweiterten Optionen:

### Verfügbare Optionen
1. **Aktuelles Szenario** - Aus config.py ENV_MODE
2. **Spezifisches Szenario** - Manuelle Auswahl
3. **Alle verfügbaren Q-Tabellen** - Komplettanalyse
4. **Formen-Vergleich** - Dimensionen aller Q-Tabellen
5. **Matrix-Darstellung** - Vollständige Q-Werte

### Ausgabe-Informationen
- **Q-Tabellen-Übersicht**: Dimensionen, Min/Max-Werte, Statistiken
- **Gelernte Einträge**: Prozentsatz der Non-Zero Q-Werte
- **Beste Aktionen**: Optimale Policy für erste Zustände
- **Vollständige Matrix**: Formatierte Q-Werte-Ausgabe

## Training-Visualisierungen

Während des Trainings werden automatisch erstellt:

### 1. Lernkurve
- **Raw Rewards**: Episode-Rewards (transparent)
- **Moving Average**: Geglättete Lernkurve (Fenster-Größe konfigurierbar)
- **Trend-Analyse**: Konvergenz-Erkennung

### 2. Erfolgskurve
- **Binäre Erfolgs-Darstellung**: 0/1 pro Episode
- **Moving Average**: Erfolgsrate-Trend
- **Dynamische Fenster-Größe**: Adaptiv basierend auf Episode-Anzahl

### 3. Trainingsstatistiken (4-Panel Übersicht)
- **Reward-Histogramm**: Verteilung mit Durchschnittslinie
- **Kumulative Erfolgsrate**: Laufende Erfolgsquote
- **Recent Reward-Entwicklung**: Aktuelle Episode-Trends
- **Erfolg vs. Misserfolg**: Balkendiagramm mit Prozentangaben

## Export-Funktionen

Alle Visualisierungen werden automatisch als hochqualitative PDFs exportiert:

### Training-Exports
```
exports/
├── train_learning_curve.pdf      # Lernkurve mit Moving Average
├── train_success_curve.pdf       # Erfolgsrate über Zeit
└── train_statistics.pdf          # 4-Panel Statistik-Übersicht
```

### Evaluation-Exports
```
exports/
├── evaluate_policy_success_rate.pdf    # Erfolgsraten-Balkendiagramm
└── evaluate_policy_reward_histogram.pdf # Reward-Verteilung
```

### Vergleich-Exports
```
exports/
├── success_rates.pdf             # Szenarien-Erfolgsraten
└── failure_modes.pdf             # Terminierungsarten-Analyse
```

### Screenshot-Exports
```
exports/
└── agent_final_position.png      # Finale Pygame-Darstellung
```

### Konfiguration
```python
EXPORT_PDF                       # PDF-Export aktivieren (True/False)
EXPORT_PATH                      # Zielordner für Exports
```

## Modulare Visualisierungs-Architektur

Das System nutzt die modulare Struktur aus `utils/visualization.py`:

- **Training-Plots**: Lernkurven, Erfolgsraten, Statistiken
- **Evaluation-Plots**: Policy-Bewertung, Reward-Verteilungen
- **Vergleichs-Plots**: Multi-Szenario Analysen
- **Export-Management**: Einheitliche PDF-Generierung

### Vorteile der modularen Struktur
- **Konsistente Formatierung**: Einheitliches Design aller Plots
- **Wiederverwendbarkeit**: Funktionen in anderen RL-Projekten nutzbar
- **Wartbarkeit**: Änderungen am Plotting-Code nur an einer Stelle
- **Erweiterbarkeit**: Einfache Integration neuer Visualisierungstypen

## Interpretation der Ergebnisse

### Erfolgreiche Policies
- **Hohe Erfolgsrate**: Abhängig vom Szenario (siehe config.py für Ziele)
- **Niedrige Timeout-Rate**: Wenige Episoden erreichen MAX_STEPS
- **Stabile Lernkurven**: Konvergenz ohne starke Schwankungen
- **Konsistente Terminierung**: Primär durch Zielerreichung

### Problematische Policies  
- **Niedrige Erfolgsrate**: Unterhalb der Szenario-spezifischen Erwartungen
- **Hohe Timeout-Rate**: Viele Episoden erreichen MAX_STEPS
- **Instabile Lernkurven**: Starke Schwankungen, keine Konvergenz
- **Schleifenprobleme**: Hoher Anteil von Loop-Abbrüchen

### Container-spezifische Metriken
- **Pickup-Erfolg**: Rate der erfolgreichen Container-Aufnahmen
- **Navigation-Effizienz**: Schritte zwischen Pickup und Dropoff
- **Zustandsraum-Exploration**: Coverage der erweiterten States

### Szenario-abhängige Erwartungen
- **Static**: Sehr hohe Erfolgsraten erwartbar
- **Random Modes**: Moderate Erfolgsraten je nach Randomisierung
- **Container**: Niedrigere Erfolgsraten aufgrund Komplexität

## Performance-Optimierung

### Für große Datenmengen
```python
SHOW_VISUALIZATIONS = False      # Nur PDF-Export, keine interaktiven Plots
EXPORT_PDF = True                # PDFs für spätere Analyse
```

### Für interaktive Analyse
```python
SHOW_VISUALIZATIONS = True       # Interaktive Matplotlib-Fenster
FRAME_DELAY                      # Anpassbare Pygame-Animation (aus config.py)
```

### Für Batch-Processing
- **Automatisierter Modus**: Keine Benutzerinteraktion erforderlich
- **PDF-Only Export**: Schnelle Verarbeitung ohne Display
- **Parallelisierung**: Multi-Szenario Training unterstützt

Das System ermöglicht schnelle visuelle Bewertung der Trainingsqualität, detaillierte wissenschaftliche Analyse und professionelle Präsentation der Ergebnisse.

**Hinweis**: Alle Visualisierungs-Parameter und -Schwellwerte sind in `config.py` konfiguriert und können dort angepasst werden.